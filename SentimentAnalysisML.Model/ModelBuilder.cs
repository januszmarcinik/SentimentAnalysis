using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalysisML.Model
{
    public class ModelBuilder
    {
        private readonly string _trainDataFilePath;
        private readonly string _modelFilePath;
        private readonly MLContext _mlContext;

        public ModelBuilder(string trainDataFilePath, string modelFilePath)
        {
            _trainDataFilePath = trainDataFilePath;
            _modelFilePath = modelFilePath;
            _mlContext = new MLContext(seed: 1);
        }
        
        public void CreateModel<TModelInput>() where TModelInput : class, IModelInput
        {
            var trainingDataView = _mlContext.Data.LoadFromTextFile<TModelInput>(
                                            path: _trainDataFilePath,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: false,
                                            allowSparse: false);

            var trainingPipeline = BuildTrainingPipeline();

            var mlModel = TrainModel(trainingDataView, trainingPipeline);

            Evaluate(trainingDataView, trainingPipeline);

            SaveModel(mlModel, trainingDataView.Schema);
        }

        private IEstimator<ITransformer> BuildTrainingPipeline()
        {
            var dataProcessPipeline = _mlContext.Transforms.Conversion
                .MapValueToKey("Sentiment", "Sentiment")
                .Append(_mlContext.Transforms.Text.FeaturizeText("SentimentText_tf", "SentimentText"))
                .Append(_mlContext.Transforms.CopyColumns("Features", "SentimentText_tf"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(_mlContext);
            
            var trainer = _mlContext.MulticlassClassification.Trainers
                .OneVersusAll(_mlContext.BinaryClassification.Trainers
                    .AveragedPerceptron(labelColumnName: "Sentiment", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Sentiment")
                .Append(_mlContext.Transforms.Conversion
                    .MapKeyToValue("PredictedLabel", "PredictedLabel"));

            return dataProcessPipeline.Append(trainer);
        }

        private static ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            var model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private void Evaluate(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = _mlContext.MulticlassClassification
                .CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Sentiment");
            PrintMultiClassClassificationFoldsAverageMetrics(crossValidationResults);
        }

        private void SaveModel(ITransformer mlModel, DataViewSchema modelInputSchema)
        {
            Console.WriteLine("=============== Saving the model  ===============");
            _mlContext.Model.Save(mlModel, modelInputSchema, _modelFilePath);
            Console.WriteLine("The model is saved to {0}", _modelFilePath);
            Console.WriteLine("=============== End of saving model ===============");
        }

        private static void PrintMultiClassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics).ToList();

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy).ToList();
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy).ToList();
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss).ToList();
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction).ToList();
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static double CalculateStandardDeviation(IList<double> values)
        {
            var average = values.Average();
            var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        private static double CalculateConfidenceInterval95(IList<double> values) => 
            1.96 * CalculateStandardDeviation(values) / Math.Sqrt(values.Count - 1);
    }
}
