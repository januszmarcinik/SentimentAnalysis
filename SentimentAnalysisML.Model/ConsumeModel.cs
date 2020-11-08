using Microsoft.ML;

namespace SentimentAnalysisML.Model
{
    public class ConsumeModel<TModelInput> where TModelInput : class, IModelInput
    {
        private readonly string _modelFilePath;
        private PredictionEngine<TModelInput, ModelOutput> _predictionEngine;

        public ConsumeModel(string modelFilePath)
        {
            _modelFilePath = modelFilePath;
        }
        
        public ModelOutput Predict(TModelInput input) => 
            PredictionEngine.Predict(input);

        private PredictionEngine<TModelInput, ModelOutput> PredictionEngine => 
            _predictionEngine ?? (_predictionEngine = CreatePredictionEngine());

        private PredictionEngine<TModelInput, ModelOutput> CreatePredictionEngine()
        {
            var mlContext = new MLContext();
            var mlModel = mlContext.Model.Load(_modelFilePath, out _);
            return mlContext.Model.CreatePredictionEngine<TModelInput, ModelOutput>(mlModel);
        }
    }
}
