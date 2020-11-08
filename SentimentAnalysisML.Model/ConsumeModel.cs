using Microsoft.ML;

namespace SentimentAnalysisML.Model
{
    public class ConsumeModel
    {
        private readonly string _modelFilePath;
        private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        public ConsumeModel(string modelFilePath)
        {
            _modelFilePath = modelFilePath;
        }
        
        public ModelOutput Predict(ModelInput input) => 
            PredictionEngine.Predict(input);

        private PredictionEngine<ModelInput, ModelOutput> PredictionEngine => 
            _predictionEngine ?? (_predictionEngine = CreatePredictionEngine());

        private PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {
            var mlContext = new MLContext();
            var mlModel = mlContext.Model.Load(_modelFilePath, out _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }
    }
}
