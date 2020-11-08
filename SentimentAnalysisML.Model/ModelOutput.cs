using Microsoft.ML.Data;

namespace SentimentAnalysisML.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
        public float[] Score { get; set; }
    }
}
