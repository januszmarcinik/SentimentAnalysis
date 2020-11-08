using Microsoft.ML.Data;

namespace SentimentAnalysisML.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float[] Score { get; set; }
    }
}
