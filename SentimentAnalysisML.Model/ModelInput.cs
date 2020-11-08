using Microsoft.ML.Data;

namespace SentimentAnalysisML.Model
{
    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName("Sentiment")]
        public string Sentiment { get; set; }

        [LoadColumn(1)]
        [ColumnName("SentimentText")]
        public string SentimentText { get; set; }
    }
}
