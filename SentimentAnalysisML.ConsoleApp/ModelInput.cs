using Microsoft.ML.Data;
using SentimentAnalysisML.Model;

namespace SentimentAnalysisML.ConsoleApp
{
    public class ModelInput : IModelInput
    {
        [LoadColumn(1)]
        [ColumnName("Sentiment")]
        public bool Sentiment { get; set; }

        [LoadColumn(0)]
        [ColumnName("SentimentText")]
        public string SentimentText { get; set; }
    }
}
