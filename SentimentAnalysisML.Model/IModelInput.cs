namespace SentimentAnalysisML.Model
{
    public interface IModelInput
    {
        string Sentiment { get; set; }

        string SentimentText { get; set; }
    }
}