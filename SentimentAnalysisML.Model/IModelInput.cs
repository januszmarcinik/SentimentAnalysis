namespace SentimentAnalysisML.Model
{
    public interface IModelInput
    {
        bool Sentiment { get; set; }

        string SentimentText { get; set; }
    }
}