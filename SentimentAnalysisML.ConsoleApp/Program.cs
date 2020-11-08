using System;
using System.IO;
using SentimentAnalysisML.Model;

namespace SentimentAnalysisML.ConsoleApp
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainDataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "train-data.txt");
            var modelFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "MLModel.zip");

            // Create single instance of sample data from first line of dataset for model input
            var sampleData = new ModelInput
            {
                SentimentText = @"I love it it's perfect",
            };
            
            var modelBuilder = new ModelBuilder(trainDataFilePath, modelFilePath);
            modelBuilder.CreateModel();

            // Make a single prediction on the sample data and print results
            var predictionResult = new ConsumeModel(modelFilePath).Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Sentiment with predicted Sentiment from sample data...\n\n");
            Console.WriteLine($"SentimentText: {sampleData.SentimentText}");
            Console.WriteLine($"\n\nPredicted Sentiment value {predictionResult.Prediction} \nPredicted Sentiment scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
