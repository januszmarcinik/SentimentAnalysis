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

            var modelBuilder = new ModelBuilder(trainDataFilePath, modelFilePath);
            modelBuilder.CreateModel<ModelInput>();
            
            var consumeModel = new ConsumeModel<ModelInput>(modelFilePath);
            
            while (true)
            {
                Console.Write("\n\nHello from the sentimental analysis program. Enter the message and check it's sentiment. (enter 'exit' to end program) \n\n\n> ");
                
                var message = Console.ReadLine();
                if (message == "exit")
                {
                    break;
                }
                
                Predict(consumeModel, message);
            }
            
            Console.WriteLine("\n\n=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static void Predict(ConsumeModel<ModelInput> consumeModel, string message)
        {
            var sampleData = new ModelInput
            {
                SentimentText = message
            };
            
            var predictionResult = consumeModel.Predict(sampleData);

            Console.Write("\n\nPredicted Sentiment value: ");
            PrintSentiment(predictionResult.Prediction);
            Console.WriteLine($"Predicted Sentiment scores: [{string.Join(", ", predictionResult.Score)}]");
        }

        private static void PrintSentiment(bool value)
        {
            if (value)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Positive");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Negative");
            }
            
            Console.ResetColor();
        }
    }
}
