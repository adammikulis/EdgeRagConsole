// Note: All data has been generated/tested with istral-7b-instruct-v0.2.Q8_0.gguf for consistency
// Use mistral-7b-instruct-v0.2.Q8_0.gguf for best results

using LLama;
using System;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string modelDirectoryPath = @"C:/ai/models"; // Change this path if you have models saved elsewhere
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json"; // Change this path if you would like to save the data elsewhere
            int numTopMatches = 3; // This is when querying the database of facts

            string[] systemMessages = { $"You are a chatbot who needs to solve the user's query by giving many detailed steps" };

            uint seed = 1;
            uint contextSize = 4096;
            int maxTokens = 4096;
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Lower is more deterministic, higher is more random
            
            string[] antiPrompts = { "<end>" }; // This is what the LLM emits to stop the message
            int numStars = 50; // This is for rendering the menu

            IOManager iOManager = await IOManager.CreateAsync(numStars);
            iOManager.OnOutputMessage += Console.Write;

            // Initialize ModelLoader and load model
            ModelManager modelManager = await ModelManager.CreateAsync(iOManager, modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            DatabaseManager databaseManager = await DatabaseManager.CreateAsync(iOManager, modelManager, databaseJsonPath, numTopMatches);
            ConversationManager conversationManager = await ConversationManager.CreateAsync(iOManager, modelManager, databaseManager, maxTokens, temperature, systemMessages, antiPrompts);
            SyntheticDataGenerator syntheticDataGenerator = await SyntheticDataGenerator.CreateAsync(iOManager, modelManager, databaseManager, conversationManager);

            // Basic menu loop
            while (true)
            {
                Console.WriteLine("\nMenu:");
                Console.WriteLine("1. Chat");
                Console.WriteLine("2. Chat using Database");
                Console.WriteLine("3. Generate Questions and Chat using Database");
                Console.WriteLine("4. Generate Questions and Quit");
                Console.WriteLine("5. Quit");
                Console.Write("Select an option: ");
                string option = Console.ReadLine();

                switch (option)
                {
                    case "1":
                        await conversationManager.StartChatAsync(false);
                        break;
                    case "2":
                        await conversationManager.StartChatAsync(true);
                        break;
                    case "3":
                        Console.Write("Enter the number of questions to generate: ");
                        int numQuestions = Convert.ToInt32(Console.ReadLine());
                        await syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                        await conversationManager.StartChatAsync(true);
                        break;
                    case "4":
                        Console.Write("Enter the number of questions to generate: ");
                        numQuestions = Convert.ToInt32(Console.ReadLine());
                        await syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                        return;
                    case "5":
                        return;
                    default:
                        Console.WriteLine("Invalid option, please try again.");
                        break;
                }
            }
        }
    }
}
