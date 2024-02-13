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
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json";

            bool useChat = false; // Turn on for chat functionality or off to just generate synthetic data
            bool useDatabaseForChat = false; // Turn on to use vector databases for response
            int numSyntheticDataToGenerate = 256; // Increase above 0 to generate more incident reports
            int numTopMatches = 3; // This is when querying the database of facts

            string[] systemMessages = { $"You are a chatbot who needs to solve the user's query by giving many detailed steps" };

            uint seed = 1;
            uint contextSize = 4096;
            int maxTokens = 4096;
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Adjust as needed
            
            string[] antiPrompts = { "<end>" };
            
            IInputHandler inputHandler = new ConsoleInputHandler();

            // Initialize ModelLoader and load model
            ModelManagerConsole modelManager = new ModelManagerConsole(modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            ModelManagerOutputs modelLoaderOutputs = await modelManager.InitializeAsync(inputHandler);

            DatabaseManager databaseManager = new DatabaseManager(databaseJsonPath, modelManager);
            await databaseManager.InitializeDatabaseAsync();
            ConversationManager conversationManager = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, maxTokens, temperature, systemMessages, antiPrompts, numTopMatches);
            SyntheticDataGenerator syntheticDataGenerator = new SyntheticDataGenerator(modelManager, databaseManager, conversationManager);


            if (numSyntheticDataToGenerate > 0)
            {
                syntheticDataGenerator.GenerateITDataPipeline(numSyntheticDataToGenerate, databaseJsonPath).Wait();
                if (useChat)
                {
                    await conversationManager.StartChatAsync(useDatabaseForChat);
                }
            }
            else if (useChat)
            {
                await conversationManager.StartChatAsync(useDatabaseForChat);
            }
        }
    }
}
