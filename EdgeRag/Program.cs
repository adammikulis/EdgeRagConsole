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
            string modelDirectoryPath = @"C:/ai/models";
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json";

            bool useChat = false; // Turn on for chat functionality or off to just generate synthetic data
            bool useDatabaseForChat = true; // Turn on to use vector databases for response (automatically turns on if generating data)
            int numSyntheticDataToGenerate = 3; // Set to 0 for normal chat, forces database usage if above 0
            int numTopMatches = 3; // This is when querying the database of facts

            string[] systemMessage = { $"You are a chatbot who needs to solve the user's query by giving many detailed steps" };

            uint seed = 1;
            uint contextSize = 4096;
            int maxTokens = 256;
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Adjust as needed
            
            string[] antiPrompts = { "<end>" };
            
            IInputHandler inputHandler = new ConsoleInputHandler();

            // Initialize ModelLoader and load model
            ModelLoaderConsole modelLoader = new ModelLoaderConsole(modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            ModelLoaderOutputs modelLoaderOutputs = await modelLoader.InitializeAsync(inputHandler);

            // Initialize DatabaseManager if useDatabase is true
            DatabaseManager databaseManager = new DatabaseManager(databaseJsonPath, modelLoaderOutputs.embedder, modelLoaderOutputs.modelType);
            await databaseManager.InitializeDatabaseAsync();
            ConversationManager conversationManager = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, useDatabaseForChat, maxTokens, temperature, antiPrompts, numTopMatches);
            SyntheticDataGenerator syntheticDataGenerator = new SyntheticDataGenerator(databaseManager, conversationManager, maxTokens, antiPrompts);


            if (numSyntheticDataToGenerate > 0)
            {
                syntheticDataGenerator.GenerateITDataPipeline(numSyntheticDataToGenerate, databaseJsonPath).Wait();
                if (useChat)
                {
                    await conversationManager.StartChatAsync(systemMessage[0], "");
                }
            }
            else if (useChat)
            {
                await conversationManager.StartChatAsync(systemMessage[0], "");
            }
        }
    }
}
