using LLama;
using System;
using System.Threading.Tasks;

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string modelDirectoryPath = @"C:/ai/models";
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json";

            bool useDatabase = true; // Turn on to use vector databases for response (automatically turns on if generating data)
            bool useChat = false; // Turn on for chat functionality or off to just generate synthetic data
            int numSyntheticDataToGenerate = 3; // Set to 0 for normal chat, forces database usage if above 0
            int numTopMatches = 3; // This is when querying the database of facts

            string[] systemMessage = { $"You are a chatbot who needs to solve the user's query by giving many detailed steps" };

            uint contextSize = 4096;
            int maxTokens = 256;
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Adjust as needed
            string[] antiPrompts = { "<endtoken>" };
            
            IInputHandler inputHandler = new ConsoleInputHandler();

            // Initialize ModelLoader and load model
            ModelLoaderConsole modelLoader = new ModelLoaderConsole(modelDirectoryPath, contextSize, numGpuLayers, numCpuThreads);
            ModelLoaderOutputs modelLoaderOutputs = await modelLoader.InitializeAsync(inputHandler);

            // Initialize DatabaseManager if useDatabase is true
            ConversationManagerConsole conversationManager = null;
            DatabaseManager databaseManager = null;

            if (numSyntheticDataToGenerate > 0) useDatabase = true;

            if (useDatabase)
            {
                databaseManager = new DatabaseManager(databaseJsonPath, modelLoaderOutputs.embedder, modelLoaderOutputs.modelType);
                await databaseManager.InitializeDatabaseAsync();
                // Initialize ConversationLoader with DatabaseManager if needed
                conversationManager = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, maxTokens, temperature, antiPrompts, numTopMatches);

                // Synthetic data generation or start chat based on numSyntheticDataToGenerate
                if (numSyntheticDataToGenerate > 0)
                {
                    conversationManager.syntheticDataGenerator.GenerateITDataPipeline(numSyntheticDataToGenerate, databaseJsonPath).Wait();
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
            else if (useChat)
            {
                conversationManager = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, maxTokens, temperature, antiPrompts, numTopMatches);
                await conversationManager.StartChatAsync(systemMessage[0], "");
            }
        }
    }
}
