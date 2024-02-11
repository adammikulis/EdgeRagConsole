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
            uint contextSize = 4096;
            bool useDatabase = false; // Assuming you want to use the database now
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Adjust as needed
            int numSyntheticDataToGenerate = 1; // Set to 0 for normal chat, forces database usage if above 0
            string syntheticDataOutputDirectory = "C:/ai/data/synthetic";
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json"; // Path to your JSON database
            string[] antiPrompts = new string[] { "<endtoken>" };
            int numTopMatches = 3; // This is when querying the database of facts

            IInputHandler inputHandler = new ConsoleInputHandler();

            // Initialize ModelLoader and load model
            ModelLoaderConsole modelLoader = new ModelLoaderConsole(modelDirectoryPath, contextSize, numGpuLayers, numCpuThreads);
            ModelLoaderOutputs modelLoaderOutputs = await modelLoader.InitializeAsync(inputHandler);

            // Initialize DatabaseManager if useDatabase is true
            ConversationManagerConsole conversationLoader = null;
            DatabaseManager databaseManager = null;

            if (useDatabase || numSyntheticDataToGenerate > 0)
            {
                databaseManager = new DatabaseManager(databaseJsonPath, modelLoaderOutputs.embedder, modelLoaderOutputs.modelType);
                // Initialize ConversationLoader with DatabaseManager if needed
                conversationLoader = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, temperature, antiPrompts, numTopMatches);

                // Synthetic data generation or start chat based on numSyntheticDataToGenerate
                if (numSyntheticDataToGenerate > 0)
                {
                    conversationLoader.syntheticDataGenerator.GenerateITDataPipeline(numSyntheticDataToGenerate, syntheticDataOutputDirectory).Wait();
                    await conversationLoader.StartChatAsync("", "");
                }
                else
                {
                    await conversationLoader.StartChatAsync("", "");
                }
            }
            else
            {
                conversationLoader = new ConversationManagerConsole(inputHandler, modelLoaderOutputs, databaseManager, temperature, antiPrompts, numTopMatches);
                await conversationLoader.StartChatAsync("", "");
            }
        }
    }
}
