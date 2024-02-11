// Note: Use a Mistral 0.2v 7B model for best results. Llama2 is inconsistent, Mixtral loads but gives nonsense results.

using LLama;
using System.Data;

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string directoryPath = @"C:\ai\models";
            string[] facts = new string[] { "" };
            string[] antiPrompts = new string[] { "<endtoken>" };
            uint contextSize = 4096;

            IInputHandler inputHandler = new ConsoleInputHandler(); // The intention is to use this for Godot as well as C# console apps, requiring different input
            bool useDatabase = false;
            int numGpuLayers = 33; // Set to 0 for cpu-only. If you get a CUDA error, lower numGpuLayers to use less VRAM
            uint numCpuThreads = 8;
            float temperature = 0.5f; // 0.0 is completely deterministic (can get repetitive), =>1.0 is much more random
            int numSyntheticDataToGenerate = 1; // Set to 0 for normal chat

            string syntheticDataOutputPath = @"C:\ai\data\synthetic";
            
            ModelLoaderConsole modelLoader = new ModelLoaderConsole(directoryPath, facts, contextSize, numGpuLayers, numCpuThreads, useDatabase);
            ModelLoaderOutputs modelLoaderOutputs = await modelLoader.InitializeAsync(inputHandler);
            ConversationLoaderConsole conversationLoader = new ConversationLoaderConsole(inputHandler, modelLoaderOutputs, temperature, useDatabase, antiPrompts);

            if (numSyntheticDataToGenerate > 0)
            {
                conversationLoader.syntheticDataGenerator.GenerateITDataPipeline(numSyntheticDataToGenerate).Wait();
                conversationLoader.syntheticDataGenerator.PrintSyntheticDataTable(numSyntheticDataToGenerate);
            }
            else
            {
                await conversationLoader.StartChatAsync("","");
            }
        }
    }
}