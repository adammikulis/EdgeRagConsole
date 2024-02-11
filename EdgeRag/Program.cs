using LLama;

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string directoryPath = @"C:\ai\models";
            string[] facts = new string[] {
            "The University of Denver is a private University that is abbreviated as 'DU'",
            "DU founded in 1864 as the Colorado Seminary",
            "DU is a private R1 University",
            "The mascot of the University of Denver is the Pioneer",
            "DU is located in south Denver, Colorado in the University neighborhood",
            "DU's has a secondary/satellite campus, the 720 acre Kennedy Mountain Campus which is located 110 miles northwest of Denver",
            "DU has 5700 undergraduate students and 7200 graduate students",
            "DU's Ritchie Center is home to the Magness Arena",
            "DU's hockey team plays in Magness Arena, named after cable television pioneer Bob Magness",
            "The Pioneers won the ice hockey NCAA National Championship in 2022",
            "DU's library is known as the Anderson Academic Commons"
        };
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
            ConversationLoaderConsole conversationLoader = new ConversationLoaderConsole(inputHandler, modelLoaderOutputs, temperature, useDatabase);

            if (numSyntheticDataToGenerate > 0)
            {
                conversationLoader.GenerateAndStoreSyntheticData(numSyntheticDataToGenerate).Wait();
                conversationLoader.PrintSyntheticDataTableHead(numSyntheticDataToGenerate);
            }
            else
            {
                await conversationLoader.StartChatAsync();
            }
        }
    }
}