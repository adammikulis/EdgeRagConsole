// This program allows the user to generate synthetic tech support data and store in a vector database with a locally-run large language model (LLM)
// The user can then chat with the LLM, querying the database with an issue to get exact solutions from past incidents
// The tickets can be generated with a higher-quality model, and then a faster model can be used to query that data leading to performance increase
// There are two distributions (x64 only): one with a CPU backend that runs on all x64 devices and one with a CUDA12 backend that requires an Nvidia GPU
// Install Cuda 12.x and run ReleaseCUDA12 to use Nvida GPU: https://developer.nvidia.com/cuda-12-1-0-download-archive
// Note: Only use Mistral models as all testing was done with them. Recommend Q4 and above for best quality, but Q2 is acceptable

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            IOManager.OnMessage += Console.Write;
            IOManager.PrintIntroMessage();

            string modelDirectory = @"models";
            string projectDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string modelDirectoryPath = Path.Combine(projectDirectory, modelDirectory);

            string dataDirectory = @"datasets";
            string dataDirectoryPath = Path.Combine(projectDirectory, dataDirectory);

            if (!Directory.Exists(modelDirectoryPath))
            {
                Directory.CreateDirectory(modelDirectoryPath);
            }
            if (!Directory.Exists(dataDirectoryPath))
            {
                Directory.CreateDirectory(dataDirectoryPath);
            }

            string[] systemMessages = { $"" }; // Set this if you would like the LLM to always get a system message first

            uint seed = 0;
            uint contextSize = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int maxTokens = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            uint numCpuThreads = 8; // Use a number that matches your physical cores for best performance
            float temperature = 0.5f; // Lower is more deterministic, higher is more random
            string[] antiPrompts = { "<end>" }; // This is what the LLM emits to stop the message, do not change
            int questionBatchSize = 8; // This allows generated questions to be saved in batches to JSON instead of at the very end

            // The PipelineManager handles all setup/initalization. It loads the ModelManager, DatabaseManager, ConversationManager, and SyntheticDataGenerator (in that order)
            var pipelineManager = await PipelineManager.CreateAsync(modelDirectoryPath, dataDirectoryPath, seed, contextSize, maxTokens, numCpuThreads, temperature, systemMessages, antiPrompts, questionBatchSize);

            // Main menu loop
            while (true)
            {
                IOManager.ClearAndPrintHeading("Main Menu");
                IOManager.SendMessage("\nMenu (choose a number):");
                IOManager.SendMessage("\n1. Chat");
                IOManager.SendMessage("\n2. Chat using Database");
                IOManager.SendMessage("\n3. Generate Questions and Chat using Database");
                IOManager.SendMessage("\n4. Generate Questions and Quit");
                IOManager.SendMessage("\n5. Download Model");
                IOManager.SendMessage("\n6. Load Different Model");
                IOManager.SendMessage("\n7. Load Different Database");
                IOManager.SendMessage("\n8. Quit\n");
                var option = IOManager.ReadLine();

                switch (option)
                {
                    // Chat
                    case "1":
                        await pipelineManager.conversationManager.StartChatAsync(false);
                        break;
                
                    // Chat using database
                    case "2":
                        await pipelineManager.conversationManager.StartChatAsync(true);
                        break;
                
                    // Generate questions and chat using database
                    case "3":
                   
                        await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline();
                        await pipelineManager.conversationManager.StartChatAsync(true);
                        break;
                
                    // Generate questions and quit
                    case "4":
                        await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline();
                        pipelineManager.modelManager.Dispose();
                        Environment.Exit(0);
                        break;
                
                    // Download model                
                    case "5":
                        await DownloadManager.DownloadModelAsync("mistral", modelDirectoryPath); // Hard-coded to Mistral for now, will allow llama and phi in the future;
                        break;
                
                    // Load different model
                    case "6":
                        pipelineManager.modelManager.Dispose();
                        pipelineManager = await PipelineManager.CreateAsync(modelDirectoryPath, dataDirectoryPath, seed, contextSize, maxTokens, numCpuThreads, temperature, systemMessages, antiPrompts, questionBatchSize);
                        break;
                
                    // Load different database
                    case "7":
                        pipelineManager.databaseManager = await DatabaseManager.CreateAsync(pipelineManager.modelManager, dataDirectoryPath);
                        pipelineManager.conversationManager = await ConversationManager.CreateAsync(pipelineManager.modelManager, pipelineManager.databaseManager, maxTokens, systemMessages, antiPrompts);
                        pipelineManager.syntheticDataGenerator = await SyntheticDataGenerator.CreateAsync(pipelineManager.modelManager, pipelineManager.databaseManager, pipelineManager.conversationManager, questionBatchSize);
                        break;
                    
                    case "8":
                        pipelineManager.modelManager.Dispose();
                        Environment.Exit(0);
                        break;

                    default:
                        IOManager.SendMessage("\nInvalid option, please try again.\n");
                        break;
                }
            }
        }
    }
}