// Install Cuda 12.1 and run ReleaseCUDA12 to use Nvida GPU: https://developer.nvidia.com/cuda-12-1-0-download-archive
// Note: Only use Mistral models as most testing was done with them (lower q means smaller, fast, less accurate)


namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            IOManager.OnOutputMessage += Console.Write;
            IOManager.SendMessage("Welcome to EdgeRag! This is a Retrieval-Augmented Generative (RAG) A.I. pipeline " +
                      "that lets a local chatbot refer to existing solutions/documentation when crafting its response. " +
                      "Everything is run on your device, creating a secure chat environment for sensitive data. " +
                      "\nGenerate synthetic data to easily populate a database and then search. You can even use " +
                      "a higher quality model to generate the tickets and then a faster, smaller model to serve as the chatbot. " +
                      "\nRefer any questions to Adam Mikulis, and have fun!\n\n");

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

            int numTopMatches = 3; // This is when querying the database of facts
            string[] systemMessages = { $"" }; // Set this if you would like the LLM to always get a message first

            uint seed = 0;
            uint contextSize = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int maxTokens = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Lower is more deterministic, higher is more random
            string[] antiPrompts = { "<end>" }; // This is what the LLM emits to stop the message

            int questionBatchSize = 32;

            var pipelineManager = await PipelineManager.CreateAsync(modelDirectoryPath, dataDirectoryPath, numTopMatches, seed, contextSize, maxTokens, numCpuThreads, temperature, systemMessages, antiPrompts, questionBatchSize);

            // Main menu loop
            await IOManager.RunMenuAsync(
            chat: () => pipelineManager.conversationManager.StartChatAsync(false),
            chatUsingDatabase: () => pipelineManager.conversationManager.StartChatAsync(true),
            generateQuestionsAndChat: async (numQuestions) => {
                await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                await pipelineManager.conversationManager.StartChatAsync(true);
            },
            generateQuestions: async (numQuestions) => {
                await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                pipelineManager.modelManager.Dispose();
                Environment.Exit(0);
            },
            downloadModel: async () => {
                await DownloadManager.DownloadModelAsync("mistral", modelDirectoryPath);
            },
            loadDifferentModel: async() => {
                if (pipelineManager.modelManager != null)
                {
                    pipelineManager.modelManager.Dispose();
                    pipelineManager.modelManager = await ModelManager.CreateAsync(modelDirectoryPath, seed, contextSize, numCpuThreads);
                    pipelineManager.conversationManager = await ConversationManager.CreateAsync(pipelineManager.modelManager, pipelineManager.databaseManager, maxTokens, systemMessages, antiPrompts, numTopMatches);
                }
            },
            quit: () => {
                pipelineManager.modelManager.Dispose();
                Environment.Exit(0);
            });
        }
    }
}
