// Note: All data has been generated/tested with mistral-7b-instruct-v0.2.Q8_0.gguf for consistency
// Use mistral-7b-instruct-v0.2.Q8_0.gguf for best results
// Use mistral-7b-instruct-v0.2.Q2_K.gguf for fastest/lowest memory network (at the cost of accuracy)

using LLama;
using System;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;
using static System.Net.WebRequestMethods;

namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            IOManager.OnOutputMessage += Console.Write;

            string modelDirectory = @"models";
            string projectDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string modelDirectoryPath = Path.Combine(projectDirectory, modelDirectory);

            string dataDirectory = @"datasets";
            string dataDirectoryPath = Path.Combine(projectDirectory, dataDirectory);
            string dataFileName = "syntheticData.json";

            if (!Directory.Exists(modelDirectoryPath))
            {
                Directory.CreateDirectory(modelDirectoryPath);
            }


            int numTopMatches = 3; // This is when querying the database of facts

            string[] url = {"https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q2_K.gguf",
                "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf" };
            // await DownloadManager.DownloadFileAsync(url[1], modelDirectoryPath);

            string[] systemMessages = { $"" };

            uint seed = 1;
            uint contextSize = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int maxTokens = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int numGpuLayers = -1; // -1 is all-gpu inference, 0 is cpu-only, set to whatever your VRAM can handle
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Lower is more deterministic, higher is more random
            string[] antiPrompts = { "<end>" }; // This is what the LLM emits to stop the message

            int questionBatchSize = 32;

            var pipelineManager = await PipelineManager.CreateAsync(modelDirectoryPath, dataDirectoryPath, dataFileName, numTopMatches, seed, contextSize, maxTokens, numGpuLayers, numCpuThreads, temperature, systemMessages, antiPrompts, questionBatchSize);

            // Menu loop
            await IOManager.RunMenuAsync(
            chat: () => pipelineManager.conversationManager.StartChatAsync(false),
            chatUsingDatabase: () => pipelineManager.conversationManager.StartChatAsync(true),
            generateQuestionsAndChat: async (numQuestions) => {
                await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                await pipelineManager.conversationManager.StartChatAsync(true);
            },
            generateQuestions: async (numQuestions) => {
                await pipelineManager.syntheticDataGenerator.GenerateITDataPipeline(numQuestions);
                Environment.Exit(0);
            },
            quit: () => {
                Environment.Exit(0);
            });
        }
    }
}
