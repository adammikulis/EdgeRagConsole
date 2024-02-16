﻿// Install Cuda 12.1 and run ReleaseCUDA12 to use Nvida GPU: https://developer.nvidia.com/cuda-12-1-0-download-archive
// Note: Only use Mistral models as most testing was done with them (lower q means smaller, fast, less accurate)

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
            if (!Directory.Exists(dataDirectoryPath)) 
            { 
                Directory.CreateDirectory(dataDirectoryPath);
            }

            int numTopMatches = 3; // This is when querying the database of facts
            string[] systemMessages = { $"" }; // Set this if you would like the LLM to always get a message first

            uint seed = 0;
            uint contextSize = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int maxTokens = 0; // Set to 0 to use the maximum allowed for whatever model type you choose
            int numGpuLayers = -1; // -1 is all-gpu inference, 0 is cpu-only, 1-33 are increasing levels of gpu usage (set to whatever your VRAM can handle)
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
