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
            string databaseJsonPath = "C:/ai/data/synthetic/syntheticData.json"; // Change this path if you would like to save the data elsewhere
            int numTopMatches = 3; // This is when querying the database of facts

            string[] systemMessages = { $"You are a chatbot who needs to solve the user's query by giving many detailed steps" };

            uint seed = 1;
            uint contextSize = 4096;
            int maxTokens = 4096;
            int numGpuLayers = 33; // Adjust based on VRAM capability
            uint numCpuThreads = 8;
            float temperature = 0.5f; // Lower is more deterministic, higher is more random
            string[] antiPrompts = { "<end>" }; // This is what the LLM emits to stop the message
            
            int numStars = 50; // This is for rendering


            var pipelineManager = await PipelineManager.CreateAsync(modelDirectoryPath, databaseJsonPath, numTopMatches, seed, contextSize, maxTokens, numGpuLayers, numCpuThreads, temperature, systemMessages, antiPrompts, numStars);

            // Menu loop
            await pipelineManager.iOManager.RunMenuAsync(
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
