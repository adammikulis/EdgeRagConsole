using System;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class PipelineManager
    {
        public IOManager iOManager;
        public ModelManager modelManager;
        public DatabaseManager databaseManager;
        public ConversationManager conversationManager;
        public SyntheticDataGenerator syntheticDataGenerator;

        public static async Task<PipelineManager> CreateAsync(string modelDirectoryPath, string databaseJsonPath, int numTopMatches, uint seed, uint contextSize, int maxTokens, int numGpuLayers, uint numCpuThreads, float temperature, string[] systemMessages, string[] antiPrompts, int numStars)
        {
            var pipelineManager = new PipelineManager();

            // Initialize IOManager
            pipelineManager.iOManager = await IOManager.CreateAsync(numStars);
            pipelineManager.iOManager.OnOutputMessage += Console.Write;

            // Initialize ModelManager
            pipelineManager.modelManager = await ModelManager.CreateAsync(pipelineManager.iOManager, modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);

            // Initialize DatabaseManager
            pipelineManager.databaseManager = await DatabaseManager.CreateAsync(pipelineManager.iOManager, pipelineManager.modelManager, databaseJsonPath, numTopMatches);

            // Initialize ConversationManager
            pipelineManager.conversationManager = await ConversationManager.CreateAsync(pipelineManager.iOManager, pipelineManager.modelManager, pipelineManager.databaseManager, maxTokens, temperature, systemMessages, antiPrompts);

            // Initialize SyntheticDataGenerator
            pipelineManager.syntheticDataGenerator = await SyntheticDataGenerator.CreateAsync(pipelineManager.iOManager, pipelineManager.modelManager, pipelineManager.databaseManager, pipelineManager.conversationManager);

            return pipelineManager;
        }
    }
}
