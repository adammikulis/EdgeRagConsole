namespace EdgeRag
{
    public class PipelineManager
    {
        public ModelManager modelManager;
        public DatabaseManager databaseManager;
        public ConversationManager conversationManager;
        public SyntheticDataGenerator syntheticDataGenerator;

        public static async Task<PipelineManager> CreateAsync(string modelDirectoryPath, string dataDirectoryPath, int numTopMatches, uint seed, uint contextSize, int maxTokens, uint numCpuThreads, float temperature, string[] systemMessages, string[] antiPrompts, int questionBatchSize)
        {
            var pipelineManager = new PipelineManager();

            // Initialize ModelManager
            pipelineManager.modelManager = await ModelManager.CreateAsync(modelDirectoryPath, seed, contextSize, numCpuThreads);

            // Initialize DatabaseManager
            pipelineManager.databaseManager = await DatabaseManager.CreateAsync(pipelineManager.modelManager, dataDirectoryPath);

            // Initialize ConversationManager
            pipelineManager.conversationManager = await ConversationManager.CreateAsync(pipelineManager.modelManager, pipelineManager.databaseManager, maxTokens, systemMessages, antiPrompts, numTopMatches);

            // Initialize SyntheticDataGenerator
            pipelineManager.syntheticDataGenerator = await SyntheticDataGenerator.CreateAsync(pipelineManager.modelManager, pipelineManager.databaseManager, pipelineManager.conversationManager, questionBatchSize);

            return pipelineManager;
        }
    }
}
