using System.Data;
using LLama;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private IOManager iOManager;
        private ModelManager modelManager;
        private DatabaseManager databaseManager;
        private ConversationManager conversationManager;
        private DataTable vectorDatabase;
        private string jsonDbPath;
        private string json;
        private int maxTokens;
        private int questionBatchSize;
        private string modelName;
        private string embeddingColumnName;
        private long currentIncidentNumber;

        public SyntheticDataGenerator(IOManager iOManager, ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager)
        {
            this.iOManager = iOManager;
            this.modelManager = modelManager;
            this.databaseManager = databaseManager;
            this.conversationManager = conversationManager;
            this.maxTokens = conversationManager.GetMaxTokens();
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            this.questionBatchSize = questionBatchSize;
            modelName = modelManager.modelName;
            jsonDbPath = databaseManager.jsonDbPath;
            json = "";
            embeddingColumnName = $"{modelName}Embeddings";
        }

        public static async Task<SyntheticDataGenerator> CreateAsync(IOManager iOManager, ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            var syntheticDataGenerator = new SyntheticDataGenerator(iOManager, modelManager, databaseManager, conversationManager);
            await syntheticDataGenerator.InitializeAsync();
            return syntheticDataGenerator;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                currentIncidentNumber = databaseManager.highestIncidentNumber;
            });
        }

        private string SelectRandomTheme()
        {
            string[] themes = { "a specific Apple device", "a specific Android device", "a specific Windows device", "a specific printer or copier", "a specific networking device", "a specific piece of software", "a specific piece of tech hardware" };
            Random rand = new Random();
            return themes[rand.Next(themes.Length)];
        }

        public async Task GenerateITDataPipeline(int numQuestions)
        {
            // Sets a minimum of 1 for questionBatchSize
            questionBatchSize = Math.Max(1, questionBatchSize);

            for (int i = 0; i < numQuestions; i++)
            {
                currentIncidentNumber++;
                iOManager.SendMessage($"Generating item {currentIncidentNumber}...\n");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context (this is what LangChain does)
                string incidentDetails = await GenerateContentAsync($"Describe a tech issue as the User in 2-3 sentences about {selectedTheme}", 16);
                string supportResponse = await GenerateContentAsync($"Work with the user to troubleshoot their issue and ask for any additional information needed for " + incidentDetails, 8);
                string userResponse = await GenerateContentAsync($"Troubleshoot " + incidentDetails + " with " + supportResponse, 8);
                string incidentSolution = await GenerateContentAsync($"Solve and summarize" + incidentDetails + " based on " + userResponse, 4);

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["userResponse"] = userResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the incidentDetails
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentDetails);
                newRow[embeddingColumnName] = embeddings;

                vectorDatabase.Rows.Add(newRow);

                // Save after every questionBatchSize items are added or if on the last item
                if ((i + 1) % questionBatchSize == 0 || i == numQuestions - 1)
                {
                    json = databaseManager.DataTableToJson(vectorDatabase);
                    databaseManager.SaveJsonToFile(json, jsonDbPath);
                }
            }
        }

        private async Task<string> GenerateContentAsync(string generateContentPrompt, int tokenAllocationFactor)
        {
            int allocatedTokens = Math.Min(maxTokens, maxTokens / tokenAllocationFactor);
            return await conversationManager.InteractWithModelAsync(generateContentPrompt, allocatedTokens, false);
        }
    }
}
