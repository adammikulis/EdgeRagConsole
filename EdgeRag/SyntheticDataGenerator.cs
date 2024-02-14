using System.Data;
using LLama;
using Newtonsoft.Json;
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
        private int maxTokens;
        private string modelName;
        private string embeddingColumnName;


        public SyntheticDataGenerator(IOManager iOManager, ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager)
        {
            this.iOManager = iOManager;
            this.modelManager = modelManager;
            this.databaseManager = databaseManager;
            this.conversationManager = conversationManager;
            this.maxTokens = conversationManager.GetMaxTokens();
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            modelName = modelManager.modelName;
            embeddingColumnName = $"{modelName}Embeddings";
        }

        public async Task GenerateITDataPipeline(int n, string databaseJsonPath)
        {
            long currentIncidentNumber = DetermineStartingIncidentNumber(databaseJsonPath);

            for (int i = 0; i < n; i++)
            {
                currentIncidentNumber++;
                iOManager.SendMessage($"Generating item {currentIncidentNumber}...\n");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context
                string incidentDetails = await GenerateContentAsync(selectedTheme, "", "details");
                string incidentResponse = await GenerateContentAsync(selectedTheme, incidentDetails, "response");
                string incidentSolution = await GenerateContentAsync(selectedTheme, incidentDetails + " " + incidentResponse, "solution");

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["incidentResponse"] = incidentResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the incidentSolution
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentSolution);
                newRow[embeddingColumnName] = embeddings;

                vectorDatabase.Rows.Add(newRow);
            }

            // Serialize DataTable to JSON and save
            string json = JsonConvert.SerializeObject(vectorDatabase, Formatting.Indented);
            System.IO.File.WriteAllText(databaseJsonPath, json);
        }

        private long DetermineStartingIncidentNumber(string databaseJsonPath)
        {
            if (System.IO.File.Exists(databaseJsonPath))
            {
                string existingJson = System.IO.File.ReadAllText(databaseJsonPath);
                if (!string.IsNullOrWhiteSpace(existingJson))
                {
                    DataTable existingTable = JsonConvert.DeserializeObject<DataTable>(existingJson);
                    if (existingTable != null && existingTable.Rows.Count > 0)
                    {
                        return existingTable.AsEnumerable().Max(row => Convert.ToInt64(row["incidentNumber"]));
                    }
                }
            }
            return 0;
        }

        private string SelectRandomTheme()
        {
            string[] themes = { "an Apple device", "an Android device", "a Windows device", "a printer or copier", "networking" };
            Random rand = new Random();
            return themes[rand.Next(themes.Length)];
        }

        private async Task<string> GenerateContentAsync(string theme, string previousContent, string contentType)
        {
            string prompt = "";
            int tokenAllocationFactor = 16; // This allows us to easily increase/reduce the max amount of tokens generated for each stage


            switch (contentType)
            {
                case "details":
                    prompt = $"Describe a tech issue as the User in 2-3 sentences about {theme}.";
                    tokenAllocationFactor = 16;
                    break;
                case "response":
                    prompt = $"{previousContent} Summarize the issue and give 3-10 troubleshooting steps.";
                    tokenAllocationFactor = 8;
                    break;
                case "solution":
                    prompt = $"{previousContent} Summarize and describe how you solved the issue with the provided steps.";
                    tokenAllocationFactor = 4;
                    break;
            }

            int allocatedTokens = Math.Min(maxTokens, maxTokens / tokenAllocationFactor);
            return await conversationManager.InteractWithModelAsync(prompt, allocatedTokens);
        }
    }
}
