
using System.Data;
using LLama;
using LLama.Common;
using Newtonsoft.Json;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private DatabaseManager databaseManager;
        private ConversationManager conversationManager;
        private DataTable vectorDatabase;
        private ChatSession chatSession;
        DataRow newRow;
        private int maxTokens;
        private string prompt;
        private string systemMessage;
        public event Action<string> OnMessage = delegate { };
        private string[] antiPrompts;
        private string modelType;
        string embeddingColumnName = string.Empty;

        public SyntheticDataGenerator(DatabaseManager databaseManager, ConversationManager conversationManager, int maxTokens, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.chatSession = chatSession;
            this.maxTokens = maxTokens;
            this.antiPrompts = antiPrompts;
            systemMessage = "";

            modelType = databaseManager.ModelType;
            vectorDatabase = databaseManager.GetVectorDatabase();
            embeddingColumnName = $"{modelType}Embeddings";
        }

        // This chains together LLM calls to build out a table of synthetic tech support data
        public async Task GenerateITDataPipeline(int n, string databaseJsonPath)
        {
            // Determine the starting incident number
            long currentIncidentNumber = 0;
            if (File.Exists(databaseJsonPath))
            {
                string existingJson = File.ReadAllText(databaseJsonPath);
                if (!string.IsNullOrWhiteSpace(existingJson))
                {
                    DataTable existingTable = JsonConvert.DeserializeObject<DataTable>(existingJson);
                    if (existingTable != null && existingTable.Rows.Count > 0)
                    {
                        currentIncidentNumber = existingTable.AsEnumerable()
                                      .Max(row => Convert.ToInt64(row["incidentNumber"]));
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                currentIncidentNumber++; // Increment the incident number for each new entry
                OnMessage?.Invoke($"Generating item {currentIncidentNumber}...\n");

                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device", "an Android device", "a Windows device", "a printer or copier", "networking" };
                Random rand = new Random();
                string selectedTheme = themes[rand.Next(themes.Length)];

                newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber; // Set the current incident number

                // Generate incident details
                string prompt = $"Describe a tech issue as the User in 2-3 sentences about {selectedTheme}";
                string incidentDetails = await InteractWithModelAsync(systemMessage, prompt, maxTokens, userTemperature, antiPrompts);
                newRow["incidentDetails"] = conversationManager.CleanUpString(incidentDetails);

                // Generate IT support's response
                prompt = $"Summarize {newRow["incidentDetails"]} and give 3-10 troubleshooting steps";
                string incidentResponse = await InteractWithModelAsync(systemMessage, prompt, maxTokens * 2, supportTemperature, antiPrompts);
                newRow["incidentResponse"] = conversationManager.CleanUpString(incidentResponse);

                // Generate user's final response
                prompt = $"As the user summarize and describe how you solved {newRow["incidentDetails"]} with steps {newRow["incidentResponse"]}";
                string userFinalResponse = await InteractWithModelAsync(systemMessage, prompt, maxTokens * 4, userTemperature, antiPrompts);
                newRow["incidentSolution"] = conversationManager.CleanUpString(userFinalResponse);

                // Concatenate texts and generate embeddings
                string concatenatedText = $"{incidentDetails} {incidentResponse} {userFinalResponse}";
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(conversationManager.CleanUpString(concatenatedText));
                newRow[embeddingColumnName] = embeddings;

                // Add to DataTable and write to JSON
                vectorDatabase.Rows.Add(newRow);
            }

            string json = databaseManager.DataTableToJson(vectorDatabase);
            databaseManager.SaveJsonToFile(json, databaseJsonPath);
        }


        private async Task<string> InteractWithModelAsync(string systemMessage, string prompt, int maxTokens, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (chatSession == null) return "chatSession still initializing, please wait.\n";
            if (prompt == "" || prompt == null)
            {
                prompt = $"{systemMessage}";
            }
            else if (this.systemMessage == "" || this.systemMessage == null)
            {
                prompt = $"{prompt}";
            }
            else
            {
                prompt = $"{systemMessage} {prompt}";
            }

            await foreach (var text in chatSession.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response;
        }
    }
}