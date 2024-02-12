
using System.Data;
using LLama;
using LLama.Common;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private DatabaseManager databaseManager;
        private DataTable vectorDatabase;
        private ChatSession session;
        private int maxTokens;
        private string prompt;
        private string systemMessage;
        public event Action<string> OnMessage = delegate { };
        private string[] antiPrompts;
        private string modelType;
        string embeddingColumnName = string.Empty;

        public SyntheticDataGenerator(DatabaseManager databaseManager, ChatSession session, int maxTokens, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.session = session;
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
            long currentIncidentNumber = 0;

            for (int i = 0; i < n; i++)
            {
                currentIncidentNumber++;
                OnMessage?.Invoke($"Generating item {currentIncidentNumber}...\n");
                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device", "an Android device", "a Windows device", "a printer or copier", "networking" };
                Random rand = new Random();
                string selectedTheme = themes[rand.Next(themes.Length)];
                prompt = "";

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                //// Generates initial incident report title
                //systemMessage = $"Write a tech issue about {selectedTheme} as the user but do not solve.";
                //string incidentTitle = await InteractWithModelAsync(systemMessage, prompt, maxTokens / 4, userTemperature, antiPrompts);
                //newRow["incidentTitle"] = CleanUpString(incidentTitle);

                // Generate incident details based on the incident
                prompt = $"Describe a tech issue in 2-3 sentences as the user about {selectedTheme}";
                string incidentDetails = await InteractWithModelAsync(systemMessage, prompt, maxTokens / 2, userTemperature, antiPrompts);
                newRow["incidentDetails"] = CleanUpString(incidentDetails);

                // Generate IT support's response
                prompt = $"Give the user 3-10 troubleshooting steps for {newRow["incidentDetails"]}";
                string incidentResponse = await InteractWithModelAsync(systemMessage, prompt, maxTokens * 2, supportTemperature, antiPrompts);
                newRow["incidentResponse"] = CleanUpString(incidentResponse);

                // Generate user's final response based on IT support's help
                prompt = $"Tell tech support how you solved {newRow["incidentDetails"]} with {newRow["incidentResponse"]}";
                string userFinalResponse = await InteractWithModelAsync(systemMessage, prompt, maxTokens * 4, userTemperature, antiPrompts);
                newRow["incidentSolution"] = CleanUpString(userFinalResponse);

                string concatenatedText = $"{incidentDetails} {incidentResponse} {userFinalResponse}";
                concatenatedText = CleanUpString(concatenatedText);

                float[] embeddings = await databaseManager.GenerateEmbeddingsAsync(concatenatedText);
                newRow[embeddingColumnName] = embeddings;

                vectorDatabase.Rows.Add(newRow);
            }
            string json = databaseManager.DataTableToJson(vectorDatabase);
            bool overwrite = true;
            databaseManager.SaveJsonToFile(json, databaseJsonPath, overwrite);
        }

        // Needed for cleaner tables/JSON files
        public string CleanUpString(string input)
        {
            string cleanedString = input.Replace(antiPrompts[0], "")
                .Replace("\n", " ")
                .Replace("\r", " ")
                .Replace("     ", " ")
                .Replace("    ", " ")
                .Replace("   ", " ")
                .Replace("  ", " ")
                .Trim();
                

            return cleanedString;
        }

        private async Task<string> InteractWithModelAsync(string systemMessage, string prompt, int maxTokens, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n";
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

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response;
        }
    }
}