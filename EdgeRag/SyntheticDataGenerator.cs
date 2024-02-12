
using System.Data;
using LLama;
using LLama.Common;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private DatabaseManager databaseManager;
        private DataTable vectorDatabase;
        private ChatSession session;
        public event Action<string> OnMessage = delegate { };
        private string[] antiPrompts;
        private string modelType;
        string embeddingColumnName = string.Empty;

        public SyntheticDataGenerator(DatabaseManager databaseManager, ChatSession session, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.session = session;
            this.antiPrompts = antiPrompts;
            modelType = databaseManager.ModelType;
            vectorDatabase = databaseManager.GetVectorDatabase();
            embeddingColumnName = $"{modelType}Embeddings";
        }


        // This chains together LLM calls to build out a table of synthetic tech support data
        public async Task GenerateITDataPipeline(int n, string databaseJsonPath)
        {
            long currentIncidentNumber = 1;

            for (int i = 0; i < n; i++)
            {
                OnMessage?.Invoke($"Generating item {currentIncidentNumber}...\n");
                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device", "an Android device", "a Windows device", "a printer or copier", "a networking issue" };
                Random rand = new Random();
                string selectedTheme = themes[rand.Next(themes.Length)];

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = i + 1; // Increment for each incident

                // Generates initial incident report title
                string prompt = "";
                string promptInstructions = $"Write a short tech issue title as a user for {selectedTheme}";
                string incidentTitle = await InteractWithModelAsync(promptInstructions, prompt, userTemperature, antiPrompts);
                incidentTitle = CleanUpString(incidentTitle);
                newRow["incidentTitle"] = incidentTitle.Replace(antiPrompts[0], "");

                // Generate incident details based on the incident
                promptInstructions = $"Write details for tech support issue as the user: {incidentTitle}";
                string incidentDetails = await InteractWithModelAsync(promptInstructions, incidentTitle, userTemperature, antiPrompts);
                incidentDetails = CleanUpString(incidentDetails);
                incidentDetails.Replace(incidentTitle, "");
                newRow["incidentDetails"] = incidentDetails.Replace(antiPrompts[0], "");

                // Generate IT support's response
                promptInstructions = $"Write troubleshooting steps as tech support for: {incidentDetails} ";
                string supportResponse = await InteractWithModelAsync(promptInstructions, incidentDetails, supportTemperature, antiPrompts);
                supportResponse = CleanUpString(supportResponse);
                newRow["supportResponse"] = supportResponse.Replace(antiPrompts[0], "");

                // Generate user's final response based on IT support's help
                promptInstructions = $"Write how you solved the problem as the user based on steps: {supportResponse}";
                string userFinalResponse = await InteractWithModelAsync(promptInstructions, supportResponse, userTemperature, antiPrompts);
                userFinalResponse = CleanUpString(userFinalResponse);
                newRow["userSolution"] = userFinalResponse.Replace(antiPrompts[0], "");

                string concatenatedText = $"{incidentTitle} {incidentDetails} {supportResponse} {userFinalResponse}";

                float[] embeddings = await databaseManager.GenerateEmbeddingsAsync(concatenatedText);

                newRow[embeddingColumnName] = embeddings;

                newRow["incidentNumber"] = currentIncidentNumber;
                vectorDatabase.Rows.Add(newRow);
            }
            string json = databaseManager.DataTableToJson(vectorDatabase);
            databaseManager.SaveJsonToFile(json, databaseJsonPath);
        }

        // Needed for cleaner tables/JSON files
        public string CleanUpString(string input)
        {
            string cleanedString = input.Replace("Title", "")
                .Replace("Description", "")
                .Replace("Tech", "")
                .Replace("\n", " ")
                .Replace("\r", " ")
                .Replace("\t", " ")
                .Replace("    ", " ")
                .Replace("   ", " ")
                .Replace("  ", " ")
                .Trim();

            return cleanedString;
        }

        private async Task<string> InteractWithModelAsync(string promptInstructions, string prompt, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n"; // Ensure the session is initialized
            prompt = $"{promptInstructions} {prompt}";
            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response.Trim();
        }
    }
}