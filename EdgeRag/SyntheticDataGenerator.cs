using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LLama;
using LLama.Common;
using Newtonsoft.Json;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private DatabaseManager databaseManager;
        private ChatSession session;
        private IInputHandler inputHandler;
        private float temperature;
        private DataTable syntheticDataTable;
        public event Action<string> OnMessage = delegate { };
        private string[] antiPrompts;
        private string modelType;

        public SyntheticDataGenerator(DatabaseManager databaseManager, ChatSession session, IInputHandler inputHandler, float temperature, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.session = session;
            this.inputHandler = inputHandler;
            this.temperature = temperature;
            this.antiPrompts = antiPrompts;
            modelType = databaseManager.ModelType;
            InitializeITSyntheticDataTable();
        }

        private void InitializeITSyntheticDataTable()
        {
            syntheticDataTable = new DataTable();
            syntheticDataTable.Columns.Add("llamaEmbedding", typeof(float[]));
            syntheticDataTable.Columns.Add("mistralEmbedding", typeof(float[]));
            syntheticDataTable.Columns.Add("mixtralEmbedding", typeof(float[]));
            syntheticDataTable.Columns.Add("phiEmbedding", typeof(float[]));
            syntheticDataTable.Columns.Add("incidentNumber", typeof(int));
            syntheticDataTable.Columns.Add("incidentTitle", typeof(string));
            syntheticDataTable.Columns.Add("incidentDetails", typeof(string));
            syntheticDataTable.Columns.Add("supportResponse", typeof(string));
            syntheticDataTable.Columns.Add("userSolution", typeof(string));
        }

        // This chains together LLM calls to build out a table of synthetic tech support data
        public async Task<DataTable> GenerateITDataPipeline(int n, string syntheticDataOutputPath)
        {
            for (int i = 0; i < n; i++)
            {
                OnMessage?.Invoke($"Generating item {i + 1}...\n");
                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device", "an Android device", "a Windows device", "a printer or copier", "a networking issue" };
                Random rand = new Random();
                string selectedTheme = themes[rand.Next(themes.Length)];

                DataRow newRow = syntheticDataTable.NewRow();
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
                var embeddings = databaseManager.GenerateEmbeddings("Concatenate all of the columns here");
                float[]? llamaEmbedding = null, mistralEmbedding = null, mixtralEmbedding = null, phiEmbedding = null;
                switch (modelType)
                {
                    case "llama":
                        llamaEmbedding = embeddings;
                        break;
                    case "mistral":
                        mistralEmbedding = embeddings;
                        break;
                    case "mixtral":
                        mixtralEmbedding = embeddings;
                        break;
                    case "phi":
                        phiEmbedding = embeddings;
                        break;
                }
            syntheticDataTable.Rows.Add(newRow);
        }
            syntheticDataOutputPath = $"{syntheticDataOutputPath}/syntheticData.json";
            syntheticDataOutputPath = @syntheticDataOutputPath;
            string json = databaseManager.DataTableToJson(syntheticDataTable);
            databaseManager.SaveJsonToFile(json, syntheticDataOutputPath);

            return syntheticDataTable;
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