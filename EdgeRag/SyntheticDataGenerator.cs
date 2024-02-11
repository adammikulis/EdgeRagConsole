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
        private ChatSession session;
        private IInputHandler inputHandler;
        private float temperature;
        private DataTable syntheticDataTable;
        public event Action<string> OnMessage = delegate { };
        private string[] antiPrompts;

        public SyntheticDataGenerator(ChatSession session, IInputHandler inputHandler, float temperature, string[] antiPrompts)
        {
            this.antiPrompts = antiPrompts;
            this.session = session;
            this.inputHandler = inputHandler;
            this.temperature = temperature;
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
            syntheticDataTable.Columns.Add("userFinalResponse", typeof(string));
        }

        public string DataTableToJson(DataTable dataTable)
        {
            string json = JsonConvert.SerializeObject(dataTable, Formatting.Indented);
            return json;
        }

        public DataTable JsonToDataTable(string json)
        {
            DataTable dataTable = JsonConvert.DeserializeObject<DataTable>(json);
            return dataTable;
        }

        public void SaveJsonToFile(string json, string filePath)
        {
            // Create directory if it doesn't exist
            string directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write the JSON string to the file
            File.WriteAllText(filePath, json);
        }

        public string ReadJsonFromFile(string filePath)
        {
            // Read the JSON string from the file
            string json = File.ReadAllText(filePath);
            return json;
        }

        // This chains together LLM calls to build out a table of synthetic tech support data
        public async Task<DataTable> GenerateITDataPipeline(int n)
        {
            for (int i = 0; i < n; i++)
            {
                OnMessage?.Invoke($"Generating item {i + 1}...\n");
                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device", "an Android device", "a Windows device" };
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
                newRow["userFinalResponse"] = userFinalResponse.Replace(antiPrompts[0], "");

                syntheticDataTable.Rows.Add(newRow);
            }
            string syntheticDataOutputPath = @"C:\ai\data\synthetic\syntheticData.json";
            string json = DataTableToJson(syntheticDataTable);
            SaveJsonToFile(json, syntheticDataOutputPath);

            return syntheticDataTable;
        }

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
                .Replace("  ", " ");

            return cleanedString;
        }


        public void PrintSyntheticDataTable(int n)
        {
            // Check if the DataTable has any rows
            if (syntheticDataTable.Rows.Count == 0)
            {
                OnMessage?.Invoke("DataTable is empty.");
                return;
            }
            OnMessage?.Invoke("\n");
            // Print column headers
            foreach (DataColumn column in syntheticDataTable.Columns)
            {
                OnMessage?.Invoke($"{column.ColumnName}\t");
            }
            OnMessage?.Invoke("\n");

            // Iterate over the first n rows or the total number of rows, whichever is smaller
            int rowsToPrint = Math.Min(n, syntheticDataTable.Rows.Count);
            for (int i = 0; i < rowsToPrint; i++)
            {
                // Print each column's value for the current row
                foreach (DataColumn column in syntheticDataTable.Columns)
                {
                    OnMessage?.Invoke($"{syntheticDataTable.Rows[i][column]}\t");
                }
                OnMessage?.Invoke("\n"); // Move to the next line after printing all columns for a row
            }
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