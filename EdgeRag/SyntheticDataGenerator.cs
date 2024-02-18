// This class is last to load and combines the model, conversation, and database managers to create synthetic data
// Currently hard-coded to generate tech support data (to match the tech support table) but will be generalized later

using System.Data;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private ModelManager modelManager;
        private DatabaseManager databaseManager;
        private ConversationManager conversationManager;
        private DataTable vectorDatabase;
        private string jsonDbPath;
        private string json;
        private int numQuestions;
        private int maxTokens;
        private int questionBatchSize;
        private string modelType;
        private long currentIncidentNumber;
        private string incidentDetails;
        private string supportResponse;
        private string incidentSolution;

        public SyntheticDataGenerator(ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            this.modelManager = modelManager;
            this.databaseManager = databaseManager;
            this.conversationManager = conversationManager;
            this.maxTokens = conversationManager.maxTokens;
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            this.questionBatchSize = questionBatchSize;
            modelType = modelManager.selectedModelType;
            jsonDbPath = databaseManager.dataDirectoryPath;
            json = "";
        }

        public static async Task<SyntheticDataGenerator> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            var syntheticDataGenerator = new SyntheticDataGenerator(modelManager, databaseManager, conversationManager, questionBatchSize);
            await syntheticDataGenerator.InitializeAsync();
            return syntheticDataGenerator;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(async () =>
            {
                string filePath = Path.Combine(jsonDbPath, databaseManager.dataFileName);
                if (File.Exists(filePath))
                {
                    string existingJson = await databaseManager.ReadJsonFromFileAsync(filePath);
                    DataTable tempTable = string.IsNullOrWhiteSpace(existingJson) ? new DataTable() : databaseManager.JsonToDataTable(existingJson);

                    // Add missing columns to the vector database
                    foreach (DataColumn column in tempTable.Columns)
                    {
                        if (!vectorDatabase.Columns.Contains(column.ColumnName))
                        {
                            vectorDatabase.Columns.Add(column.ColumnName, column.DataType);
                        }
                    }
                }
            });
        }

        private string SelectRandomTheme()
        {
            string[] themes = { "a specific Apple device", "a specific Android device", "a specific Windows device", "a specific printer or copier", "a specific networking device", "a specific piece of software", "a specific piece of tech hardware" };
            Random rand = new Random();
            return themes[rand.Next(themes.Length)];
        }

        public async Task GenerateITDataPipeline()
        {
            IOManager.ClearAndPrintHeading("Synthetic Ticket Generation");
            IOManager.SendMessageLine("\nChoose a number:\n1: View tickets as they generate\n2. Silent ticket generation");
            int choice = Convert.ToInt32(IOManager.ReadLine());
            bool internalDialog = choice == 1 ? false : true;

            IOManager.SendMessage("\nEnter the number of questions to generate: ");
            numQuestions = Convert.ToInt32(IOManager.ReadLine());
            currentIncidentNumber = await databaseManager.GetHighestIncidentNumberAsync();
            
            // Sets a minimum of 1 for questionBatchSize
            questionBatchSize = Math.Max(1, questionBatchSize);

            float supportTemperature = 0.5f; // Lower is more deterministic
            float userTemperature = 0.8f; // Higher is more random

            for (int i = 0; i < numQuestions; i++)
            {
                currentIncidentNumber++;
                IOManager.SendMessageLine($"\nGenerating item {currentIncidentNumber}...");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context (this is what LangChain does)
                string incidentDetailsPrompt = "Describe a specific tech issue about: ";
                incidentDetails = await conversationManager.InteractWithModelAsync($"{incidentDetailsPrompt} {selectedTheme}", maxTokens / 16, userTemperature, internalDialog);
                incidentDetails = incidentDetails.Replace(incidentDetailsPrompt, ""); // Always remove previous content after generating a response to avoid duplicated tokens
                incidentDetails = incidentDetails.Replace(selectedTheme, "");
                incidentDetails = conversationManager.CleanUpString(incidentDetails);

                string supportResponsePrompt = "Try to solve this tech issue: ";
                supportResponse = await conversationManager.InteractWithModelAsync($"{supportResponsePrompt} {incidentDetails}", maxTokens / 8, supportTemperature, internalDialog);
                supportResponse = supportResponse.Replace(supportResponsePrompt, "");
                supportResponse = supportResponse.Replace(incidentDetails, "");
                supportResponse = conversationManager.CleanUpString(supportResponse);

                string incidentSolutionPrompt = "Choose the most likely solution from: ";
                incidentSolution = await conversationManager.InteractWithModelAsync($"{incidentSolutionPrompt} {supportResponse}", maxTokens / 4, userTemperature, internalDialog);
                incidentSolution = incidentSolution.Replace(incidentSolutionPrompt, "");
                incidentSolution = incidentSolution.Replace(supportResponse, "");
                incidentSolution = conversationManager.CleanUpString(incidentSolution);

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the incidentDetails
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentDetails + " " + incidentSolution);
                newRow[modelType] = embeddings;

                vectorDatabase.Rows.Add(newRow);

                // Save after every questionBatchSize
                if ((i + 1) % questionBatchSize == 0 || i == numQuestions - 1)
                {
                    json = databaseManager.DataTableToJson(vectorDatabase);
                    await databaseManager.SaveJsonToFileAsync(json, databaseManager.dataFileName);
                }
            }
        }
    }
}
