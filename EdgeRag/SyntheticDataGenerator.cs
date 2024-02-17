

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
                // Assuming databaseManager has an asynchronous method to read JSON from a file
                string filePath = Path.Combine(jsonDbPath, databaseManager.dataFileName);
                if (File.Exists(filePath))
                {
                    // Use the asynchronous method to read JSON from file
                    string existingJson = await databaseManager.ReadJsonFromFileAsync(filePath);
                    DataTable tempTable = string.IsNullOrWhiteSpace(existingJson) ? new DataTable() : databaseManager.JsonToDataTable(existingJson);

                    // Add missing columns to the vectorDatabase DataTable
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
            IOManager.SendMessage("\nEnter the number of questions to generate: ");
            numQuestions = Convert.ToInt32(IOManager.ReadLine());
            currentIncidentNumber = await databaseManager.GetHighestIncidentNumberAsync();
            // Sets a minimum of 1 for questionBatchSize
            questionBatchSize = Math.Max(1, questionBatchSize);

            float supportTemperature = 0.5f;
            float userTemperature = 0.8f;

            for (int i = 0; i < numQuestions; i++)
            {
                currentIncidentNumber++;
                IOManager.SendMessage($"Generating item {currentIncidentNumber}...\n");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();

                // Ensure the 'incidentNumber' column exists and add it if necessary
                if (!vectorDatabase.Columns.Contains("incidentNumber"))
                {
                    vectorDatabase.Columns.Add("incidentNumber", typeof(long));
                }

                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context (this is what LangChain does)
                string incidentDetailsPrompt = "Describe a specific tech issue about: ";
                incidentDetails = await conversationManager.InteractWithModelAsync($"{incidentDetailsPrompt}{selectedTheme}", maxTokens / 16, userTemperature, false);
                incidentDetails = incidentDetails.Replace(incidentDetailsPrompt, "");
                incidentDetails = incidentDetails.Replace(selectedTheme, "");

                string supportResponsePrompt = "Try to solve this tech issue: ";
                supportResponse = await conversationManager.InteractWithModelAsync($"{supportResponsePrompt}{incidentDetails}", maxTokens / 8, supportTemperature, false);
                supportResponse = supportResponse.Replace(supportResponsePrompt, "");

                // Always remove previous content after generating a response to avoid duplicated tokens
                supportResponse = supportResponse.Replace(incidentDetails, "");

                string incidentSolutionPrompt = "Choose the most likely solution from: ";
                incidentSolution = await conversationManager.InteractWithModelAsync($"{incidentSolutionPrompt}{supportResponse}", maxTokens / 4, userTemperature, false);
                incidentSolution = incidentSolution.Replace(incidentSolutionPrompt, "");
                incidentSolution = incidentSolution.Replace(supportResponse, "");

                string summarizedIncidentSolutionPrompt = "Summarize: ";
                string summarizedIncidentSolution = await conversationManager.InteractWithModelAsync($"{summarizedIncidentSolutionPrompt}{incidentSolution}", maxTokens / 16, userTemperature, false);
                summarizedIncidentSolution = summarizedIncidentSolution.Replace(summarizedIncidentSolutionPrompt, "");
                summarizedIncidentSolution = summarizedIncidentSolution.Replace(incidentSolution, "");

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["incidentSolution"] = summarizedIncidentSolution;

                // Generate embeddings for the incidentDetails
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentDetails);
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
