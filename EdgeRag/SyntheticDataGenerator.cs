using System.Data;
using LLama;
using System;
using System.Linq;
using System.Threading.Tasks;

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
            this.maxTokens = conversationManager.GetMaxTokens();
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            this.questionBatchSize = questionBatchSize;
            modelType = modelManager.selectedModelname;
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
                

                // Update vectorDatabase with missing columns based on loaded JSON data
                string filePath = Path.Combine(jsonDbPath, databaseManager.dataFileName);
                if (File.Exists(filePath))
                {
                    string existingJson = databaseManager.ReadJsonFromFile(filePath);
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

        public async Task GenerateITDataPipeline(int numQuestions)
        {
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
                incidentDetails = await conversationManager.InteractWithModelAsync($"As the user, describe a tech issue you are having with {selectedTheme}", maxTokens / 8, userTemperature, false);
                supportResponse = await conversationManager.InteractWithModelAsync($"As support, work with the user and ask for any additional information about " + incidentDetails, maxTokens / 4, supportTemperature, false);
                incidentSolution = await conversationManager.InteractWithModelAsync($"Solve " + incidentDetails + " based on " + supportResponse, maxTokens, userTemperature, false);

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the incidentDetails
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentDetails);
                newRow[modelType] = embeddings;

                vectorDatabase.Rows.Add(newRow);

                // Save after every questionBatchSize
                if ((i + 1) % questionBatchSize == 0 || i == numQuestions - 1)
                {
                    json = databaseManager.DataTableToJson(vectorDatabase);
                    databaseManager.SaveJsonToFile(json);
                }
            }
        }
    }
}
