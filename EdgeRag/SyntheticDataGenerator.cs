// This class is used to generate fake support tickets that can then be queried by the model
// It demonstrates two concepts: prompt chaining and knowledge distillation
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

        // Factory method
        public static async Task<SyntheticDataGenerator> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            var syntheticDataGenerator = new SyntheticDataGenerator(modelManager, databaseManager, conversationManager, questionBatchSize);
            await syntheticDataGenerator.InitializeAsync();
            return syntheticDataGenerator;
        }

        // Initialization method
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

        // Add/change themes for a different variety of tickets
        private string SelectRandomTheme()
        {
            string[] themes = { "specific Apple product", "specific Android device", "specific Windows device", "specific printer or copier", "specific networking device", "specific Adobe product", "specific Microsoft product", "specific piece of software", "specific piece of tech hardware" };
            Random rand = new Random();
            return themes[rand.Next(themes.Length)];
        }

        // This is the main method of this class and it chains together prompts. Future iterations will make this process recursive
        public async Task GenerateITDataPipeline()
        {
            IOManager.ClearAndPrintHeading("Fake Support Ticket Generation");
            IOManager.SendMessageLine("\nChoose a number:\n1: View tickets as they generate\n2. Silent ticket generation");
            int choice = Convert.ToInt32(IOManager.ReadLine());
            bool internalDialog = choice == 1 ? false : true;

            IOManager.SendMessage("\nEnter the number of questions to generate: ");
            numQuestions = Convert.ToInt32(IOManager.ReadLine());
            currentIncidentNumber = await databaseManager.GetHighestIncidentNumberAsync();
            
            // Sets a minimum of 1 for questionBatchSize
            questionBatchSize = Math.Max(1, questionBatchSize);

            float supportTemperature = 0.65f; // Lower is more deterministic/repetitive
            float userTemperature = 0.85f; // Higher is more random

            for (int i = 0; i < numQuestions; i++)
            {
                IOManager.ClearAndPrintHeading("Synthetic Ticket Generation");
                currentIncidentNumber++;
                IOManager.SendMessageLine($"\nGenerating item {currentIncidentNumber}...");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context (this is what LangChain does)
                string incidentDetailsPrompt = "Pretend you are the user having a tech issue with your ";
                incidentDetails = await conversationManager.InteractWithModelAsync($"{incidentDetailsPrompt} {selectedTheme}", maxTokens / 16, userTemperature, internalDialog);
                incidentDetails = incidentDetails.Replace(incidentDetailsPrompt, "").Replace(selectedTheme, ""); // Always remove previous content after generating a response to avoid duplicated tokens
                incidentDetails = conversationManager.CleanUpString(incidentDetails);

                string supportResponsePrompt = "Try to solve this tech issue: ";
                supportResponse = await conversationManager.InteractWithModelAsync($"{supportResponsePrompt} {incidentDetails}", maxTokens / 8, supportTemperature, internalDialog);
                supportResponse = supportResponse.Replace(supportResponsePrompt, "").Replace(incidentDetails, "");
                supportResponse = conversationManager.CleanUpString(supportResponse);

                string incidentSolutionPrompt = "Choose the most likely solution from: ";
                incidentSolution = await conversationManager.InteractWithModelAsync($"{incidentSolutionPrompt} {supportResponse}", maxTokens / 4, userTemperature, internalDialog);
                incidentSolution = incidentSolution.Replace(incidentSolutionPrompt, "").Replace(supportResponse, "");
                incidentSolution = conversationManager.CleanUpString(incidentSolution);

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the combined incident details and solution
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
