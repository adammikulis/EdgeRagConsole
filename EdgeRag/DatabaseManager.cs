using System.Data;
using Newtonsoft.Json;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private DataTable vectorDatabase;
        private ModelManager modelManager;
        public string dataDirectoryPath;
        public string dataFileName;
        public string[] databaseTypes;
        private List<(string Name, Type DataType)> techSupportColumns;

        public DatabaseManager(ModelManager modelManager, string dataDirectoryPath)
        {
            this.dataDirectoryPath = dataDirectoryPath;
            this.modelManager = modelManager;
            vectorDatabase = new DataTable();
            databaseTypes = new string[] { "Tech Support" };
            string dataFileName = "";

        }
        public static async Task<DatabaseManager> CreateAsync(ModelManager modelManager, string dataDirectoryPath)
        {
            var databaseManager = new DatabaseManager(modelManager, dataDirectoryPath);
            await databaseManager.InitializeAsync();
            return databaseManager;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(async () =>
            {
                techSupportColumns = new List<(string Name, Type DataType)>
                {
                    ("incidentNumber", typeof(long)),
                    ("incidentDetails", typeof(string)),
                    ("supportResponse", typeof(string)),
                    ("incidentSolution", typeof(string)),
                    (modelManager.selectedModelType, typeof(double[]))
                };

                IOManager.ClearConsole();
                IOManager.PrintHeading("Vector Databases");
                
                var jsonFiles = Directory.GetFiles(dataDirectoryPath, "*.json");
                if (jsonFiles.Length > 0)
                {
                    await LoadDatabase(jsonFiles);
                }
                else
                {
                    IOManager.SendMessageLine("\nNo database found.");
                    await CreateNewDatabase();
                }
            });
        }

        private void AddDatabaseColumns(List<(string Name, Type DataType)> columns)
        {
            foreach (var column in columns)
            {
                if (!vectorDatabase.Columns.Contains(column.Name))
                {
                    vectorDatabase.Columns.Add(column.Name, column.DataType);
                }
            }
        }
        private async Task CreateNewDatabase()
        {
            IOManager.ClearConsole();
            IOManager.PrintHeading("Vector Databases");
            IOManager.SendMessage("\nPlease enter a name for the new database file (without extension): ");
            dataFileName = IOManager.ReadLine();
            dataFileName = $"{dataFileName}.json";
            
            if (string.IsNullOrEmpty(dataFileName) || dataFileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                IOManager.SendMessage("\nInvalid file name. Returning to menu.");
                return;
            }

            // This only includes Tech Support dbs for now but program will be generalized to allow the user to customize the db
            IOManager.SendMessage("\nPlease select the type of database you want to create:\n");
            for (int i = 0; i < databaseTypes.Length; i++)
            {
                IOManager.SendMessage($"{i + 1}: {databaseTypes[i]}\n");
            }

            IOManager.SendMessage("\nEnter your choice: ");
            string databaseType = IOManager.ReadLine();
            if (int.TryParse(databaseType, out int databaseChoice) && databaseChoice >= 1 && databaseChoice <= databaseTypes.Length)
            {
                // Create an empty database for tech support (later updates will allow different types of dbs)
                AddDatabaseColumns(techSupportColumns);
                await SaveJsonToFileAsync(DataTableToJson(vectorDatabase), dataFileName);
                IOManager.SendMessage($"\n{databaseTypes[databaseChoice - 1]} database named '{dataFileName}' created successfully.\n");
            }
            else
            {
                IOManager.SendMessage("\nInvalid choice. Returning to menu.\n");
            }
        }

        private async Task LoadDatabase(string[] jsonFiles)
        {
            IOManager.ClearConsole();
            IOManager.PrintHeading("Vector Databases");
            IOManager.SendMessage("\nEnter the number of which database to load, or enter " + (jsonFiles.Length + 1) + " to *Create New Database*:\n");
            for (int i = 0; i < jsonFiles.Length; i++)
            {
                IOManager.SendMessageLine($"{i + 1}. {Path.GetFileName(jsonFiles[i])}");
            }
            IOManager.SendMessageLine($"{jsonFiles.Length + 1}. *Create New Database*");

            string databaseChoice = IOManager.ReadLine();
            if (int.TryParse(databaseChoice, out int choice))
            {
                if (choice >= 1 && choice <= jsonFiles.Length)
                {
                    // Load the selected database
                    string selectedFilePath = jsonFiles[choice - 1];
                    dataFileName = Path.GetFileName(selectedFilePath);
                    string existingJson = await ReadJsonFromFileAsync(selectedFilePath);
                    if (!string.IsNullOrWhiteSpace(existingJson))
                    {
                        DataTable existingTable = JsonToDataTable(existingJson);
                        if (existingTable != null)
                        {
                            vectorDatabase = existingTable;
                            AddDatabaseColumns(techSupportColumns);
                        }
                    }
                }
                else if (choice == jsonFiles.Length + 1)
                {
                    await CreateNewDatabase();
                    jsonFiles = Directory.GetFiles(dataDirectoryPath, "*.json");
                    await LoadDatabase(jsonFiles);
                }
                else
                {
                    IOManager.SendMessage("\nInvalid option selected. Please try again.");
                }
            }
            else
            {
                IOManager.SendMessage("\nPlease enter a valid number.");
            }
        }

        public async Task<long> GetHighestIncidentNumberAsync()
        {
            return await Task.Run(() =>
            {
                if (vectorDatabase == null || vectorDatabase.Rows.Count == 0) { return 0; }
                return vectorDatabase.AsEnumerable().Max(row => Convert.ToInt64(row["incidentNumber"]));
            });
        }

        // LLamaEmbedder generates floats which need to be converted to double due to JSON behavior
        public async Task<double[]> GenerateEmbeddingsAsync(string textToEmbed)
        {
            float[] embeddingsFloat = await modelManager.embedder.GetEmbeddings(textToEmbed);
            double[] embeddingsDouble = embeddingsFloat.Select(f => (double)f).ToArray();
            return embeddingsDouble;
        }

        public string DataTableToJson(DataTable dataTable)
        {
            return JsonConvert.SerializeObject(dataTable, Formatting.Indented);
        }

        public DataTable JsonToDataTable(string json)
        {
            try
            {
                return JsonConvert.DeserializeObject<DataTable>(json);
            }
            catch
            {
                return new DataTable();
            }
        }

        public async Task SaveJsonToFileAsync(string json, string dataFileName)
        {
            string filePath = Path.Combine(dataDirectoryPath, dataFileName);

            if (!Directory.Exists(dataDirectoryPath))
            {
                Directory.CreateDirectory(dataDirectoryPath);
            }

            await File.WriteAllTextAsync(filePath, json);
        }


        public async Task<string> ReadJsonFromFileAsync(string filePath)
        {
            return File.Exists(filePath) ? await File.ReadAllTextAsync(filePath) : string.Empty;
        }


        public DataTable GetVectorDatabase()
        {
            return vectorDatabase;
        }

        // Not needed until I implement additional model familiesl like llama or phi
        //private async Task GenerateMissingEmbeddingsAsync()
        //{
        //    // Generate missing embeddings for the current model type
        //    foreach (DataRow row in vectorDatabase.Rows)
        //    {
        //        if (row[currentModelType] == null)
        //        {
        //            // Generate embeddings based on incidentDetails
        //            IOManager.SendMessage($"Generating missing embeddings for {row["incidentNumber"]}...");
        //            string incidentDetails = row["incidentDetails"].ToString();
        //            double[] newEmbeddings = await GenerateEmbeddingsAsync(incidentDetails);
        //            row[currentModelType] = newEmbeddings;
        //        }
        //    }

        //    string json = DataTableToJson(vectorDatabase);
        //    SaveJsonToFile(json);
        //}

    }
}
