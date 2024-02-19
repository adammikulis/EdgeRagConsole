// This class initializes the vector database
// It needs the current modelManager so it knows what column to add for vector embeddings
// Future iterations will decouple this and instead check for a loaded model before acting on the database


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

        // Factory method
        public static async Task<DatabaseManager> CreateAsync(ModelManager modelManager, string dataDirectoryPath)
        {
            var databaseManager = new DatabaseManager(modelManager, dataDirectoryPath);
            await databaseManager.InitializeAsync();
            return databaseManager;
        }

        // Initialization method
        public async Task InitializeAsync()
        {
            await Task.Run(async () =>
            {
                // Future iterations will allow for custom datatables
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

        // Prevents adding data to columns that don't yet exist
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

            // Prompt for valid filename
            bool validFileName = false;
            while (!validFileName)
            {
                IOManager.SendMessage("\nPlease enter a name for the new database file (without extension): ");
                dataFileName = IOManager.ReadLine().Trim();
                dataFileName = $"{dataFileName}.json";

                if (!string.IsNullOrEmpty(dataFileName) && dataFileName.IndexOfAny(Path.GetInvalidFileNameChars()) < 0)
                {
                    validFileName = true;
                }
                else
                {
                    IOManager.SendMessage("\nInvalid file name. Please try again.");
                }
            }

            // Prompt for valid database type
            bool validChoiceMade = false;
            while (!validChoiceMade)
            {
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
                    validChoiceMade = true;
                }
                else
                {
                    IOManager.SendMessage("\nInvalid choice. Please try again.");
                }
            }
        }

        // Uses Newtonsoft.json to load a previously saved database directly to a DataTable. Future iterations will remove this library and do DataTable -> Dictionary -> Json with just System.Text.Json
        private async Task LoadDatabase(string[] jsonFiles)
        {
            IOManager.ClearConsole();
            IOManager.PrintHeading("Vector Databases");

            bool validChoice = false;
            while (!validChoice)
            {
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
                        validChoice = true;
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
                        validChoice = true;
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
        }

        // Used when generating new incidents for an existing database
        public async Task<long> GetHighestIncidentNumberAsync()
        {
            return await Task.Run(() =>
            {
                if (vectorDatabase == null || vectorDatabase.Rows.Count == 0) { return 0; }
                return vectorDatabase.AsEnumerable().Max(row => Convert.ToInt64(row["incidentNumber"]));
            });
        }

        // LLamaEmbedder generates floats which need to be converted to double due to JSON deserialization behavior
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
            return JsonConvert.DeserializeObject<DataTable>(json);
        }

        // Async method to write to an external file
        public async Task SaveJsonToFileAsync(string json, string dataFileName)
        {
            string filePath = Path.Combine(dataDirectoryPath, dataFileName);

            if (!Directory.Exists(dataDirectoryPath))
            {
                Directory.CreateDirectory(dataDirectoryPath);
            }

            await File.WriteAllTextAsync(filePath, json);
        }

        // Async method to read from an external file
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
