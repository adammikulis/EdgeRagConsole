﻿using System;
using System.Data;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private DataTable vectorDatabase;
        private ModelManager modelManager;
        private int numTopMatches;
        public string dataDirectoryPath;
        public string dataFileName;
        public string[] databaseTypes;
        private List<(string Name, Type DataType)> techSupportColumns;

        public DatabaseManager(ModelManager modelManager, string dataDirectoryPath, int numTopMatches)
        {
            this.dataDirectoryPath = dataDirectoryPath;
            this.modelManager = modelManager;
            this.numTopMatches = numTopMatches;
            vectorDatabase = new DataTable();
            databaseTypes = new string[] { "Tech Support" };
            string dataFileName = "";

        }

        public static async Task<DatabaseManager> CreateAsync(ModelManager modelManager, string dataDirectoryPath, int numTopMatches)
        {
            var databaseManager = new DatabaseManager(modelManager, dataDirectoryPath, numTopMatches);
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
                // List all .json files in the directory
                var jsonFiles = Directory.GetFiles(dataDirectoryPath, "*.json");
                if (jsonFiles.Length > 0)
                {
                    ListDatabases(jsonFiles);
                    await LoadDatabase(jsonFiles);
                }
                else
                {
                    IOManager.SendMessage("No database found.\n");
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
            IOManager.SendMessage("Please enter a name for the new database file (without extension): ");
            dataFileName = await IOManager.ReadLineAsync();
            dataFileName = $"{dataFileName}.json";
            // Basic validation for custom file name
            if (string.IsNullOrEmpty(dataFileName) || dataFileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                IOManager.SendMessage("Invalid file name. Returning to the previous menu.");
                return;
            }

            // Define the available database types
            

            IOManager.SendMessage("\nPlease select the type of database you want to create:\n");
            for (int i = 0; i < databaseTypes.Length; i++)
            {
                IOManager.SendMessage($"{i + 1}: {databaseTypes[i]}\n");
            }

            IOManager.SendMessage("\nEnter your choice: ");
            string databaseType = await IOManager.ReadLineAsync();
            if (int.TryParse(databaseType, out int databaseChoice) && databaseChoice >= 1 && databaseChoice <= databaseTypes.Length)
            {
                // Create an empty database for tech support (later updates will allow different types of dbs)
                AddDatabaseColumns(techSupportColumns);
                SaveJsonToFile(DataTableToJson(vectorDatabase), dataFileName);
                IOManager.SendMessage($"\n{databaseTypes[databaseChoice - 1]} database named '{dataFileName}' created successfully.\n");
            }
            else
            {
                IOManager.SendMessage("\nInvalid choice. Returning to the previous menu.\n");
            }
        }

        private async Task LoadDatabase(string[] jsonFiles)
        {
            IOManager.SendMessage("\nChoose which database to load (enter the number):\n");
            string databaseChoice = await IOManager.ReadLineAsync();
            if (int.TryParse(databaseChoice, out int choice) && choice >= 1 && choice <= jsonFiles.Length)
            {
                // Load the selected database
                string selectedFilePath = jsonFiles[choice - 1];
                dataFileName = Path.GetFileName(selectedFilePath);
                string existingJson = ReadJsonFromFile(selectedFilePath);
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
        }

        private static void ListDatabases(string[] jsonFiles)
        {
            IOManager.SendMessage("\nAvailable databases:\n");
            for (int i = 0; i < jsonFiles.Length; i++)
            {
                IOManager.SendMessage($"{i + 1}: {Path.GetFileName(jsonFiles[i])}\n");
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

        public async Task<(string summarizedText, long[] incidentNumbers, double[] scores)> QueryDatabase(string prompt)
        {
            // Check if the DataTable is empty
            if (vectorDatabase.Rows.Count == 0)
            {
                return (prompt, new long[0], new double[0]);
            }

            var queryEmbeddings = await GenerateEmbeddingsAsync(prompt);
            List<Tuple<double, long, string>> scoresIncidents = new List<Tuple<double, long, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[modelManager.selectedModelType];
                double score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                long incidentNumber = Convert.ToInt64(row["incidentNumber"]);
                string originalText = row["incidentSolution"].ToString();
                scoresIncidents.Add(new Tuple<double, long, string>(score, incidentNumber, originalText));
            }

            // If no matches were found, return early with empty arrays
            if (scoresIncidents.Count == 0)
            {
                return (prompt, new long[0], new double[0]);
            }

            var topMatches = scoresIncidents.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();
            long[] incidentNumbers = topMatches.Select(m => m.Item2).ToArray();
            double[] scores = topMatches.Select(m => m.Item1).ToArray();

            prompt = topMatches.Count > 0 ? $"{topMatches[0].Item3} " : "";
            return (prompt, incidentNumbers, scores);
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

        public void SaveJsonToFile(string json, string dataFileName)
        {
            string filePath = Path.Combine(dataDirectoryPath, dataFileName);

            if (!Directory.Exists(dataDirectoryPath))
            {
                Directory.CreateDirectory(dataDirectoryPath);
            }

            File.WriteAllText(filePath, json);
        }

        public string ReadJsonFromFile(string filePath)
        {
            return File.Exists(filePath) ? File.ReadAllText(filePath) : string.Empty;
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
