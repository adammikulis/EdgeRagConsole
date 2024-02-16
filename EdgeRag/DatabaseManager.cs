using System;
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
        public string summarizedText;
        public long highestIncidentNumber;

        public DatabaseManager(ModelManager modelManager, string dataDirectoryPath, string dataFileName, int numTopMatches)
        {
            this.dataDirectoryPath = dataDirectoryPath;
            this.dataFileName = dataFileName;
            this.modelManager = modelManager;
            this.numTopMatches = numTopMatches;
            vectorDatabase = new DataTable();
            summarizedText = "";

            // Add the fixed columns to the DataTable
            vectorDatabase.Columns.Add("incidentNumber", typeof(long));
            vectorDatabase.Columns.Add("incidentDetails", typeof(string));
            vectorDatabase.Columns.Add("supportResponse", typeof(string));
            vectorDatabase.Columns.Add("incidentSolution", typeof(string));
            vectorDatabase.Columns.Add("codellama", typeof(double[]));
            vectorDatabase.Columns.Add("llama", typeof(double[]));
            vectorDatabase.Columns.Add("mistral", typeof(double[]));
            vectorDatabase.Columns.Add("mixtral", typeof(double[]));
            vectorDatabase.Columns.Add("phi", typeof(double[]));
        }

        public static async Task<DatabaseManager> CreateAsync(ModelManager modelManager, string dataDirectoryPath, string dataFileName, int numTopMatches)
        {
            var databaseManager = new DatabaseManager(modelManager, dataDirectoryPath, dataFileName, numTopMatches);
            await databaseManager.InitializeAsync();
            return databaseManager;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(async () =>
            {
                string filePath = Path.Combine(dataDirectoryPath, dataFileName);
                if (File.Exists(filePath))
                {
                    string existingJson = ReadJsonFromFile(filePath);
                    if (!string.IsNullOrWhiteSpace(existingJson))
                    {
                        DataTable existingTable = JsonToDataTable(existingJson);
                        if (existingTable != null && existingTable.Rows.Count > 0)
                        {
                            // Populate vectorDatabase with existing data
                            vectorDatabase = existingTable.Clone();

                            // Populate vectorDatabase with existing data
                            foreach (DataRow row in existingTable.Rows)
                            {
                                vectorDatabase.ImportRow(row);
                            }

                            await GenerateMissingEmbeddingsAsync();
                            highestIncidentNumber = existingTable.AsEnumerable().Max(row => Convert.ToInt64(row["incidentNumber"]));
                        }
                    }
                }
            });
        }

        private async Task GenerateMissingEmbeddingsAsync()
        {
            // Generate missing embeddings for the current model type
            string currentModelType = modelManager.modelType;
            foreach (DataRow row in vectorDatabase.Rows)
            {
                var embeddings = row[currentModelType] as double[];
                if (embeddings == null || embeddings.Length == 0)
                {
                    // Generate embeddings based on incidentDetails
                    IOManager.SendMessage($"Generating missing embeddings for {row["incidentNumber"]}...");
                    string incidentDetails = row["incidentDetails"].ToString();
                    double[] newEmbeddings = await GenerateEmbeddingsAsync(incidentDetails);
                    row[currentModelType] = newEmbeddings;
                }
            }

            // Serialize the DataTable to JSON with correct embedding data
            string json = DataTableToJson(vectorDatabase);
            SaveJsonToFile(json);
        }

        public async Task<double[]> GenerateEmbeddingsAsync(string textToEmbed)
        {
            float[] embeddingsFloat = await modelManager.embedder.GetEmbeddings(textToEmbed);
            double[] embeddingsDouble = embeddingsFloat.Select(f => (double)f).ToArray();
            return embeddingsDouble;
        }

        public async Task<(string summarizedText, long[] incidentNumbers, double[] scores)> QueryDatabase(string query)
        {
            summarizedText = "";
            // Check if the DataTable is empty
            if (vectorDatabase.Rows.Count == 0)
            {
                return (summarizedText, new long[0], new double[0]);
            }

            var queryEmbeddings = await GenerateEmbeddingsAsync(query);
            List<Tuple<double, long, string>> scoresIncidents = new List<Tuple<double, long, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[modelManager.modelType];
                double score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                long incidentNumber = Convert.ToInt64(row["incidentNumber"]);
                string originalText = row["incidentSolution"].ToString();
                scoresIncidents.Add(new Tuple<double, long, string>(score, incidentNumber, originalText));
            }

            // If no matches were found, return early with empty arrays
            if (scoresIncidents.Count == 0)
            {
                return (summarizedText, new long[0], new double[0]);
            }

            var topMatches = scoresIncidents.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();
            long[] incidentNumbers = topMatches.Select(m => m.Item2).ToArray();
            double[] scores = topMatches.Select(m => m.Item1).ToArray();

            summarizedText = topMatches.Count > 0 ? $"{topMatches[0].Item3} " : "";
            return (summarizedText, incidentNumbers, scores);
        }

        public string DataTableToJson(DataTable dataTable)
        {
            List<Dictionary<string, object>> rows = new List<Dictionary<string, object>>();
            foreach (DataRow row in dataTable.Rows)
            {
                Dictionary<string, object> dictRow = new Dictionary<string, object>();
                foreach (DataColumn column in dataTable.Columns)
                {
                    if (row[column] is double[])
                    {
                        // Explicitly serialize double[] as a JSON array
                        double[] embeddings = (double[])row[column];
                        string jsonArray = JsonConvert.SerializeObject(embeddings);
                        dictRow[column.ColumnName] = JsonConvert.DeserializeObject(jsonArray);
                    }
                    else
                    {
                        dictRow[column.ColumnName] = row[column];
                    }
                }
                rows.Add(dictRow);
            }
            return JsonConvert.SerializeObject(rows, Formatting.Indented);
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

        public void SaveJsonToFile(string json)
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

        public string GetJsonDbPath()
        {
            return Path.Combine(dataDirectoryPath, dataFileName);
        }
    }
}
