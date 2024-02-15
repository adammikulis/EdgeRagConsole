using System.Data;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using LLama;
using System.Text.RegularExpressions;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private IOManager iOManager;
        private DataTable vectorDatabase;
        private ModelManager modelManager;
        private int numTopMatches;
        public string jsonDbPath;
        public string embeddingColumnName;
        public string summarizedText;
        public long highestIncidentNumber;
        public DatabaseManager(IOManager iOManager, ModelManager modelManager, string jsonDbPath, int numTopMatches)
        {
            this.iOManager = iOManager;
            this.jsonDbPath = jsonDbPath;
            this.modelManager = modelManager;
            this.numTopMatches = numTopMatches;
            vectorDatabase = new DataTable();
            summarizedText = "";
            embeddingColumnName = $"{modelManager.modelName}Embeddings";
        }

        public static async Task<DatabaseManager> CreateAsync(IOManager ioManager, ModelManager modelManager, string jsonDbPath, int numTopMatches )
        {
            var databaseManager = new DatabaseManager(ioManager, modelManager, jsonDbPath, numTopMatches);
            await databaseManager.InitializeAsync();
            return databaseManager;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                DetermineStartingIncidentNumber();
                vectorDatabase.Columns.Add("incidentNumber", typeof(long));
                vectorDatabase.Columns.Add("incidentDetails", typeof(string));
                vectorDatabase.Columns.Add("supportResponse", typeof(string));
                vectorDatabase.Columns.Add("userResponse", typeof(string));
                vectorDatabase.Columns.Add("incidentSolution", typeof(string));
                vectorDatabase.Columns.Add(embeddingColumnName, typeof(double[]));

                if (File.Exists(jsonDbPath))
                {
                    string existingJson = ReadJsonFromFile(jsonDbPath);
                    vectorDatabase = string.IsNullOrWhiteSpace(existingJson) ? new DataTable() : JsonToDataTable(existingJson);
                }
            });
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
                var factEmbeddings = (double[])row[embeddingColumnName];
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


        private long DetermineStartingIncidentNumber()
        {
            if (System.IO.File.Exists(jsonDbPath))
            {
                string existingJson = System.IO.File.ReadAllText(jsonDbPath);
                if (!string.IsNullOrWhiteSpace(existingJson))
                {
                    DataTable existingTable = JsonToDataTable(existingJson);
                    if (existingTable != null && existingTable.Rows.Count > 0)
                    {
                        highestIncidentNumber = existingTable.AsEnumerable().Max(row => Convert.ToInt64(row["incidentNumber"]));
                    }
                }
            }
            return 0;
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

        public void SaveJsonToFile(string json, string filePath)
        {
            string directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            if (File.Exists(filePath))
            {
                File.WriteAllText(filePath, json);
            }
            else
            {
                // Create a new file and write the JSON data
                File.WriteAllText(filePath, json);
            }
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
            return jsonDbPath;
        }
    }
}
