using System.Data;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using LLama;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private IOManager iOManager;
        private DataTable vectorDatabase;
        private ModelManager ModelManager;
        private LLamaEmbedder embedder;
        private string modelName;
        private string jsonDbPath;
        string embeddingColumnName;
        string summarizedText;
        public DatabaseManager(IOManager iOManager, string jsonDbPath, ModelManager modelManager)
        {
            this.iOManager = iOManager;
            this.jsonDbPath = jsonDbPath;
            this.ModelManager = modelManager;
            vectorDatabase = new DataTable();
            summarizedText = "";
            embeddingColumnName = $"{this.modelName}Embeddings";
        }

        public static async Task<DatabaseManager> CreateAsync(IOManager ioManager, string jsonDbPath, ModelManager modelManager)
        {
            var databaseManager = new DatabaseManager(ioManager, jsonDbPath, modelManager);
            await databaseManager.InitializeDatabaseAsync();
            return databaseManager;
        }

        public DataTable GetVectorDatabase()
        {
            return vectorDatabase;
        }

        public async Task InitializeDatabaseAsync()
        {
            await Task.Run(() =>
            {
                vectorDatabase.Columns.Add("incidentNumber", typeof(long));
                vectorDatabase.Columns.Add("incidentDetails", typeof(string));
                vectorDatabase.Columns.Add("incidentResponse", typeof(string));
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
            return await Task.Run(() =>
            {
                float[] embeddingsFloat = embedder.GetEmbeddings(textToEmbed);
                double[] embeddingsDouble = embeddingsFloat.Select(f => (double)f).ToArray();
                return embeddingsDouble;
            });
        }

        public async Task<(string summarizedText, long[] incidentNumbers, double[] scores)> QueryDatabase(string query, int numTopMatches)
        {
            summarizedText = "";
            var queryEmbeddings = await GenerateEmbeddingsAsync(query);
            List<Tuple<double, long, string>> scoresIncidents = new List<Tuple<double, long, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[embeddingColumnName];
                double score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                long incidentNumber = Convert.ToInt64(row["incidentNumber"]);
                string originalText = $"{row["incidentDetails"]} {row["incidentSolution"]}";
                scoresIncidents.Add(new Tuple<double, long, string>(score, incidentNumber, originalText));
            }

            // Sort the scores to find the top matches
            var topMatches = scoresIncidents.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();

            // Extract incident numbers and scores into separate arrays for return
            long[] incidentNumbers = topMatches.Select(m => m.Item2).ToArray();
            double[] scores = topMatches.Select(m => m.Item1).ToArray();

            // Generate summarized text using direct string concatenation
            foreach (var match in topMatches)
            {
                summarizedText += $"{match.Item3} ";
            }

            // Return the summarized text, incident numbers, and their scores
            return (summarizedText, incidentNumbers, scores);
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
    }
}
