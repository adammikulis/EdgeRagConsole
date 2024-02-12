using System.Data;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using LLama;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private DataTable vectorDatabase = new DataTable();
        private LLamaEmbedder embedder;
        private string modelType;
        private string jsonDbPath;
        string embeddingColumnName;
        public DatabaseManager(string jsonDbPath, LLamaEmbedder embedder, string modelType)
        {
            this.jsonDbPath = jsonDbPath;
            this.embedder = embedder;
            this.modelType = modelType;
            embeddingColumnName = $"{modelType}Embeddings";
        }

        public string ModelType
        {
            get { return modelType; }
        }

        public DataTable GetVectorDatabase()
        {
            return vectorDatabase;
        }

        public async Task InitializeDatabaseAsync()
        {
            vectorDatabase.Columns.Add("incidentNumber", typeof(long));
            vectorDatabase.Columns.Add("incidentDetails", typeof(string));
            vectorDatabase.Columns.Add("incidentResponse", typeof(string));
            vectorDatabase.Columns.Add("incidentSolution", typeof(string));
            vectorDatabase.Columns.Add(embeddingColumnName, typeof(double[]));

            // Check if the database JSON file exists; if not, initialize a new DataTable
            if (File.Exists(jsonDbPath))
            {
                string existingJson = ReadJsonFromFile(jsonDbPath);
                vectorDatabase = string.IsNullOrWhiteSpace(existingJson) ? new DataTable() : JsonToDataTable(existingJson);
            }
        }

        public async Task<double[]> GenerateEmbeddingsAsync(string textToEmbed)
        {
            // Directly call the synchronous method without await
            float[] embeddingsFloat = embedder.GetEmbeddings(textToEmbed);
            // Convert each float to double
            double[] embeddingsDouble = embeddingsFloat.Select(f => (double)f).ToArray();
            return embeddingsDouble;
        }

        public async Task<string> QueryDatabase(string query, int numTopMatches)
        {
            var queryEmbeddings = await GenerateEmbeddingsAsync(query);
            List<Tuple<double, string>> scores = new List<Tuple<double, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[embeddingColumnName];
                var score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                string originalText = $"{row["incidentDetails"]} {row["incidentResponse"]} {row["incidentSolution"]}";
                scores.Add(new Tuple<double, string>(score, originalText));
            }

            // Sort the scores to find the top matches
            var topMatches = scores.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();
            var queriedPrompt = query;

            // Append the top matching texts to the queriedPrompt
            foreach (var match in topMatches)
            {
                queriedPrompt += $"{match.Item2}\n";
            }

            return queriedPrompt;
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
