using System.Data;
using LLama;
using Newtonsoft.Json;

namespace EdgeRag
{
    public class DatabaseManager
    {
        private DataTable vectorDatabase = new DataTable();
        private LLamaEmbedder embedder;
        private string modelType;
        private string jsonDbPath;

        public DatabaseManager(string jsonDbPath, LLamaEmbedder embedder, string modelType)
        {
            this.jsonDbPath = jsonDbPath;
            this.embedder = embedder;
            this.modelType = modelType;
            InitializeDataTable();
        }

        public string ModelType
        {
            get { return modelType; }
        }

        private void InitializeDataTable()
        {
            vectorDatabase = JsonToDataTable(ReadJsonFromFile(jsonDbPath));
            // User's Json likely will not have embedding columns already
            if (!vectorDatabase.Columns.Contains("llamaEmbedding")) vectorDatabase.Columns.Add("llamaEmbedding", typeof(float[]));
            if (!vectorDatabase.Columns.Contains("mistralEmbedding")) vectorDatabase.Columns.Add("mistralEmbedding", typeof(float[]));
            if (!vectorDatabase.Columns.Contains("mixtralEmbedding")) vectorDatabase.Columns.Add("mixtralEmbedding", typeof(float[]));
            if (!vectorDatabase.Columns.Contains("phiEmbedding")) vectorDatabase.Columns.Add("phiEmbedding", typeof(float[]));

        }

        public async Task<string> QueryDatabase(string query, int numTopMatches)
        {
            // Asynchronously generating embeddings might be beneficial if the operation is CPU-bound and you're considering offloading it to a background thread.
            // However, since LLamaEmbedder.GetEmbeddings is likely a CPU-bound synchronous method, using Task.Run for CPU-bound operations is a contentious choice and generally not recommended.
            // For demonstration purposes and future-proofing for potentially async operations:
            var queryEmbeddings = await Task.Run(() => GenerateEmbeddings(query));

            var scores = new List<Tuple<double, string>>();
            string embeddingColumnName = $"{modelType}Embedding";

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (float[])row[embeddingColumnName];
                var score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                scores.Add(new Tuple<double, string>(score, (string)row["originalText"]));
            }

            var topMatches = scores.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();
            var queriedPrompt = query; // Start with the original query for appending DB facts

            foreach (var match in topMatches)
            {
                queriedPrompt += $"DB Fact: {match.Item2}\n";
            }

            return queriedPrompt; // No need to use Task.FromResult since we're using await
        }

        public float[] GenerateEmbeddings(string textToEmbed)
        {
            return embedder.GetEmbeddings(textToEmbed);
        }

        public string DataTableToJson(DataTable dataTable)
        {
            string json = JsonConvert.SerializeObject(dataTable, Formatting.Indented);
            return json;
        }

        public DataTable JsonToDataTable(string json)
        {
            DataTable dataTable = JsonConvert.DeserializeObject<DataTable>(json);
            return dataTable;
        }

        public void SaveJsonToFile(string json, string filePath)
        {
            // Create directory if it doesn't exist
            string directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write the JSON string to the file
            File.WriteAllText(filePath, json);
        }

        public string ReadJsonFromFile(string filePath)
        {
            // Read the JSON string from the file
            string json = File.ReadAllText(filePath);
            return json;
        }
    }
}
