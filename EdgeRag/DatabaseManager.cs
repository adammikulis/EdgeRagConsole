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
            vectorDatabase.Columns.Add(embeddingColumnName, typeof(float[])); // Change to float[]
            vectorDatabase.Columns.Add("incidentNumber", typeof(long));
            vectorDatabase.Columns.Add("incidentDetails", typeof(string));
            vectorDatabase.Columns.Add("incidentResponse", typeof(string));
            vectorDatabase.Columns.Add("incidentSolution", typeof(string));

            // Check if the database JSON file exists; if not, initialize a new DataTable
            if (File.Exists(jsonDbPath))
            {
                string existingJson = ReadJsonFromFile(jsonDbPath);
                vectorDatabase = string.IsNullOrWhiteSpace(existingJson) ? new DataTable() : JsonToDataTable(existingJson);
            }
        }

        public async Task<float[]> GenerateEmbeddingsAsync(string textToEmbed)
        {
            return embedder.GetEmbeddings(textToEmbed);
        }

        public async Task<string> QueryDatabase(string query, int numTopMatches)
        {
            var queryEmbeddings = await GenerateEmbeddingsAsync(query);
            List<Tuple<double, string>> scores = new List<Tuple<double, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = ((IEnumerable<double>)row[embeddingColumnName]).Select(x => (float)x).ToArray(); // Compensates for implicit JSON conversion
                var score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                string textToEmbed = $"{row["incidentDetails"]} {row["incidentResponse"]} {row["incidentSolution"]}";
                scores.Add(new Tuple<double, string>(score, textToEmbed));
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
                var settings = new JsonSerializerSettings
                {
                    Converters = { new SingleArrayConverter() }
                };

                return JsonConvert.DeserializeObject<DataTable>(json, settings);
            }
            catch
            {
                return new DataTable();
            }
        }

        public void SaveJsonToFile(string json, string filePath, bool overwrite)
        {
            string directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            if (File.Exists(filePath) && overwrite)
            {
                // Overwrite the existing file with the new JSON data
                File.WriteAllText(filePath, json);
            }
            else if (File.Exists(filePath))
            {
                // Append the new JSON data to the existing file
                using (StreamWriter file = File.AppendText(filePath))
                {
                    file.WriteLine(json);
                }
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

        public class SingleArrayConverter : JsonConverter
        {
            public override bool CanConvert(Type objectType)
            {
                return (objectType == typeof(float[]));
            }

            public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
            {
                JArray array = JArray.Load(reader);
                return array.Select(jv => jv.Value<float>()).ToArray();
            }

            public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
            {
                // Leave this method empty since we are only deserializing float arrays
            }
        }
    }
}
