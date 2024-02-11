using LLama;
using LLama.Common;
using System.Data;
using Newtonsoft.Json;
using System.IO;

namespace EdgeRag
{
    public class ConversationLoader
    {
        protected bool useDatabase;
        protected string? directoryPath;
        protected string? selectedModelPath;
        protected string? fullModelName;
        protected string? modelType;
        protected string[]? facts;
        protected uint? contextSize;
        protected string[] systemMessages;
        private string[] antiPrompts;
        private float temperature;
        protected int prompt_number_chosen;
        protected string conversation;
        protected string query;
        protected string prompt;
        protected int num_top_matches;

        protected LLamaWeights? model;
        protected ModelParams? modelParams;
        protected LLamaEmbedder? embedder;
        protected LLamaContext? context;
        protected InteractiveExecutor? executor;
        protected ChatSession? session;

        protected DataTable vectorDatabase { get; private set; }
        protected DataTable generatedITSyntheticDataTable { get; private set; }
        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationLoader(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, float temperature, bool useDatabase)
        {
            this.inputHandler = inputHandler;
            this.model = modelLoaderOutputs.model;
            this.modelType = modelLoaderOutputs.modelType;
            this.modelParams = modelLoaderOutputs.modelParams;
            this.embedder = modelLoaderOutputs.embedder;
            this.context = modelLoaderOutputs.context;
            this.vectorDatabase = modelLoaderOutputs.embeddingsTable;
            this.temperature = temperature;
            this.useDatabase = useDatabase;      
            this.modelType = modelLoaderOutputs.modelType;

            // Still haven't figured out a prompt that allows the network to use its existing knowledge and not just the DB Facts
            systemMessages = new string[] {
                $"Reply in a natural manner and utilize your existing knowledge. If you don't know the answer, use one of the relevant DB facts in the prompt. Be a friendly, concise, never offensive chatbot to help users learn more about the University of Denver. Query: {query}\n"
            };
            antiPrompts = new string[] { "User:" };
            prompt_number_chosen = 0;
            query = "";
            prompt = "";
            conversation = "";
            num_top_matches = 3;
            InitializeConversation();
            InitializeITSyntheticDataTable();
        }

        protected void InitializeConversation()
        {
            if (model == null || modelParams == null)
            {
                OnMessage?.Invoke("Model or modelParams is null. Cannot initialize conversation.");
                return;
            }

            context = model.CreateContext(modelParams);
            if (context == null)
            {
                OnMessage?.Invoke("Failed to create context. Cannot initialize conversation.");
                return;
            }

            executor = new InteractiveExecutor(context);
            session = new ChatSession(executor);

            if (session == null)
            {
                OnMessage?.Invoke("Failed to create chat session.");
            }
        }

        public async Task StartChatAsync(string promptInstructions, string prompt)
        {
            if (session != null)
            {
                OnMessage?.Invoke("Chat session started, please input query:\n");
                while (true)
                {
                    string embeddingColumnName = useDatabase ? $"{modelType}Embedding" : string.Empty;
                    prompt = useDatabase ? await QueryDatabase(embeddingColumnName) : await GetPromptWithoutDatabase();

                    // Use the reusable method
                    string response = await InteractWithModelAsync(promptInstructions, prompt, temperature);

                    // Send the complete response
                    OnMessage?.Invoke(response);

                    conversation += prompt + " " + response;
                }
            }
        }
        private async Task<string> InteractWithModelAsync(string promptInstructions, string prompt,  float temperature)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n"; // Ensure the session is initialized
            prompt = $"{promptInstructions} { prompt }";
            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response.Trim();
        }

        // This is used for RAG, appends user query and vector db facts to system prompt
        private async Task<string> QueryDatabase(string embeddingColumnName)
        {
            // How many matches to present to the LLM in its prompt
            string queried_prompt = systemMessages[prompt_number_chosen];


            query = await inputHandler.ReadLineAsync();
            if (string.IsNullOrWhiteSpace(query) || query == "exit" || query == "quit") Environment.Exit(0);
            Console.WriteLine("\nQuerying database and processing with LLM...\n");

            // Get embedding for the query to match against vector DB
            var queryEmbeddings = embedder.GetEmbeddings(query);
            List<Tuple<double, string>> scores = new List<Tuple<double, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (float[])row[embeddingColumnName];
                var score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                scores.Add(new Tuple<double, string>(score, (string)row["originalText"]));
            }

            var topMatches = scores.OrderByDescending(s => s.Item1).Take(num_top_matches).ToList();

            for (int i = 0; i < topMatches.Count; i++)
            {
                queried_prompt += $"DB Fact {i + 1}: {topMatches[i].Item2}\n";
            }
            queried_prompt += "Answer:";
            return queried_prompt;
        }

        private async Task<string> GetPromptWithoutDatabase()
        {
            query = await inputHandler.ReadLineAsync();
            if (string.IsNullOrWhiteSpace(query) || query == "exit" || query == "quit") Environment.Exit(0);

            // Directly return the user's query or modify as needed for your application
            string queriedPrompt = $"User: {query}\nAnswer:";
            return queriedPrompt;
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

        private void InitializeITSyntheticDataTable()
        {
            generatedITSyntheticDataTable = new DataTable();
            generatedITSyntheticDataTable.Columns.Add("incidentNumber", typeof(int));
            generatedITSyntheticDataTable.Columns.Add("incidentTitle", typeof(string));
            generatedITSyntheticDataTable.Columns.Add("incidentDetails", typeof(string));
            generatedITSyntheticDataTable.Columns.Add("supportResponse", typeof(string));
            generatedITSyntheticDataTable.Columns.Add("userFinalResponse", typeof(string));
        }

        public async Task<DataTable> GenerateITDataPipeline(int n)
        {
            for (int i = 0; i < n; i++)
            {
                OnMessage?.Invoke($"Generating item {i + 1}...\n");
                float userTemperature = 0.75f;
                float supportTemperature = 0.25f;
                string[] themes = { "an Apple device issue", "an Android device problem", "a Windows device malfunction" };
                Random rand = new Random();
                string selectedTheme = themes[rand.Next(themes.Length)];
                
                DataRow newRow = generatedITSyntheticDataTable.NewRow();
                newRow["incidentNumber"] = i + 1; // Increment for each incident

                // Generates initial incident report title
                string prompt = "";
                string promptInstructions = $"Generate an incident report title for {selectedTheme} a user may submit to the IT Help Desk. Do not say Incident Report Title";
                string incidentTitle = await InteractWithModelAsync(promptInstructions, prompt, userTemperature);
                newRow["incidentTitle"] = incidentTitle;

                // Generate incident details based on the incident
                promptInstructions = $"Generate tech support incident details for {incidentTitle} using no more than 50 words";
                string incidentDetails = await InteractWithModelAsync(promptInstructions, incidentTitle, userTemperature);
                newRow["incidentDetails"] = incidentDetails;

                // Generate IT support's response
                promptInstructions = $"As Tech Support: Answer the user's tech support incident civily using no more than 100 words: {incidentDetails} ";
                string supportResponse = await InteractWithModelAsync(promptInstructions, incidentDetails, supportTemperature);
                newRow["supportResponse"] = supportResponse;

                // Generate user's final response based on IT support's help
                promptInstructions = $"As the user only, troubleshoot based on the supplied steps and respond with what the solution ended up as the final message using no more than 50 words. Steps: {supportResponse}";
                string userFinalResponse = await InteractWithModelAsync(promptInstructions, supportResponse, userTemperature);
                newRow["userFinalResponse"] = userFinalResponse;

                generatedITSyntheticDataTable.Rows.Add(newRow);
            }
            string syntheticDataOutputPath = @"C:\ai\data\synthetic\syntheticData.json";
            string json = DataTableToJson(generatedITSyntheticDataTable);
            SaveJsonToFile(json, syntheticDataOutputPath);

            return generatedITSyntheticDataTable;
        }

        public void PrintSyntheticDataTable(int n)
        {
            // Check if the DataTable has any rows
            if (generatedITSyntheticDataTable.Rows.Count == 0)
            {
                OnMessage?.Invoke("DataTable is empty.");
                return;
            }
            OnMessage?.Invoke("\n");
            // Print column headers
            foreach (DataColumn column in generatedITSyntheticDataTable.Columns)
            {
                OnMessage?.Invoke($"{column.ColumnName}\t");
            }
            OnMessage?.Invoke("\n");

            // Iterate over the first n rows or the total number of rows, whichever is smaller
            int rowsToPrint = Math.Min(n, generatedITSyntheticDataTable.Rows.Count);
            for (int i = 0; i < rowsToPrint; i++)
            {
                // Print each column's value for the current row
                foreach (DataColumn column in generatedITSyntheticDataTable.Columns)
                {
                    OnMessage?.Invoke($"{generatedITSyntheticDataTable.Rows[i][column]}\t");
                }
                OnMessage?.Invoke("\n"); // Move to the next line after printing all columns for a row
            }
        }
    }

    public interface IInputHandler
    {
        Task<string> ReadLineAsync();
    }

    public class ConsoleInputHandler : IInputHandler
    {
        public async Task<string> ReadLineAsync()
        {
            return await Task.Run(() => Console.ReadLine());
        }
    }

    public class ConversationLoaderConsole : ConversationLoader
    {
        public ConversationLoaderConsole(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, float temperature, bool useDatabase) : base(inputHandler, modelLoaderOutputs, temperature, useDatabase)
        {
            OnMessage += Console.Write;
        }
    }
}
