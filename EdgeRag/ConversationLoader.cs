using LLama;
using LLama.Common;
using System.Data;

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
        protected string[] prompts;
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
        protected DataTable generatedSyntheticDataTable { get; private set; }
        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationLoader(IInputHandler inputHandler, LLamaWeights model, string modelType, ModelParams modelParams, LLamaEmbedder embedder, LLamaContext context, DataTable vectorDatabase, float temperature, bool useDatabase)
        {
            this.inputHandler = inputHandler;
            this.model = model;
            this.modelParams = modelParams;
            this.embedder = embedder;
            this.context = context;
            this.vectorDatabase = vectorDatabase;
            this.temperature = temperature;
            this.useDatabase = useDatabase;      
            this.modelType = modelType;

            generatedSyntheticDataTable = new DataTable();
            generatedSyntheticDataTable.Columns.Add("Question", typeof(string));
            generatedSyntheticDataTable.Columns.Add("Answer", typeof(string));


            // Still haven't figured out a prompt that allows the network to use its existing knowledge and not just the DB Facts
            prompts = new string[] {
                $"Reply in a natural manner and utilize your existing knowledge. If you don't know the answer, use one of the relevant DB facts in the prompt. Be a friendly, concise, never offensive chatbot to help users learn more about the University of Denver. Query: {query}\n"
            };
            antiPrompts = new string[] { "User:" };
            prompt_number_chosen = 0;
            query = "";
            prompt = "";
            conversation = "";
            num_top_matches = 3;
            InitializeConversation();
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

        public async Task StartChatAsync()
        {
            if (session != null)
            {
                OnMessage?.Invoke("Hello! I am your friendly DU Chatbot, how can I help you today?\n");
                while (true)
                {
                    string embeddingColumnName = useDatabase ? modelType + "Embedding" : string.Empty;
                    prompt = useDatabase ? await QueryDatabase(embeddingColumnName) : await GetPromptWithoutDatabase();
                    string currentMessage = "";
                    await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
                    {
                        // Accumulate text and send it out word by word.
                        currentMessage += text;
                        var lastSpaceIndex = currentMessage.LastIndexOf(' ');
                        if (lastSpaceIndex != -1) // Ensure there's at least one word to send.
                        {
                            // Send everything up to the last space.
                            var messageToSend = currentMessage.Substring(0, lastSpaceIndex + 1);
                            OnMessage?.Invoke(messageToSend);

                            // Keep the remaining part that hasn't been sent.
                            currentMessage = currentMessage.Substring(lastSpaceIndex + 1);
                        }
                    }
                    // After the loop, send any remaining text not ending with a space.
                    if (!string.IsNullOrEmpty(currentMessage))
                    {
                        OnMessage?.Invoke(currentMessage);
                    }
                    conversation += prompt;
                    prompt = "";
                }
            }
        }


        // This is used for RAG, appends user query and vector db facts to system prompt
        private async Task<string> QueryDatabase(string embeddingColumnName)
        {
            // How many matches to present to the LLM in its prompt
            string queried_prompt = prompts[prompt_number_chosen];


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

        private async Task<string> GenerateQuestionAsync(int questionNumber)
        {
            float questionTemperature = 0.75f; // Set the temperature for creativity
            OnMessage?.Invoke($"Generating tech support question {questionNumber}...\n");

            // Themes can represent common categories of tech support issues
            string[] themes = new string[] { "an Apple device issue", "an Android device problem", "a Windows device malfunction" };
            Random rand = new Random();

            // Selecting a random theme for a tech support question
            string selectedTheme = themes[rand.Next(themes.Length)];

            // Crafting a prompt that asks the LLM to generate a tech support ticket question
            // This prompt is designed to mimic the type of queries tech support might receive, focusing on common user issues.
            string llmPrompt = $"Think of a user facing a typical {selectedTheme}. Generate a detailed tech support ticket question they might ask. Include the question title, question details about the issue(s), and at most 3 troubleshooting steps they've already attempted.";

            // Asking the LLM to generate a question based on the prompt
            string generatedQuestion = await AskLLMForQuestion(llmPrompt, questionTemperature);

            return generatedQuestion.Trim(); // Trim to ensure no leading/trailing whitespace
        }


        private async Task<string> AskLLMForQuestion(string prompt, float questionTemperature)
        {
            string question = "";
            // Assuming session.ChatAsync is the way you interact with the LLM
            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = questionTemperature, AntiPrompts = antiPrompts }))
            {
                question += text;
            }
            return question.Trim();
        }

        // Helper method to generate an answer for a given question using the LLM
        private async Task<string> GenerateAnswerForQuestion(string question)
        {
            // Set the initial prompt for the LLM
            string systemPrompt = "As Tech Support: Answer the user's tech support question";
            string prompt = $"{systemPrompt} Question: {question}\r\nAnswer:";
            string answer = "";

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                answer += text;
            }

            return answer;
        }

        public async Task GenerateAndStoreSyntheticData(int n)
        {
            for (int i = 0; i < n; i++)
            {
                // Await the asynchronous call to generate a question
                string syntheticQuestion = await GenerateQuestionAsync(i + 1);
                string syntheticAnswer = await GenerateAnswerForQuestion(syntheticQuestion);

                // Store the question and answer in the DataTable
                DataRow newRow = generatedSyntheticDataTable.NewRow();
                newRow["Question"] = syntheticQuestion;
                newRow["Answer"] = syntheticAnswer;
                generatedSyntheticDataTable.Rows.Add(newRow);
            }
        }


        public void PrintSyntheticDataTableHead(int n)
        {
            // Check if the DataTable has any rows
            if (generatedSyntheticDataTable.Rows.Count == 0)
            {
                OnMessage?.Invoke("DataTable is empty.");
                return;
            }
            OnMessage?.Invoke("\n");
            // Print column headers
            foreach (DataColumn column in generatedSyntheticDataTable.Columns)
            {
                OnMessage?.Invoke($"{column.ColumnName}\t");
            }
            OnMessage?.Invoke("\n");

            // Iterate over the first n rows or the total number of rows, whichever is smaller
            int rowsToPrint = Math.Min(n, generatedSyntheticDataTable.Rows.Count);
            for (int i = 0; i < rowsToPrint; i++)
            {
                // Print each column's value for the current row
                foreach (DataColumn column in generatedSyntheticDataTable.Columns)
                {
                    OnMessage?.Invoke($"{generatedSyntheticDataTable.Rows[i][column]}\t");
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
        public ConversationLoaderConsole(IInputHandler inputHandler, LLamaWeights model, string modelType, ModelParams modelParams, LLamaEmbedder embedder, LLamaContext context, DataTable dt, float temperature, bool useDatabase) : base(inputHandler, model, modelType, modelParams, embedder, context, dt, temperature, useDatabase)
        {
            OnMessage += Console.Write;
        }
    }
}
