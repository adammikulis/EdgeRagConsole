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

        protected DataTable dt { get; private set; }
        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationLoader(IInputHandler inputHandler, LLamaWeights model, string modelType, ModelParams modelParams, LLamaEmbedder embedder, LLamaContext context, DataTable dt, bool useDatabase)
        {
            this.inputHandler = inputHandler;
            this.model = model;
            this.modelParams = modelParams;
            this.embedder = embedder;
            this.context = context;
            this.dt = dt;
            this.useDatabase = useDatabase;      
            this.modelType = modelType;

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
                float temperature = 0.25f;
                OnMessage?.Invoke("Hello! I am your friendly DU Chatbot, how can I help you today?\n");
                while (true)
                {
                    string embeddingColumnName = useDatabase ? modelType + "Embedding" : string.Empty;
                    prompt = useDatabase ? await QueryDatabase(embeddingColumnName) : await GetPromptWithoutDatabase();
                    await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
                    {
                        Console.Write(text);
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

            foreach (DataRow row in dt.Rows)
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
        public ConversationLoaderConsole(IInputHandler inputHandler, LLamaWeights model, string modelType, ModelParams modelParams, LLamaEmbedder embedder, LLamaContext context, DataTable dt, bool useDatabase, bool generateSyntheticData) : base(inputHandler, model, modelType, modelParams, embedder, context, dt, useDatabase, generateSyntheticData)
        {
            OnMessage += Console.WriteLine;
        }
    }
}
