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

        public SyntheticDataGenerator syntheticDataGenerator;

        protected DataTable vectorDatabase { get; private set; }
        protected DataTable generatedITSyntheticDataTable { get; private set; }
        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationLoader(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, float temperature, bool useDatabase, string[] antiPrompts)
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
            this.antiPrompts = antiPrompts;

            // Still haven't figured out a prompt that allows the network to use its existing knowledge and not just the DB Facts
            systemMessages = new string[] {
                $"Reply in a natural manner and utilize your existing knowledge. If you don't know the answer, use one of the relevant DB facts in the prompt. Be a friendly, concise, never offensive chatbot to help users learn more about the University of Denver. Query: {query}\n"
            };
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
            syntheticDataGenerator = new SyntheticDataGenerator(session, inputHandler, temperature, antiPrompts);

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
                    string response = await InteractWithModelAsync(promptInstructions, prompt, temperature, antiPrompts);

                    // Send the complete response
                    OnMessage?.Invoke(response);

                    conversation += prompt + " " + response;
                }
            }
        }
        private async Task<string> InteractWithModelAsync(string promptInstructions, string prompt, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n"; // Ensure the session is initialized
            prompt = $"{promptInstructions} {prompt}";
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

        public SyntheticDataGenerator SyntheticDataGenerator
        {
            get { return syntheticDataGenerator; }
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
        public ConversationLoaderConsole(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, float temperature, bool useDatabase, string[] antiPrompts) : base(inputHandler, modelLoaderOutputs, temperature, useDatabase, antiPrompts)
        {
            OnMessage += Console.Write;
        }
    }
}
