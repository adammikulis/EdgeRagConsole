using LLama;
using LLama.Common;
using System.Data;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class ConversationManager
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
        protected int numTopMatches;

        protected LLamaWeights? model;
        protected ModelParams? modelParams;
        protected LLamaEmbedder? embedder;
        protected LLamaContext? context;
        protected InteractiveExecutor? executor;
        protected ChatSession? session;
        public DatabaseManager? databaseManager; 
        public SyntheticDataGenerator? syntheticDataGenerator;

        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        // Adjust the constructor to accept DatabaseManager as nullable for flexibility
        public ConversationManager(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, DatabaseManager? databaseManager, float temperature, string[] antiPrompts, int numTopMatches)
        {
            this.inputHandler = inputHandler;
            this.model = modelLoaderOutputs.model;
            this.modelType = modelLoaderOutputs.modelType;
            this.modelParams = modelLoaderOutputs.modelParams;
            this.embedder = modelLoaderOutputs.embedder;
            this.context = modelLoaderOutputs.context;
            this.temperature = temperature;
            this.antiPrompts = antiPrompts;
            this.numTopMatches = numTopMatches;
            this.databaseManager = databaseManager;

            systemMessages = new string[] {
                "Reply in a natural manner and utilize your existing knowledge. If you don't know the answer, use one of the relevant DB facts in the prompt. Be a friendly, concise, never offensive chatbot to help users learn more about the University of Denver. Query: {query}\n"
            };
            prompt_number_chosen = 0;
            query = "";
            prompt = "";
            conversation = "";
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
            executor = new InteractiveExecutor(context);
            session = new ChatSession(executor);
            syntheticDataGenerator = new SyntheticDataGenerator(databaseManager, session, inputHandler, temperature, antiPrompts);
        }

        public async Task StartChatAsync(string promptInstructions, string prompt)
        {
            if (session != null)
            {
                OnMessage?.Invoke("Chat session started, please input query:\n");
                while (true)
                {
                    query = await inputHandler.ReadLineAsync();
                    
                    if (databaseManager != null)
                    {
                        prompt = await databaseManager.QueryDatabase(query, numTopMatches);
                    }
                    else
                    {
                        prompt = await GetPromptWithoutDatabase(query);
                    }

                    string response = await InteractWithModelAsync(promptInstructions, prompt, temperature, antiPrompts);
                    OnMessage?.Invoke(response);
                    conversation += prompt + " " + response;
                }
            }
        }

        private async Task<string> InteractWithModelAsync(string promptInstructions, string prompt, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n";
            prompt = $"{promptInstructions} {prompt}";
            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response.Trim();
        }

        private Task<string> GetPromptWithoutDatabase(string userQuery)
        {
            // Adjusted to take the user's query directly
            string queriedPrompt = $"User: {userQuery}\nAnswer:";
            return Task.FromResult(queriedPrompt);
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

    public class ConversationManagerConsole : ConversationManager
    {
        public ConversationManagerConsole(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, DatabaseManager? databaseManager, float temperature, string[] antiPrompts, int numTopMatches) : base(inputHandler, modelLoaderOutputs, databaseManager, temperature, antiPrompts, numTopMatches)
        {
            OnMessage += Console.Write;
        }
    }
}
