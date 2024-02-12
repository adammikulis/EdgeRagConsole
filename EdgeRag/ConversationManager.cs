using LLama;
using LLama.Common;


namespace EdgeRag
{
    public class ConversationManager
    {
        protected bool useDatabaseForChat;
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
        protected int maxTokens;

        protected LLamaWeights? model;
        protected ModelParams? modelParams;
        protected LLamaEmbedder? embedder;
        protected LLamaContext? context;
        protected InteractiveExecutor? executor;
        protected ChatSession? session;
        public DatabaseManager? databaseManager; 

        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationManager(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, DatabaseManager? databaseManager, bool useDatabaseForChat, int maxTokens, float temperature, string[] antiPrompts, int numTopMatches)
        {
            this.inputHandler = inputHandler;
            this.model = modelLoaderOutputs.model;
            this.modelType = modelLoaderOutputs.modelType;
            this.modelParams = modelLoaderOutputs.modelParams;
            this.embedder = modelLoaderOutputs.embedder;
            this.context = modelLoaderOutputs.context;
            this.maxTokens = maxTokens;
            this.temperature = temperature;
            this.antiPrompts = antiPrompts;
            this.numTopMatches = numTopMatches;
            this.databaseManager = databaseManager;
            this.useDatabaseForChat = useDatabaseForChat;

            prompt_number_chosen = 0;
            query = "";
            prompt = "";
            conversation = "";
            InitializeConversation();
        }

        public ChatSession? GetSession()
        {
            return this.session;
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
        }

        public async Task StartChatAsync(string systemMessage, string prompt)
        {
            if (session != null)
            {
                OnMessage?.Invoke("Chat session started, please input query:\n");
                while (true)
                {
                    string userQuery = await inputHandler.ReadLineAsync();

                    // Check if the user input is empty or contains "exit"
                    if (string.IsNullOrWhiteSpace(userQuery) || userQuery.ToLower() == "exit")
                    {
                        OnMessage?.Invoke("Exiting chat session.");
                        break;
                    }

                    userQuery = systemMessages[0] + userQuery;

                    if (useDatabaseForChat)
                    {
                        userQuery += " use the following data to help solve the problem:";
                        prompt = await databaseManager.QueryDatabase(userQuery, numTopMatches);
                    }
                    else
                    {
                        prompt = await GetPromptWithoutDatabase(userQuery);
                    }

                    string response = await InteractWithModelAsync(systemMessage, prompt, maxTokens, temperature, antiPrompts);
                    OnMessage?.Invoke(response);
                    conversation += prompt + " " + response;
                }
            }
        }

        public string CleanUpString(string input)
        {
            string cleanedString = input.Replace(antiPrompts[0], "")
                .Replace("Narrator:", "")
                .Replace("AI:", "")
                .Replace("\n", " ")
                .Replace("\r", " ")
                .Replace("     ", " ")
                .Replace("    ", " ")
                .Replace("   ", " ")
                .Replace("  ", " ")
                .Trim();

            return cleanedString;
        }

        private async Task<string> InteractWithModelAsync(string promptInstructions, string prompt, int maxTokens, float temperature, string[] antiPrompts)
        {
            string response = "";
            if (session == null) return "Session still initializing, please wait.\n";
            if (prompt == "" || prompt == null)
            {
                prompt = $"{promptInstructions}";
            }
            else
            {
                prompt = $"{promptInstructions} {prompt}";
            }

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response;
        }

        private Task<string> GetPromptWithoutDatabase(string userQuery)
        {
            string queriedPrompt = $"User: {userQuery}\nResponse:";
            return Task.FromResult(queriedPrompt);
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
        public ConversationManagerConsole(IInputHandler inputHandler, ModelLoaderOutputs modelLoaderOutputs, DatabaseManager? databaseManager, bool useDatabaseForChat, int maxTokens, float temperature, string[] antiPrompts, int numTopMatches) : base(inputHandler, modelLoaderOutputs, databaseManager, useDatabaseForChat, maxTokens, temperature, antiPrompts, numTopMatches)
        {
            OnMessage += Console.Write;
        }
    }
}
