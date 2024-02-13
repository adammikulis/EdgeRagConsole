using LLama;
using LLama.Common;


namespace EdgeRag
{
    public class ConversationManager
    {
        private bool useDatabaseForChat;
        private string[] systemMessages;
        private string[] antiPrompts;

        private float temperature;
        private string prompt;
        private int numTopMatches;
        private int maxTokens;
        private int systemMessage;

        private LLamaWeights? model;
        private ModelParams? modelParams;
        private LLamaEmbedder? embedder;
        private LLamaContext? context;
        private InteractiveExecutor? executor;
        private ChatSession? session;
        public DatabaseManager? databaseManager; 

        private IInputHandler inputHandler;
        public event Action<string> OnMessage = delegate { };

        public ConversationManager(IInputHandler inputHandler, ModelManagerOutputs modelLoaderOutputs, DatabaseManager? databaseManager, bool useDatabaseForChat, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts, int numTopMatches)
        {
            this.inputHandler = inputHandler;
            this.model = modelLoaderOutputs.model;
            this.modelParams = modelLoaderOutputs.modelParams;
            this.embedder = modelLoaderOutputs.embedder;
            this.context = modelLoaderOutputs.context;
            this.maxTokens = maxTokens;
            this.temperature = temperature;
            this.antiPrompts = antiPrompts;
            this.systemMessages = systemMessages;
            this.numTopMatches = numTopMatches;
            this.databaseManager = databaseManager;
            this.useDatabaseForChat = useDatabaseForChat;
            systemMessage = 0;
            prompt = "";
            InitializeConversation();
        }

        public ChatSession? GetSession()
        {
            return this.session;
        }

        public int GetMaxTokens()
        {
            return this.maxTokens;
        }

        public float GetTemperature()
        {
            return this.temperature;
        }

        public string[] GetAntiPrompts()
        {
            return this.antiPrompts;
        }

        public string[] GetSystemMessages()
        {
            return this.systemMessages;
        }

        private void InitializeConversation()
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

        public async Task StartChatAsync()
        {
            if (session != null)
            {
                OnMessage?.Invoke("Chat session started, please input query:\n");
                while (true)
                {
                    string prompt = await inputHandler.ReadLineAsync();

                    // Check if the user input is empty or contains "exit"
                    if (string.IsNullOrWhiteSpace(prompt) || prompt.ToLower() == "exit")
                    {
                        OnMessage?.Invoke("Exiting chat session.");
                        break;
                    }

                    if (useDatabaseForChat)
                    {
                        // Query the database and get results
                        var queryResults = await databaseManager.QueryDatabase(prompt, numTopMatches);
                        string summarizedTextResponse = await InteractWithModelAsync($"Summarize the troubleshooting steps: {queryResults.summarizedText}", maxTokens);

                        // Join the incident numbers and scores arrays into strings for display
                        string incidentNumbersStr = string.Join(", ", queryResults.incidentNumbers);
                        string scoresStr = string.Join(", ", queryResults.scores.Select(score => score.ToString("F2"))); // Formatting scores to 2 decimal places

                        // Display the results
                        OnMessage?.Invoke($"\nIncident numbers: [{incidentNumbersStr}]\t Scores: [{scoresStr}]\n");
                        OnMessage?.Invoke(summarizedTextResponse + "\n"); // Display the model's response correctly after awaiting its result
                    }
                    else
                    {
                        // If not using the database for chat, directly interact with the model using the user input
                        string response = await InteractWithModelAsync(prompt, maxTokens);
                        OnMessage?.Invoke(response + "\n");
                    }
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

        public async Task<string> InteractWithModelAsync(string prompt, int maxTokens)
        {
            if (session == null) return "Session still initializing, please wait.\n";
            string response = "";

            // Assuming systemMessages[systemMessage] is a prefix you want to add to every prompt
            string fullPrompt = $"{systemMessages[systemMessage]} {prompt}".Trim();

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, fullPrompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                response += text;
            }
            return response;
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
        public ConversationManagerConsole(IInputHandler inputHandler, ModelManagerOutputs modelLoaderOutputs, DatabaseManager? databaseManager, bool useDatabaseForChat, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts, int numTopMatches) : base(inputHandler, modelLoaderOutputs, databaseManager, useDatabaseForChat, maxTokens, temperature, systemMessages, antiPrompts, numTopMatches)
        {
            OnMessage += Console.Write;
        }
    }
}
