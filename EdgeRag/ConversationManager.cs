using LLama;
using LLama.Common;

namespace EdgeRag
{
    public class ConversationManager
    {
        private IOManager iOManager;
        private string[] systemMessages;
        private string[] antiPrompts;

        private float temperature;
        private string prompt;
        private int maxTokens;
        private int systemMessage;

        private LLamaWeights? model;
        private ModelParams? modelParams;
        public LLamaContext? context;
        public InteractiveExecutor? executor;
        public ChatSession? session;
        public DatabaseManager? databaseManager; 

        public event Action<string> OnMessage = delegate { };

        public ConversationManager(IOManager iOManager, ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts)
        {
            this.iOManager = iOManager;
            this.model = modelManager.model;
            this.modelParams = modelManager.modelParams;
            this.context = modelManager.context;
            this.maxTokens = maxTokens;
            this.temperature = temperature;
            this.antiPrompts = antiPrompts;
            this.systemMessages = systemMessages;
            this.databaseManager = databaseManager;
            systemMessage = 0;
            prompt = "";
        }

        public static async Task<ConversationManager> CreateAsync(IOManager iOManager, ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts)
        {
            var conversationManager = new ConversationManager(iOManager, modelManager, databaseManager, maxTokens, temperature, systemMessages, antiPrompts);
            await conversationManager.InitializeAsync();
            return conversationManager;
        }

        
        private async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                if (model == null || modelParams == null)
                {
                    OnMessage?.Invoke("Model or modelParams is null. Cannot initialize conversation.");
                    return;
                }

                context = model.CreateContext(modelParams);
                executor = new InteractiveExecutor(context);
                session = new ChatSession(executor);
            });
             
        }

        public async Task StartChatAsync(bool useDatabaseForChat)
        {
            if (session == null) return;

            iOManager.SendMessage("Chat session started, please input your query:\n");
            while (true)
            {
                string userInput = await iOManager.ReadLineAsync();

                if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "exit" || userInput.ToLower() == "back")
                {
                    iOManager.SendMessage("Exiting chat session.");
                    break;
                }

                if (useDatabaseForChat)
                {
                    var withDatabaseResponse = await databaseManager.QueryDatabase(userInput);
                    iOManager.DisplayGraphicalScores(withDatabaseResponse.incidentNumbers, withDatabaseResponse.scores);
                    string response = await InteractWithModelAsync(withDatabaseResponse.summarizedText, maxTokens);
                    iOManager.SendMessage(response + "\n");
                }
                else
                {
                    string response = await InteractWithModelAsync(userInput, maxTokens);
                    iOManager.SendMessage(response + "\n");
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



    }
}
