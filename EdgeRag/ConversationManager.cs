using LLama;
using LLama.Common;

namespace EdgeRag
{
    public class ConversationManager
    {
        
        private ModelManager modelManager;
        private DatabaseManager databaseManager;
        private string[] systemMessages;
        private string selectedModelType;
        private int systemMessageNumber;
        private string[] antiPrompts;

        private float temperature;
        private int maxTokens;
        

        public InteractiveExecutor? executor;
        public ChatSession? session;

        public event Action<string> OnMessage = delegate { };

        public ConversationManager(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.modelManager = modelManager;
            this.maxTokens = maxTokens;
            this.antiPrompts = antiPrompts;
            this.systemMessages = systemMessages;
            this.databaseManager = databaseManager;
            this.selectedModelType = modelManager.selectedModelType;
            systemMessageNumber = 0;

        }

        public static async Task<ConversationManager> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts)
        {
            var conversationManager = new ConversationManager(modelManager, databaseManager, maxTokens, systemMessages, antiPrompts);
            await conversationManager.InitializeAsync();
            return conversationManager;
        }

        private async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                if (modelManager.model == null || modelManager.modelParams == null)
                {
                    OnMessage?.Invoke("Model or modelParams is null. Cannot initialize conversation.\n");
                    return;
                }
                if (maxTokens == 0)
                {
                    switch (selectedModelType)
                    {
                        case "phi":
                            maxTokens = 2048;
                            break;
                        case "llama":
                        case "mistral":
                            maxTokens = 4096;
                            break;
                        case "mixtral":
                            maxTokens = 32768;
                            break;
                        case "codellama":
                            maxTokens = 65536;
                            break;
                        default:
                            maxTokens = 4096;
                            break;
                    }
                    IOManager.SendMessage($"{selectedModelType} detected, max tokens set to {maxTokens}\n");
                }

                executor = new InteractiveExecutor(modelManager.context);
                session = new ChatSession(executor);
            });
        }

        public async Task StartChatAsync(bool useDatabaseForChat)
        {
            if (session == null) return;

            IOManager.SendMessage("Chat session started, please input your query:\n");
            while (true)
            {
                string userInput = await IOManager.ReadLineAsync();

                if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "exit" || userInput.ToLower() == "back")
                {
                    IOManager.SendMessage("Exiting chat session.");
                    break;
                }

                if (useDatabaseForChat)
                {
                    var withDatabaseResponse = await databaseManager.QueryDatabase(userInput);
                    IOManager.DisplayGraphicalScores(withDatabaseResponse.incidentNumbers, withDatabaseResponse.scores);
                    string response = await InteractWithModelAsync(withDatabaseResponse.summarizedText, maxTokens, temperature, false);
                    IOManager.SendMessage(response + "\n");
                }
                else
                {
                    string response = await InteractWithModelAsync(userInput, maxTokens, temperature, false);
                    IOManager.SendMessage(response + "\n");
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

        public async Task<string> InteractWithModelAsync(string prompt, int maxTokens, float temperature, bool internalChat)
        {
            if (session == null) return "Session still initializing, please wait.\n";
            prompt = $"{systemMessages[systemMessageNumber]} {prompt}".Trim();

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                // This allows control over whether the message is streamed or not
                if (!internalChat)
                {
                    IOManager.SendMessage(text);
                }
                prompt += text;
            }
            return prompt;
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
