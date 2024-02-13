using LLama;
using LLama.Common;

namespace EdgeRag
{
    public class ConversationManager
    {
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

        public ConversationManager(IInputHandler inputHandler, ModelManagerOutputs modelLoaderOutputs, DatabaseManager? databaseManager, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts, int numTopMatches)
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

        public async Task StartChatAsync(bool useDatabaseForChat)
        {
            if (session == null) return;

            OnMessage?.Invoke("Chat session started, please input your query:\n");
            while (true)
            {
                string userInput = await inputHandler.ReadLineAsync();

                if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "exit")
                {
                    OnMessage?.Invoke("Exiting chat session.");
                    break;
                }

                if (useDatabaseForChat)
                {
                    var withDatabaseResponse = await databaseManager.QueryDatabase(userInput, numTopMatches);
                    DisplayGraphicalScores(withDatabaseResponse.incidentNumbers, withDatabaseResponse.scores);
                    string response = await InteractWithModelAsync(withDatabaseResponse.summarizedText, maxTokens);
                    OnMessage?.Invoke(response + "\n");
                }
                else
                {
                    string response = await InteractWithModelAsync(userInput, maxTokens);
                    OnMessage?.Invoke(response + "\n");
                }
            }
        }

        private void DisplayGraphicalScores(long[] incidentNumbers, double[] scores)
        {
            int maxStars = 50;
            OnMessage?.Invoke($"Most similar tickets:\n");
            for (int i = 0; i < incidentNumbers.Length && i < 3; i++)
            {
                long incidentNumber = incidentNumbers[i];
                double score = scores[i];
                int starsCount = (int)Math.Round(score * maxStars);
                string stars = new string('*', starsCount).PadRight(maxStars, '-');

                OnMessage?.Invoke($"Incident {incidentNumber}: [{stars}] {score:F2}\n");
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
        public ConversationManagerConsole(IInputHandler inputHandler, ModelManagerOutputs modelLoaderOutputs, DatabaseManager? databaseManager, int maxTokens, float temperature, string[] systemMessages, string[] antiPrompts, int numTopMatches) : base(inputHandler, modelLoaderOutputs, databaseManager, maxTokens, temperature, systemMessages, antiPrompts, numTopMatches)
        {
            OnMessage += Console.Write;
        }
    }
}
