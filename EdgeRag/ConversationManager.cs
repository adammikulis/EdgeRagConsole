using System.Data;
using LLama;
using LLama.Common;

namespace EdgeRag
{
    public class ConversationManager
    {
        private ModelManager modelManager;
        private DatabaseManager databaseManager;
        private DataTable vectorDatabase;
        private string[] systemMessages;
        private string selectedModelType;
        private int systemMessageNumber;
        private string[] antiPrompts;
        private const float averageTemperature = 0.5f;
        public int maxTokens;
        private const int numTopMatches = 5;
        

        public InteractiveExecutor? executor;
        public ChatSession? session;

        public event Action<string> OnMessage = delegate { };

        public ConversationManager(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts)
        {
            this.databaseManager = databaseManager;
            this.vectorDatabase = databaseManager.GetVectorDatabase();
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
                    IOManager.SendMessage($"{selectedModelType}-type model detected, max tokens set to {maxTokens}\n");
                }

                executor = new InteractiveExecutor(modelManager.context);
                session = new ChatSession(executor);
            });
        }

        public async Task StartChatAsync(bool useDatabaseForChat)
        {
            if (useDatabaseForChat)
            {
                IOManager.ClearAndPrintHeading("Chatbot - Using Database");
            }
            else
            {
                IOManager.ClearAndPrintHeading("Chatbot - No Database");
            }
            
                
            if (session == null) return;

            IOManager.SendMessageLine("\nChat session started, please input your query (back to go back and quit to quit):");
            while (true)
            {
                string userInput = IOManager.ReadLine();

                if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "back")
                {
                    IOManager.SendMessage("Exiting chat session.");
                    break;
                }

                if (userInput.ToLower() == "quit")
                {
                    modelManager.Dispose();
                    System.Environment.Exit(0);
                }


                if (useDatabaseForChat)
                {
                    var summarizedResult = await QueryDatabase(userInput, 5, 3);
                    string response = await InteractWithModelAsync($"Solve {userInput} with {summarizedResult}", maxTokens, averageTemperature, false);
                    IOManager.SendMessage(response + "\n");
                }
                else
                {
                    string response = await InteractWithModelAsync(userInput, maxTokens, averageTemperature, false);
                    IOManager.SendMessage(response + "\n");
                }
            }
        }
        public string CleanUpString(string input)
        {
            string cleanedString = input.Replace(antiPrompts[0], "")
                .Replace("Narrator:", "")
                .Replace("AI:", "")
                .Replace("User:", "")
                .Replace("Support:", "")
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
            prompt = $"{systemMessages[systemMessageNumber]}{prompt}".Trim();

            await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), new InferenceParams { MaxTokens = maxTokens, Temperature = temperature, AntiPrompts = antiPrompts }))
            {
                // This allows control over whether the message is streamed or not, set to true for "internal" dialog that doesn't print to console
                if (!internalChat)
                {
                    IOManager.SendMessage(text);
                }
                prompt += text;
            }
            return prompt;
        }

        // This is the method to turn a prompt into embeddings, match to the database, and then return the original solution
        public async Task<string> QueryDatabase(string prompt, int topMatchesDisplayed, int topMatchesSummarized)
        {
            if (vectorDatabase.Rows.Count == 0)
            {
                return (prompt);
            }

            var queryEmbeddings = await databaseManager.GenerateEmbeddingsAsync(prompt);
            List<Tuple<double, long, string>> scoresIncidents = new List<Tuple<double, long, string>>();

            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[modelManager.selectedModelType];
                double score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                long incidentNumber = Convert.ToInt64(row["incidentNumber"]);
                string originalText = row["incidentSolution"].ToString();
                scoresIncidents.Add(new Tuple<double, long, string>(score, incidentNumber, originalText));
            }

            if (scoresIncidents.Count == 0)
            {
                return (prompt);
            }
            var topMatches = scoresIncidents.OrderByDescending(s => s.Item1).Take(topMatchesDisplayed).ToList();
            long[] incidentNumbers = topMatches.Select(m => m.Item2).ToArray();
            double[] scores = topMatches.Select(m => m.Item1).ToArray();

            IOManager.DisplayGraphicalScores(incidentNumbers, scores);


            string summaryRequest = $"Solve {prompt} with: ";

            // Allows for different amount of tickets displayed vs actually summarized
            for (int i = 0; i < topMatchesSummarized && i < topMatches.Count; i++)
            {
                var (_, _, originalText) = topMatches[i];
                summaryRequest += $"{originalText} ";
            }

            string summary = await InteractWithModelAsync(summaryRequest, maxTokens / 8, 0.5f, true); // Internal dialog
            summary = CleanUpString(summary);
            summary = summary.Replace(summaryRequest, "");
            summary = summary.Replace(prompt, "");

            return (summary);
        }
    }
}
