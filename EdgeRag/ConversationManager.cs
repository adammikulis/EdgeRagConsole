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
        private int numTopMatches;
        

        public InteractiveExecutor? executor;
        public ChatSession? session;

        public event Action<string> OnMessage = delegate { };

        public ConversationManager(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts, int numTopMatches)
        {
            this.databaseManager = databaseManager;
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            this.modelManager = modelManager;
            this.maxTokens = maxTokens;
            this.antiPrompts = antiPrompts;
            this.systemMessages = systemMessages;
            this.databaseManager = databaseManager;
            this.selectedModelType = modelManager.selectedModelType;
            this.numTopMatches = numTopMatches;
            systemMessageNumber = 0;

        }

        public static async Task<ConversationManager> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts, int numTopMatches)
        {
            var conversationManager = new ConversationManager(modelManager, databaseManager, maxTokens, systemMessages, antiPrompts, numTopMatches);
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

            IOManager.SendMessage("Chat session started, please input your query:\n");
            while (true)
            {
                string userInput = IOManager.ReadLine();

                if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "exit" || userInput.ToLower() == "back")
                {
                    IOManager.SendMessage("Exiting chat session.");
                    break;
                }

                if (useDatabaseForChat)
                {
                    var withDatabaseResponse = await QueryDatabase(userInput);
                    IOManager.DisplayGraphicalScores(withDatabaseResponse.incidentNumbers, withDatabaseResponse.scores);
                    string response = await InteractWithModelAsync(withDatabaseResponse.summarizedText, maxTokens, averageTemperature, false);
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
        public async Task<(string summarizedText, long[] incidentNumbers, double[] scores)> QueryDatabase(string prompt)
        {
            if (vectorDatabase.Rows.Count == 0)
            {
                return (prompt, new long[0], new double[0]);
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
                return (prompt, new long[0], new double[0]);
            }

            int numSummarizedIncidents = 1; // Set the number of incidents to summarize
            var topMatches = scoresIncidents.OrderByDescending(s => s.Item1).Take(numTopMatches).ToList();
            long[] incidentNumbers = topMatches.Select(m => m.Item2).ToArray();
            double[] scores = topMatches.Select(m => m.Item1).ToArray();

            string summaryRequest = $"Choose the most relevant solution for {prompt}: ";

            for (int i = 0; i < numSummarizedIncidents && i < topMatches.Count; i++)
            {
                var (_, _, originalText) = topMatches[i];
                summaryRequest += $"{originalText} ";
            }

            string summary = await InteractWithModelAsync(summaryRequest, maxTokens / 16, 0.5f, true); // Internal dialog
            summary = CleanUpString(summary);
            summary = summary.Replace(summaryRequest, "");
            summary = summary.Replace(prompt, "");

            return (summary, incidentNumbers, scores);
        }
    }
}
