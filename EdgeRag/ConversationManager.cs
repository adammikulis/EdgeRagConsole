// This class loads and manages the conversation aspect, which includes the InteractiveExecutor and ChatSession
// Database querying happens here

using System.Data;
using System.Text;
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
        
        // These are the last parts to load to make the model functional for conversation
        public InteractiveExecutor? executor;
        public ChatSession? session;

        // Platform-agnostic IO to allow to porting to Godot
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

        // Factory method
        public static async Task<ConversationManager> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, int maxTokens, string[] systemMessages, string[] antiPrompts)
        {
            var conversationManager = new ConversationManager(modelManager, databaseManager, maxTokens, systemMessages, antiPrompts);
            await conversationManager.InitializeAsync();
            return conversationManager;
        }

        // Initialization method
        private async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                if (modelManager.model == null || modelManager.modelParams == null)
                {
                    OnMessage?.Invoke("Model or modelParams is null. Cannot initialize conversation.\n");
                    return;
                }

                // Lets the user automatically pick the highest amount of tokens allowed per model (cases must be manually updated with different models)
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
                            maxTokens = 2048; // 2048 is the bare minimum for models like biollama and phi
                            break;
                    }
                    IOManager.SendMessage($"{selectedModelType}-type model detected, max tokens set to {maxTokens}\n");
                }

                executor = new InteractiveExecutor(modelManager.context); // Future updates will use the new BatchedExecutor for multiple conversations
                session = new ChatSession(executor);
            });
        }

        public async Task StartChatAsync(bool useDatabaseForChat)
        {
             
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

                // This is where the disctiontion between regular chat and database-enabled chat is made
                if (useDatabaseForChat)
                {
                    var summarizedResult = await QueryDatabase(userInput, 5, 3);
                    summarizedResult.Replace(userInput, "");
                    summarizedResult = CleanUpString(summarizedResult);
                    
                    
                    string responseNoDB = await InteractWithModelAsync($"Solve {userInput}", maxTokens / 8, averageTemperature, false);
                    responseNoDB.Replace(userInput, "");
                    responseNoDB = CleanUpString(responseNoDB);
                    string response = await InteractWithModelAsync($"Pick the best solution(s) from {summarizedResult} and {responseNoDB}", maxTokens, averageTemperature, false);
                    
                    response = CleanUpString(response);
                    IOManager.SendMessageLine("Hit a key to continue...");
                    IOManager.AwaitKeypress();
                    IOManager.ClearAndPrintHeading("Chatbot - Using Database");
                }
                
                // No database chat
                else
                {
                    string response = await InteractWithModelAsync($"Solve {userInput}", maxTokens / 8, averageTemperature, false);
                    response.Replace(userInput, "");
                    response = CleanUpString(response); // Response not yet used but available for future iterations
                    IOManager.SendMessageLine("Hit a key to continue...");
                    IOManager.AwaitKeypress();
                    IOManager.ClearAndPrintHeading("Chatbot - No Database");
                }
            }
        }
        
        // It is important to clean the data both in and out of the LLM (particularly when chaining responses)
        public string CleanUpString(string input)
        {
            string cleanedString = input.Replace(antiPrompts[0], "")
                .Replace("Narrator:", "")
                .Replace("AI:", "")
                .Replace("User:", "")
                .Replace("Support:", "")
                .Replace("\r", " ")
                .Replace("      ", " ")
                .Replace("     ", " ")
                .Replace("    ", " ")
                .Replace("   ", " ")
                .Replace("  ", " ")
                .Trim();

            return cleanedString;
        }

        // This is the most important/called method in the entire program, used for interacting with the model
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

        // This is the method to turn a new incident prompt into embeddings, match to the database, and then return the original incident response/solution
        public async Task<string> QueryDatabase(string prompt, int topMatchesDisplayed, int topMatchesSummarized)
        {
            IOManager.ClearAndPrintHeading("Chatbot - Using Database");
            if (vectorDatabase.Rows.Count == 0)
            {
                return prompt;
            }

            // This is where the user query is turned into embeddings to compare to the DB embeddings
            var queryEmbeddings = await databaseManager.GenerateEmbeddingsAsync(prompt); // Must have EmbeddingMode = true in ModelParams for this to work
            List<Tuple<double, long, string>> scoresIncidents = new List<Tuple<double, long, string>>();

            // Match the query vector embedding with the database embeddings to determine a score, storing that and the incincident as a tuple
            foreach (DataRow row in vectorDatabase.Rows)
            {
                var factEmbeddings = (double[])row[modelManager.selectedModelType];
                double score = VectorSearchUtility.CosineSimilarity(queryEmbeddings, factEmbeddings);
                long incidentNumber = Convert.ToInt64(row["incidentNumber"]);
                string originalText = row["supportResponse"].ToString(); // Could possibly use incidentSolution, but supportResponse seems to have more data
                scoresIncidents.Add(new Tuple<double, long, string>(score, incidentNumber, originalText));
            }

            // Fallback if the database is empty
            if (scoresIncidents.Count == 0)
            {
                IOManager.SendMessageLine("No matches found! Using standard model response:");
                return prompt;
            }

            // Sort and display top matches
            var topMatchesForDisplay = scoresIncidents.OrderByDescending(s => s.Item1).Take(topMatchesDisplayed).ToList();
            long[] incidentNumbersForDisplay = topMatchesForDisplay.Select(m => m.Item2).ToArray();
            double[] scoresForDisplay = topMatchesForDisplay.Select(m => m.Item1).ToArray();

            IOManager.SendMessageLine($"\nClosest ticket matches for: {prompt}\n\n");
            IOManager.DisplayGraphicalScores(incidentNumbersForDisplay, scoresForDisplay); // This is what renders the sideways bar graph of similarity scores

            // Prepare summary from top summarized matches
            var topMatchesForSummary = scoresIncidents.OrderByDescending(s => s.Item1).Take(topMatchesSummarized).ToList();
            // Initialize StringBuilder for efficient string manipulation
            StringBuilder summaryRequestBuilder = new StringBuilder($"Solve {prompt} with: ");
            foreach (var (score, incidentNumber, originalText) in topMatchesForSummary)
            {
                summaryRequestBuilder.Append($"{originalText} ");
            }
            string summaryRequest = summaryRequestBuilder.ToString();
            string summary = await InteractWithModelAsync(summaryRequest, maxTokens / 8, 0.5f, true);
            summary = CleanUpString(summary);
            summary = summary.Replace(summaryRequest, "").Replace(prompt, "");

            return summary;

        }

    }
}
