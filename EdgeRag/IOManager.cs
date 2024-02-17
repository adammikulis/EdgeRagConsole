using System;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class IOManager
    {
        private const int maxStars = 50;

        public static event Action<string> OnOutputMessage;

        // Private constructor to prevent instantiation from outside the class
        private IOManager()
        {
        }

        public static void SendMessage(string message)
        {
            OnOutputMessage?.Invoke(message);
        }

        public static void DisplayGraphicalScores(long[] incidentNumbers, double[] scores)
        {
            SendMessage("Most similar tickets:\n");
            for (int i = 0; i < incidentNumbers.Length && i < 3; i++)
            {
                long incidentNumber = incidentNumbers[i];
                double score = scores[i];
                int starsCount = (int)Math.Round(score * maxStars);
                string stars = new string('*', starsCount).PadRight(maxStars, '-');

                SendMessage($"Incident {incidentNumber}: [{stars}] {score:F2}\n");
            }
        }

        public static async Task<string> ReadLineAsync()
        {
            return await Task.Run(() => Console.ReadLine());
        }

        public static async Task RunMenuAsync(Func<Task> chat, Func<Task> chatUsingDatabase, Func<int, Task> generateQuestionsAndChat, Func<int, Task> generateQuestions, Func<Task> downloadModel, Func<Task> loadDifferentModel, Action quit)
        {
            while (true)
            {
                SendMessage("\nMenu:");
                SendMessage("\n1. Chat");
                SendMessage("\n2. Chat using Database");
                SendMessage("\n3. Generate Questions and Chat using Database");
                SendMessage("\n4. Generate Questions and Quit");
                SendMessage("\n5. Download Model");
                SendMessage("\n6. Load Different Model");
                SendMessage("\n7. Quit");
                SendMessage("\nSelect an option: ");

                var option = await ReadLineAsync();
                switch (option)
                {
                    case "1":
                        await chat();
                        break;
                    case "2":
                        await chatUsingDatabase();
                        break;
                    case "3":
                        SendMessage("\nEnter the number of questions to generate: ");
                        int numQuestions = Convert.ToInt32(await ReadLineAsync());
                        await generateQuestionsAndChat(numQuestions);
                        break;
                    case "4":
                        SendMessage("\nEnter the number of questions to generate: ");
                        numQuestions = Convert.ToInt32(await ReadLineAsync());
                        await generateQuestions(numQuestions);
                        break;
                    case "5":
                        await downloadModel();
                        break;
                    case "6":
                        await loadDifferentModel();
                        break;
                    case "7":
                        quit();
                        return;
                    default:
                        SendMessage("\nInvalid option, please try again.\n");
                        break;
                }
            }
        }
    }
}
