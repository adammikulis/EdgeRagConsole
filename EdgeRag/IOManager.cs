using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static LLama.Common.ChatHistory;

namespace EdgeRag
{
    public class IOManager
    {

        int maxStars;
        public event Action<string> OnOutputMessage;

        public IOManager(int maxStars)
        {
            this.maxStars = maxStars;
        }

        public static async Task<IOManager> CreateAsync(int maxStars)
        {
            var iOManager = new IOManager(maxStars);
            await iOManager.InitializeAsync();
            return iOManager;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                SendMessage("IO Manager initialized!");
            });
        }

        public void SendMessage(string message)
        {
            OnOutputMessage?.Invoke(message);
        }

        public void DisplayGraphicalScores(long[] incidentNumbers, double[] scores)
        {
            SendMessage($"Most similar tickets:\n");
            for (int i = 0; i < incidentNumbers.Length && i < 3; i++)
            {
                long incidentNumber = incidentNumbers[i];
                double score = scores[i];
                int starsCount = (int)Math.Round(score * maxStars);
                string stars = new string('*', starsCount).PadRight(maxStars, '-');

                SendMessage($"Incident {incidentNumber}: [{stars}] {score:F2}\n");
            }
        }

        public async Task<string> ReadLineAsync()
        {
            return await Task.Run(() => Console.ReadLine());
        }

        public async Task RunMenuAsync(Func<Task> chat, Func<Task> chatUsingDatabase, Func<int, Task> generateQuestionsAndChat, Func<int, Task> generateQuestions, Action quit)
        {
            while (true)
            {
                SendMessage("\nMenu:");
                SendMessage("\n1. Chat");
                SendMessage("\n2. Chat using Database");
                SendMessage("\n3. Generate Questions and Chat using Database");
                SendMessage("\n4. Generate Questions and Quit");
                SendMessage("\n5. Quit");
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
                        SendMessage("Enter the number of questions to generate: ");
                        int numQuestions = Convert.ToInt32(await ReadLineAsync());
                        await generateQuestionsAndChat(numQuestions);
                        break;
                    case "4":
                        SendMessage("Enter the number of questions to generate: ");
                        numQuestions = Convert.ToInt32(await ReadLineAsync());
                        await generateQuestions(numQuestions);
                        break;
                    case "5":
                        quit();
                        return;
                    default:
                        SendMessage("Invalid option, please try again.");
                        break;
                }
            }
        }
    }
}
