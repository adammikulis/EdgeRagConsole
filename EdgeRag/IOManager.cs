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
            SendMessage("IO Manager initialized!");
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
    }

}
