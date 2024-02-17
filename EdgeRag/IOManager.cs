namespace EdgeRag
{
    public class IOManager
    {
        private const int maxStars = 75; // Used for similarity match
        private const int headingTotalWidth = 100;

        public static event Action<string> OnOutputMessage;

        private IOManager()
        {
        }

        public static void ClearConsole()
        {
            Console.Clear();
        }

        public static void SendMessage(string message)
        {
            OnOutputMessage?.Invoke(message);
        }

        public static void SendMessageLine(string message)
        {
            OnOutputMessage?.Invoke(message);
            SendMessage("\n");
        }

        public static void AwaitKeypress()
        {
            Console.ReadKey();
        }

        public static void AwaitKeyPressAndClear()
        {
            AwaitKeypress();
            ClearConsole();
        }

        public static void PrintIntroMessage()
        {
            ClearConsole();
            PrintHeading("EdgeRag - A Local Tech Support Chatbot");
            SendMessage("\nWelcome to EdgeRag! This is a Retrieval-Augmented Generative (RAG) A.I. pipeline " +
                        "that lets you run a local chatbot\nand refer to existing solutions/documentation. " +
                        "Everything is run on your device, creating a secure chat environment for sensitive data.\n\n" +
                        "Generate synthetic data to easily populate a database and then search. You can even use " +
                        "a higher quality model\nto generate the tickets and then a faster, smaller model to serve as the chatbot.\n\n" +
                        "Refer any questions to Adam Mikulis, and have fun!\n\nPress any key to continue...");
            AwaitKeyPressAndClear();
        }

        // This prints a formatted header at the top of the console with the chosen heading
        public static void PrintHeading(string heading)
        {
            int headingLength = heading.Length;
            int starsWidth = (headingTotalWidth - headingLength) / 2;
            string starsSide = new string('*', starsWidth);

            string fullHeader = $"{starsSide} {heading} {starsSide}";

            SendMessageLine(new string('*', headingTotalWidth + 2));
            SendMessageLine(fullHeader);
            SendMessageLine(new string('*', headingTotalWidth + 2));
        }

        public static void ClearAndPrintHeading(string heading)
        {
            ClearConsole();
            PrintHeading(heading);
        }


        public static void PrintCudaInitialization()
        {
            ClearAndPrintHeading("CUDA Setup -- GPU Acceleration");
            SendMessage("\nCUDA 12.1 is installed, GPU inference enabled! Set the number of layers loaded to GPU based on your VRAM\n\n" +
                "Set GpuLayerCount to -1 to move the entire model to VRAM, or 0 for cpu-only.\n\n" +
                "If you get an error when loading the model, reduce the number of layers.\n\n" +
                "How many layers to GPU? (range: -1 to 33): ");
        }

        public static void PrintCudaError()
        {
            SendMessageLine("CUDA 12.1 is not installed. Use ReleaseCPU version if you don't have an Nvidia GPU or download here: https://developer.nvidia.com/cuda-12-1-0-download-archive\nHit any key to exit...");
            AwaitKeypress();
            Environment.Exit(0);
        }

        public static void DisplayGraphicalScores(long[] incidentNumbers, double[] scores)
        {
            SendMessageLine("Most similar tickets:");
            for (int i = 0; i < incidentNumbers.Length && i < 3; i++)
            {
                long incidentNumber = incidentNumbers[i];
                double score = scores[i];
                int starsCount = (int)Math.Round(score * maxStars);
                string stars = new string('*', starsCount).PadRight(maxStars, '-');

                SendMessageLine($"Incident {incidentNumber}: Similarity: {score:F2} [{stars}]");
            }
        }

        public static string ReadLine()
        {
            return Console.ReadLine();
        }
    }
}

