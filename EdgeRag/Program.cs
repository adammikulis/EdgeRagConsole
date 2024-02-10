namespace EdgeRag
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string directoryPath = @"C:\ai\models";
            string[] facts = new string[] {
            "The University of Denver is a private University that is abbreviated as 'DU'",
            "DU founded in 1864 as the Colorado Seminary",
            "DU is a private R1 University",
            "The mascot of the University of Denver is the Pioneer",
            "DU is located in south Denver, Colorado in the University neighborhood",
            "DU's has a secondary/satellite campus, the 720 acre Kennedy Mountain Campus which is located 110 miles northwest of Denver",
            "DU has 5700 undergraduate students and 7200 graduate students",
            "DU's Ritchie Center is home to the Magness Arena",
            "DU's hockey team plays in Magness Arena, named after cable television pioneer Bob Magness",
            "The Pioneers won the ice hockey NCAA National Championship in 2022",
            "DU's library is known as the Anderson Academic Commons"
        };
            uint contextSize = 4096;
            IInputHandler inputHandler = new ConsoleInputHandler();
            bool useDatabase = false;
            ModelLoaderConsole modelLoader = new ModelLoaderConsole(directoryPath, facts, contextSize, useDatabase);
            await modelLoader.InitializeAsync(inputHandler);
        }
    }
}