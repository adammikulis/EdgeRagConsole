using LLama;
using LLama.Abstractions;
using LLama.Common;
using System.Data;
using System.IO;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class ModelManager
    {
        IOManager iOManager;
        private string directoryPath;
        private uint contextSize;
        private int numGpuLayers;
        private uint numCpuThreads;
        private uint seed;
        public string SelectedModelPath;
        public string? modelName;
        public string? modelType;
        public ModelParams? modelParams;
        public LLamaWeights? model;
        public LLamaEmbedder? embedder;
        public LLamaContext? context;

        public ModelManager(IOManager iOManager, string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            this.iOManager = iOManager;
            this.directoryPath = modelDirectoryPath;
            this.contextSize = contextSize;
            this.numGpuLayers = numGpuLayers;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
            SelectedModelPath = "";
        }

        public static async Task<ModelManager> CreateAsync(IOManager iOManager, string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            var modelManager = new ModelManager(iOManager, modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            await modelManager.InitializeAsync();
            return modelManager;
        }

        public async Task InitializeAsync()
        {
            if (!Directory.Exists(directoryPath))
            {
                iOManager.SendMessage("The directory does not exist.");
                Environment.Exit(0);
            }

            var filePaths = Directory.GetFiles(directoryPath);
            if (filePaths.Length == 0)
            {
                iOManager.SendMessage("No models found in the directory");
                Environment.Exit(0);
            }

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                for (int i = 0; i < filePaths.Length; i++)
                {
                    iOManager.SendMessage($"{i + 1}: {Path.GetFileName(filePaths[i])}");
                }

                iOManager.SendMessage("\nEnter the number of the model you want to load: ");
                if (int.TryParse(await iOManager.ReadLineAsync(), out int index) && index >= 1 && index <= filePaths.Length)
                {
                    index -= 1;
                    SelectedModelPath = filePaths[index];
                    modelName = Path.GetFileNameWithoutExtension(SelectedModelPath);
                    iOManager.SendMessage($"Model selected: {modelName}");
                    validModelSelected = true;

                    modelType = modelName.Split('-')[0].ToLower();
                }
                else
                {
                    iOManager.SendMessage("Invalid input, please enter a number corresponding to the model list.\n");
                }
            }

            modelParams = new ModelParams(SelectedModelPath)
            {
                Seed = seed,
                ContextSize = contextSize,
                EmbeddingMode = true, // Needs to be true to retrieve embeddings
                GpuLayerCount = numGpuLayers,
                Threads = numCpuThreads
            };

            model = LLamaWeights.LoadFromFile(modelParams);
            embedder = new LLamaEmbedder(model, modelParams);
            context = model.CreateContext(modelParams);
            iOManager.SendMessage($"\nModel: {modelName} from {SelectedModelPath} loaded\n");
        }
    }
}


