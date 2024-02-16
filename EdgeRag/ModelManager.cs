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

        public ModelManager(string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            this.directoryPath = modelDirectoryPath;
            this.contextSize = contextSize;
            this.numGpuLayers = numGpuLayers;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
            SelectedModelPath = "";
        }

        public static async Task<ModelManager> CreateAsync(string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            var modelManager = new ModelManager(modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            await modelManager.InitializeAsync();
            return modelManager;
        }

        public async Task InitializeAsync()
        {
            
            if (!Directory.Exists(directoryPath))
            {
                IOManager.SendMessage("The directory does not exist.");
                Environment.Exit(0);
            }

            var filePaths = Directory.GetFiles(directoryPath);
            if (filePaths.Length == 0)
            {
                IOManager.SendMessage("No models found in the directory");
                Environment.Exit(0);
            }

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                for (int i = 0; i < filePaths.Length; i++)
                {
                    IOManager.SendMessage($"{i + 1}: {Path.GetFileName(filePaths[i])}");
                }

                IOManager.SendMessage("\nEnter the number of the model you want to load: ");
                if (int.TryParse(await IOManager.ReadLineAsync(), out int index) && index >= 1 && index <= filePaths.Length)
                {
                    index -= 1;
                    SelectedModelPath = filePaths[index];
                    modelName = Path.GetFileNameWithoutExtension(SelectedModelPath);
                    IOManager.SendMessage($"Model selected: {modelName}");
                    validModelSelected = true;

                    modelType = modelName.Split('-')[0].ToLower();

                    if (contextSize == 0)
                    {
                        if (modelType == "phi")
                        {
                            contextSize = 2048;
                        }
                        else if (modelType == "llama" || modelType == "mistral")
                        {
                            contextSize = 4096;
                        }
                        else if (modelType == "mixtral")
                        {
                            contextSize = 32768;
                        }
                        else if (modelType == "codellama")
                        {
                            contextSize = 65536;
                        }
                        IOManager.SendMessage($"{modelType} detected, context size set to {contextSize}");
                    }
                }
                else
                {
                    IOManager.SendMessage("Invalid input, please enter a number corresponding to the model list.\n");
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
            IOManager.SendMessage($"\nModel: {modelName} from {SelectedModelPath}loaded\n");
        }
    }
}


