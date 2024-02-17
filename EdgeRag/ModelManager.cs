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
        private string modelDirectoryPath;
        private const string defaultModelUrl = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf";
        private uint contextSize;
        private int numGpuLayers;
        private uint numCpuThreads;
        private uint seed;
        public string selectedModelPath;
        public string? selectedModelName;
        public string selectedModelType;
        public ModelParams? modelParams;
        public LLamaWeights? model;
        public LLamaEmbedder? embedder;
        public LLamaContext? context;

        public ModelManager(string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            this.modelDirectoryPath = modelDirectoryPath;
            this.contextSize = contextSize;
            this.numGpuLayers = numGpuLayers;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
        }

        public static async Task<ModelManager> CreateAsync(string modelDirectoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads)
        {
            var modelManager = new ModelManager(modelDirectoryPath, seed, contextSize, numGpuLayers, numCpuThreads);
            await modelManager.InitializeAsync();
            return modelManager;
        }

        public async Task InitializeAsync()
        {
            CheckDirectoryExists();
            string[] filePaths = await CheckModelsExist();

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                validModelSelected = await DisplayAndLoadModels(filePaths, validModelSelected);
            }
            CreateModelParams();
            LoadModelEmbedderContext();
        }

        private void LoadModelEmbedderContext()
        {
            model = LLamaWeights.LoadFromFile(modelParams);
            embedder = new LLamaEmbedder(model, modelParams);
            context = model.CreateContext(modelParams);
            IOManager.SendMessage($"Model: {selectedModelName} from {modelDirectoryPath} loaded\n");
        }

        private void CreateModelParams()
        {
            modelParams = new ModelParams(selectedModelPath)
            {
                Seed = seed,
                ContextSize = contextSize,
                EmbeddingMode = true, // Needs to be true to retrieve embeddings
                GpuLayerCount = numGpuLayers,
                Threads = numCpuThreads
            };
        }

        private async Task<bool> DisplayAndLoadModels(string[] filePaths, bool validModelSelected)
        {
            IOManager.SendMessage($"\nCurrent model directory: {modelDirectoryPath}\nEnter the number of the model you want to load:\n");
            for (int i = 0; i < filePaths.Length; i++)
            {
                IOManager.SendMessage($"{i + 1}: {Path.GetFileName(filePaths[i])}\n");
            }

            if (int.TryParse(await IOManager.ReadLineAsync(), out int index) && index >= 1 && index <= filePaths.Length)
            {
                index -= 1;
                selectedModelPath = filePaths[index];
                selectedModelName = Path.GetFileNameWithoutExtension(selectedModelPath);
                selectedModelType = selectedModelName.Split('-')[0].ToLower();
                IOManager.SendMessage($"\nModel selected: {selectedModelName}\n");
                validModelSelected = true;

                // Determine the context size based on the model type
                DetermineMaxContextSize();
            }
            else
            {
                IOManager.SendMessage("\nInvalid input, please enter a number corresponding to the model list.\n");
            }

            return validModelSelected;
        }

        private void DetermineMaxContextSize()
        {
            switch (selectedModelType)
            {
                case "phi":
                    contextSize = 2048;
                    break;
                case "llama":
                case "mistral":
                    contextSize = 4096;
                    break;
                case "mixtral":
                    contextSize = 32768;
                    break;
                case "codellama":
                    contextSize = 65536;
                    break;
                default:
                    contextSize = 4096;
                    break;
            }
            IOManager.SendMessage($"{selectedModelType} detected, context size set to {contextSize}\n");
        }

        private async Task<string[]> CheckModelsExist()
        {
            var filePaths = Directory.GetFiles(modelDirectoryPath);
            if (filePaths.Length == 0)
            {
                filePaths = await DownloadDefaultModel(filePaths);
            }

            return filePaths;
        }

        private async Task<string[]> DownloadDefaultModel(string[] filePaths)
        {
            IOManager.SendMessage($"\nNo models found in the {modelDirectoryPath}. Attempting to download default model...\n");

            // Specify the default model to download
            
            // Specify the destination folder, which is the current directoryPath
            await DownloadManager.DownloadModelAsync(defaultModelUrl, modelDirectoryPath);

            // Recheck the directory for models after attempting the download
            filePaths = Directory.GetFiles(modelDirectoryPath);
            if (filePaths.Length == 0)
            {
                IOManager.SendMessage("\nFailed to download the default model. Please verify your internet connection and try again.\n");
                Environment.Exit(0);
            }

            return filePaths;
        }

        private void CheckDirectoryExists()
        {
            if (!Directory.Exists(modelDirectoryPath))
            {
                IOManager.SendMessage("The directory does not exist. Creating directory...\n");
                Directory.CreateDirectory(modelDirectoryPath);
            }
        }
    }
}


