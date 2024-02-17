using LLama;
using LLama.Abstractions;
using LLama.Common;
using System.Data;
using System.IO;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class ModelManager : IDisposable
    {
        private string modelDirectoryPath;
        private uint contextSize;
        private int gpuLayerCount;
        private const int maxGpuLayers = 33;
        private uint numCpuThreads;
        private uint seed;
        public string selectedModelPath;
        public string? selectedModelName;
        public string selectedModelType;
        public ModelParams? modelParams;
        public LLamaWeights? model;
        public LLamaEmbedder? embedder;
        public LLamaContext? context;

        public ModelManager(string modelDirectoryPath, uint seed, uint contextSize, uint numCpuThreads)
        {
            this.modelDirectoryPath = modelDirectoryPath;
            this.contextSize = contextSize;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
        }

        public static async Task<ModelManager> CreateAsync(string modelDirectoryPath, uint seed, uint contextSize, uint numCpuThreads)
        {
            var modelManager = new ModelManager(modelDirectoryPath, seed, contextSize, numCpuThreads);
            await modelManager.InitializeAsync();
            return modelManager;
        }

        public async Task InitializeAsync()
        {
            CheckDirectoryExists();
            await CheckAndDownloadModelIfNeeded();
            string[] filePaths = Directory.GetFiles(modelDirectoryPath);

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                validModelSelected = await DisplayAndLoadModels(filePaths, validModelSelected);
            }

            // GPU initialization depends on which release user is running
            gpuLayerCount = 0;
            #if RELEASECUDA12
                // CUDA-specific initialization
                string windowsCudaPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1";
                string linuxCudaPath = "/usr/local/cuda-12.1";
                if (System.IO.Directory.Exists(windowsCudaPath) || System.IO.Directory.Exists(linuxCudaPath))
                {
                    IOManager.SendMessage("CUDA 12.1 is installed, GPU inference enabled\nSet GpuLayerCount (-1 is entire model to VRAM, 0 is cpu-only, layer range is 1-33): ");
                    string input = await IOManager.ReadLineAsync();
                    gpuLayerCount = int.Parse(input);
                    if (gpuLayerCount > 33)
                    {
                        gpuLayerCount = 33;
                    }
                }
                else
                {
                    IOManager.SendMessage("CUDA 12.1 is not installed. Use ReleaseCPU version if you don't have an Nvidia GPU or download here: https://developer.nvidia.com/cuda-12-1-0-download-archive\nExiting...\n");
                    Environment.Exit(0);
                }
            #endif

            #if RELEASECPU
                // CPU initialization
                IOManager.SendMessage("Running in CPU mode, no CUDA checks required.");
                    numGpuLayers = 0;
            #endif

            CreateModelParams();
            LoadModelEmbedderContext();
        }

        // Used to manually unload model
        public void Dispose()
        {
            model.Dispose();
            embedder.Dispose();
            context.Dispose();

            model = null;
            embedder = null;
            context = null;
        }

        public void UnloadModel()
        {
            Dispose();
            IOManager.SendMessage("\nModel unloaded successfully.\n");
        }

        public async Task LoadDifferentModelAsync(string modelPath)
        {
            UnloadModel();
            selectedModelPath = modelPath;
            selectedModelName = Path.GetFileNameWithoutExtension(selectedModelPath);
            selectedModelType = selectedModelName.Split('-')[0].ToLower();

            await InitializeAsync();
        }
        private void LoadModelEmbedderContext()
        {
    
            model = LLamaWeights.LoadFromFile(modelParams);
            embedder = new LLamaEmbedder(model, modelParams);
            context = model.CreateContext(modelParams);
            IOManager.SendMessage($"Model: {selectedModelName} from {modelDirectoryPath} loaded\n");
            if ((gpuLayerCount == -1) || (gpuLayerCount == maxGpuLayers))
            {
                IOManager.SendMessage("\nAll layers moved to GPU\n");
            }
            else if (gpuLayerCount == 0)
            {
                IOManager.SendMessage("\nCPU inference only\n");
            }
            else if ((gpuLayerCount > 0) && (gpuLayerCount < maxGpuLayers))
            {
                IOManager.SendMessage($"\n{gpuLayerCount}/{maxGpuLayers} possible layers moved to GPU\n");
            }
        }

        private void CreateModelParams()
        {
            modelParams = new ModelParams(selectedModelPath)
            {
                Seed = seed,
                ContextSize = contextSize,
                EmbeddingMode = true, // Needs to be true to retrieve embeddings
                GpuLayerCount = gpuLayerCount,
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

        private async Task CheckAndDownloadModelIfNeeded()
        {
            var filePaths = Directory.GetFiles(modelDirectoryPath);
            if (filePaths.Length == 0)
            {
                IOManager.SendMessage($"\nNo models found in the {modelDirectoryPath}\n");

                // Directly attempt to download the default model without calling another method
                await DownloadManager.DownloadModelAsync("mistral", modelDirectoryPath);

                // Recheck the directory for models after attempting the download
                filePaths = Directory.GetFiles(modelDirectoryPath);
                if (filePaths.Length == 0)
                {
                    IOManager.SendMessage("\nFailed to download the default model. Please verify your internet connection and try again.\n");
                    Environment.Exit(0);
                }
            }
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
