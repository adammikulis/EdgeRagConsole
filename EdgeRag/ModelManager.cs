// This class manages ands loads the current model based on specific parameters
// It implements IDisposable, allowing the user to load a different model without closing the program

using LLama;
using LLama.Common;

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
                IOManager.ClearConsole();
                validModelSelected = DisplayAndLoadModels(filePaths, validModelSelected);
            }

            // GPU initialization depends on which release user is running
            gpuLayerCount = 0;
            #if RELEASECUDA12
                // CUDA-specific initialization
                string windowsCudaPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1";
                string linuxCudaPath = "/usr/local/cuda-12.1";
                if (System.IO.Directory.Exists(windowsCudaPath) || System.IO.Directory.Exists(linuxCudaPath))
                {
                    IOManager.PrintCudaInitialization();
                    string input = IOManager.ReadLine();
                    gpuLayerCount = int.Parse(input);
                    if (gpuLayerCount > 33)
                    {
                        gpuLayerCount = 33;
                    }
                }
                else
                {
                    IOManager.PrintCudaError();
                }
            #endif

            #if RELEASECPU
                // CPU initialization
                IOManager.SendMessage("Running in CPU mode, no CUDA checks required.");
                    gpuLayerCount = 0;
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
            IOManager.SendMessage("Model unloaded successfully.\n");
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
                IOManager.SendMessage("All layers moved to GPU\n");
            }
            else if (gpuLayerCount == 0)
            {
                IOManager.SendMessage("CPU inference only\n");
            }
            else if ((gpuLayerCount > 0) && (gpuLayerCount < maxGpuLayers))
            {
                IOManager.SendMessage($"{gpuLayerCount}/{maxGpuLayers} possible layers moved to GPU\n");
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

        private bool DisplayAndLoadModels(string[] filePaths, bool validModelSelected)
        {
            IOManager.PrintHeading("Large Language Model Selection");
            IOManager.SendMessage($"\nCurrent model directory: {modelDirectoryPath}\n\nAvailable models:\n");
            for (int i = 0; i < filePaths.Length; i++)
            {
                IOManager.SendMessage($"{i + 1}: {Path.GetFileName(filePaths[i])}\n");
            }
            IOManager.SendMessage("\nEnter the number of the model you want to load: ");

            if (int.TryParse(IOManager.ReadLine(), out int index) && index >= 1 && index <= filePaths.Length)
            {
                index -= 1;
                selectedModelPath = filePaths[index];
                selectedModelName = Path.GetFileNameWithoutExtension(selectedModelPath);
                selectedModelType = selectedModelName.Split('-')[0].ToLower();
                IOManager.SendMessage($"Model selected: {selectedModelName}\n");
                validModelSelected = true;

                // Determine the context size based on the model type
                DetermineMaxContextSize();
            }
            else
            {
                IOManager.SendMessage("Invalid input, please enter a number corresponding to the model list.\n");
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
                IOManager.ClearAndPrintHeading("Download a Model");
                IOManager.SendMessageLine($"\nNo models found in the {modelDirectoryPath}");

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
