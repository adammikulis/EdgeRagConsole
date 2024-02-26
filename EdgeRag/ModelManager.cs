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

        // Constructor
        public ModelManager(string modelDirectoryPath, uint seed, uint contextSize, uint numCpuThreads)
        {
            this.modelDirectoryPath = modelDirectoryPath;
            this.contextSize = contextSize;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
        }

        // Factory method
        public static async Task<ModelManager> CreateAsync(string modelDirectoryPath, uint seed, uint contextSize, uint numCpuThreads)
        {
            var modelManager = new ModelManager(modelDirectoryPath, seed, contextSize, numCpuThreads);
            await modelManager.InitializeAsync();
            return modelManager;
        }

        // Initialization
        public async Task InitializeAsync()
        {
            CheckDirectoryExists();
            await CheckAndDownloadModelIfNeeded();
            string[] filePaths = Directory.GetFiles(modelDirectoryPath);

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                IOManager.ClearConsole();
                validModelSelected = DisplayAndSelectModel(filePaths, validModelSelected);
            }

            // GPU initialization depends on which release user is running
            gpuLayerCount = 0;
            #if RELEASECUDA12
                // CUDA-specific initialization
                string windowsCudaPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
                string[] linuxCudaPaths = { "/usr/local/cuda-12.0", "/usr/local/cuda-12.1", "/usr/local/cuda-12.2", "/usr/local/cuda-12.3" };
                if (System.IO.Directory.Exists(windowsCudaPath) || linuxCudaPaths.Any(path => System.IO.Directory.Exists(path)))
                {
                    IOManager.PrintCudaInitialization();
                    string input = IOManager.ReadLine();
                    gpuLayerCount = int.Parse(input);
                    if (gpuLayerCount > 33)
                    {
                        gpuLayerCount = 33; // This is the maximum layer count llama.cpp uses (1 non-repeating layer and 32 repeating layers)
                }
                }
                else
                {
                    IOManager.PrintCudaError();
                }
            #endif

            // CPU initialization
            #if RELEASECPU
                // CPU initialization
                IOManager.SendMessage("Running in CPU mode, no CUDA checks required.");
                    gpuLayerCount = 0;
            #endif

            CreateModelParams();
            LoadModelEmbedderContext(); // Loads the model, embedder, and context needed for the ConversationManager
        }

        // Used to manually unload model to free up memory, each of those classes implements IDisposable
        public void Dispose()
        {
            model.Dispose();
            embedder.Dispose();
            context.Dispose();

            model = null;
            embedder = null;
            context = null;
        }

        // Call the disposal methods
        public void UnloadModel()
        {
            Dispose();
            IOManager.SendMessageLine("Model unloaded successfully.");
        }

        // Method to load a different model is async to not block UI
        public async Task LoadDifferentModelAsync(string modelPath)
        {
            UnloadModel();
            selectedModelPath = modelPath;
            selectedModelName = Path.GetFileNameWithoutExtension(selectedModelPath);
            selectedModelType = selectedModelName.Split('-')[0].ToLower();

            await InitializeAsync();
        }

        // Loads the model, the embedder, and the context variables (embedder needed to return embeddings and context needed for  later InteractiveExecutor/ChatSession
        private void LoadModelEmbedderContext()
        {
            // Load the model into memory, putting the specified amount of layers to the GPU
            model = LLamaWeights.LoadFromFile(modelParams);

            // OpenAI uses a separate embedding model from its chat models, whereas with LLamaSharp your chat model is your embedder
            embedder = new LLamaEmbedder(model, modelParams);
            
            // The context is used for the conversation, takes up its own amount of memory (longer context means more usage)
            context = model.CreateContext(modelParams);
            
            // Inform the user how many layers were moved to the GPU
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

        // ModelParams are passed into the intial model loading
        private void CreateModelParams()
        {
            modelParams = new ModelParams(selectedModelPath)
            {
                Seed = seed,
                ContextSize = contextSize,
                EmbeddingMode = true, // Needs to be true to retrieve embeddings
                GpuLayerCount = gpuLayerCount, // Set to -1 for all layers, 0 for cpu-only, 1-33 for precise layer number control
                Threads = numCpuThreads
            };
        }

        // Used for printing out available models for selection
        private bool DisplayAndSelectModel(string[] filePaths, bool validModelSelected)
        {
            IOManager.ClearAndPrintHeading("Large Language Model Selection");
            IOManager.SendMessageLine($"\nCurrent model directory: {modelDirectoryPath}\n\nAvailable models (choose a number):");
            for (int i = 0; i < filePaths.Length; i++)
            {
                IOManager.SendMessageLine($"{i + 1}: {Path.GetFileName(filePaths[i])}");
            }

            // Get the user's selection
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
                IOManager.SendMessageLine("Invalid input, please enter a number corresponding to the model list.");
            }

            return validModelSelected;
        }

        // These need to be manually updated as models are added, should just get linked to maxTokens in the future
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
                    contextSize = 2048;
                    break;
            }
            IOManager.SendMessage($"{selectedModelType} detected, context size set to {contextSize}\n");
        }

        // Starts the download model process if there are none saved in the directory
        private async Task CheckAndDownloadModelIfNeeded()
        {
            var filePaths = Directory.GetFiles(modelDirectoryPath);
            if (filePaths.Length == 0)
            {
                IOManager.ClearAndPrintHeading("Download a Model");
                IOManager.SendMessageLine($"\nNo models found in the {modelDirectoryPath}");
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

        // Checks for the model directory and creates it if it doesn't exist
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