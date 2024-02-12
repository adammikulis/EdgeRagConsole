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
        protected int numGpuLayers;
        protected uint numCpuThreads;
        protected uint seed;
        public string SelectedModelPath { get; private set; }
        public string? fullModelName { get; private set; }
        public string? modelType { get; private set; }
        public ModelParams? modelParams { get; private set; }
        public LLamaWeights? model { get; private set; }
        public LLamaEmbedder? embedder { get; private set; }
        public LLamaContext? context { get; private set; }
        public event Action<string> onMessage = delegate { };


        public ModelManager(string directoryPath, uint seed, uint contextSize, int num_gpu_layers, uint numCpuThreads)
        {
            this.directoryPath = directoryPath;
            this.contextSize = contextSize;
            this.numGpuLayers = num_gpu_layers;
            this.numCpuThreads = numCpuThreads;
            this.seed = seed;
            SelectedModelPath = "";
        }

        public async Task<ModelLoaderOutputs> InitializeAsync(IInputHandler inputHandler)
        {
            if (!Directory.Exists(directoryPath))
            {
                onMessage?.Invoke("The directory does not exist.");
                Environment.Exit(0);
            }

            var filePaths = Directory.GetFiles(directoryPath);
            if (filePaths.Length == 0)
            {
                onMessage?.Invoke("No models found in the directory");
                Environment.Exit(0);
            }

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                for (int i = 0; i < filePaths.Length; i++)
                {
                    onMessage?.Invoke($"{i + 1}: {Path.GetFileName(filePaths[i])}");
                }

                onMessage?.Invoke("\nEnter the number of the model you want to load: ");
                if (int.TryParse(await inputHandler.ReadLineAsync(), out int index) && index >= 1 && index <= filePaths.Length)
                {
                    index -= 1;
                    SelectedModelPath = filePaths[index];
                    fullModelName = Path.GetFileNameWithoutExtension(SelectedModelPath);
                    onMessage?.Invoke($"Model selected: {fullModelName}");
                    validModelSelected = true;

                    modelType = fullModelName.Split('-')[0].ToLower();
                }
                else
                {
                    onMessage?.Invoke("Invalid input, please enter a number corresponding to the model list.\n");
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
            onMessage?.Invoke($"\nModel: {fullModelName} from {SelectedModelPath} loaded\n");

            return new ModelLoaderOutputs(model, modelType, embedder, modelParams, context);
        }
    }

    public class ModelLoaderConsole : ModelManager
    {
        public ModelLoaderConsole(string directoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads) : base(directoryPath, seed, contextSize, numGpuLayers, numCpuThreads)
        {
            onMessage += Console.WriteLine;
        }
    }

    public class ModelLoaderOutputs
    {
        public LLamaWeights model { get; set; }
        public LLamaEmbedder embedder { get; set; }
        public ModelParams modelParams { get; set; }
        public string modelType { get; set; }
        public LLamaContext context { get; set; }
        public DataTable embeddingsTable { get; set; }

        public ModelLoaderOutputs(LLamaWeights model, string modelType, LLamaEmbedder embedder, ModelParams modelParams, LLamaContext context)
        {
            this.model = model;
            this.modelType = modelType;
            this.embedder = embedder;
            this.modelParams = modelParams;
            this.context = context;
        }
    }
}


