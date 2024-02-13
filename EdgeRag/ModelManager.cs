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
        public string SelectedModelPath { get; private set; }
        public string? modelName;
        public string? modelType { get; private set; }
        public ModelParams? modelParams { get; private set; }
        public LLamaWeights? model { get; private set; }
        public LLamaEmbedder? embedder;
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

        public string GetModelName()
        {
            return this.modelName;
        }

        public LLamaEmbedder GetModelEmbedder()
        {
            return this.embedder;
        }

        public async Task<ModelManagerOutputs> InitializeAsync(IInputHandler inputHandler)
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
                    modelName = Path.GetFileNameWithoutExtension(SelectedModelPath);
                    onMessage?.Invoke($"Model selected: {modelName}");
                    validModelSelected = true;

                    modelType = modelName.Split('-')[0].ToLower();
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
            onMessage?.Invoke($"\nModel: {modelName} from {SelectedModelPath} loaded\n");

            return new ModelManagerOutputs(model, modelType, embedder, modelParams, context);
        }
    }

    public class ModelManagerConsole : ModelManager
    {
        public ModelManagerConsole(string directoryPath, uint seed, uint contextSize, int numGpuLayers, uint numCpuThreads) : base(directoryPath, seed, contextSize, numGpuLayers, numCpuThreads)
        {
            onMessage += Console.WriteLine;
        }
    }

    public class ModelManagerOutputs
    {
        public LLamaWeights model { get; set; }
        public LLamaEmbedder embedder { get; set; }
        public ModelParams modelParams { get; set; }
        public string modelName { get; set; }
        public LLamaContext context { get; set; }
        public DataTable embeddingsTable { get; set; }

        public ModelManagerOutputs(LLamaWeights model, string modelName, LLamaEmbedder embedder, ModelParams modelParams, LLamaContext context)
        {
            this.model = model;
            this.modelName = modelName;
            this.embedder = embedder;
            this.modelParams = modelParams;
            this.context = context;
        }
    }
}


