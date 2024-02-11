using LLama;
using LLama.Abstractions;
using LLama.Common;
using System.Data;
using System.IO;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class ModelLoader
    {
        private string directoryPath;
        private uint contextSize;
        protected int numGpuLayers;
        protected uint numCpuThreads;
        public string SelectedModelPath { get; private set; }
        public string? fullModelName { get; private set; }
        public string? modelType { get; private set; }
        public ModelParams? modelParams { get; private set; }
        public LLamaWeights? model { get; private set; }
        public LLamaEmbedder? embedder { get; private set; }
        public LLamaContext? context { get; private set; }
        public DataTable dt { get; private set; }
        public event Action<string> onMessage = delegate { };


        public ModelLoader(string directoryPath, uint contextSize, int num_gpu_layers, uint numCpuThreads)
        {
            this.directoryPath = directoryPath;
            this.contextSize = contextSize;
            this.numGpuLayers = num_gpu_layers;
            this.numCpuThreads = numCpuThreads;
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
                ContextSize = contextSize,
                EmbeddingMode = true, // Needs to be true to retrieve embeddings
                GpuLayerCount = numGpuLayers, // Assuming a default value or this can be passed as a parameter
                Threads = numCpuThreads // Assuming a default value or this can be passed as a parameter
            };

            model = LLamaWeights.LoadFromFile(modelParams);
            embedder = new LLamaEmbedder(model, modelParams);
            context = model.CreateContext(modelParams);
            onMessage?.Invoke($"Model: {fullModelName} from {SelectedModelPath} loaded");

            return new ModelLoaderOutputs(model, modelType, embedder, modelParams, context, dt);
        }
    }

    public class ModelLoaderConsole : ModelLoader
    {
        public ModelLoaderConsole(string directoryPath, uint contextSize, int numGpuLayers, uint numCpuThreads) : base(directoryPath, contextSize, numGpuLayers, numCpuThreads)
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

        public ModelLoaderOutputs(LLamaWeights model, string modelType, LLamaEmbedder embedder, ModelParams modelParams, LLamaContext context, DataTable embeddingsTable)
        {
            this.model = model;
            this.modelType = modelType;
            this.embedder = embedder;
            this.modelParams = modelParams;
            this.context = context;
            this.embeddingsTable = embeddingsTable;
        }
    }

}


