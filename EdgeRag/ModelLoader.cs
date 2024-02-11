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
        private bool useDatabase;
        private string directoryPath;
        private uint contextSize;
        private string[] facts;
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


        public ModelLoader(string directoryPath, string[] facts, uint contextSize, int num_gpu_layers, uint numCpuThreads, bool useDatabase)
        {
            this.directoryPath = directoryPath;
            this.useDatabase = useDatabase;
            this.contextSize = contextSize;
            this.numGpuLayers = num_gpu_layers;
            this.numCpuThreads = numCpuThreads;
            this.facts = facts;
            SelectedModelPath = "";
            dt = new DataTable();
        }

        public async Task InitializeAsync(IInputHandler inputHandler)
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
                return;
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

            if (useDatabase && embedder != null)
            {
                InitializeDataTable();
            }
        }

        private void InitializeDataTable()
        {
            dt.Columns.Add("llamaEmbedding", typeof(float[]));
            dt.Columns.Add("mistralEmbedding", typeof(float[]));
            dt.Columns.Add("mixtralEmbedding", typeof(float[]));
            dt.Columns.Add("phiEmbedding", typeof(float[]));
            dt.Columns.Add("originalText", typeof(string));

            onMessage?.Invoke("\nEmbedding facts with LLM in vector database, please wait.\n");

            foreach (var fact in facts)
            {
                var embeddings = embedder.GetEmbeddings(fact);
                float[]? llamaEmbedding = null, mistralEmbedding = null, mixtralEmbedding = null, phiEmbedding = null;

                switch (modelType)
                {
                    case "llama":
                        llamaEmbedding = embeddings;
                        break;
                    case "mistral":
                        mistralEmbedding = embeddings;
                        break;
                    case "mixtral":
                        mixtralEmbedding = embeddings;
                        break;
                    case "phi":
                        phiEmbedding = embeddings;
                        break;
                    default:
                        onMessage?.Invoke($"Unsupported model type: {modelType}");
                        break;
                }

                dt.Rows.Add(llamaEmbedding, mistralEmbedding, mixtralEmbedding, phiEmbedding, fact);
            }
            onMessage?.Invoke("Facts embedded!");
        }
    }

    public class ModelLoaderConsole : ModelLoader
    {
        public ModelLoaderConsole(string directoryPath, string[] facts, uint contextSize, int numGpuLayers, uint numCpuThreads, bool useDatabase) : base(directoryPath, facts, contextSize, numGpuLayers, numCpuThreads, useDatabase)
        {
            onMessage += Console.WriteLine;
        }
    }
}
