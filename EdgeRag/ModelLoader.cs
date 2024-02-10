using LLama;
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
        public string SelectedModelPath { get; private set; }
        public string? FullModelName { get; private set; }
        public string? ModelType { get; private set; }
        public ModelParams? ModelParams { get; private set; }
        public LLamaWeights? Model { get; private set; }
        public LLamaEmbedder? Embedder { get; private set; }
        public DataTable Dt { get; private set; }
        public event Action<string> OnMessage = delegate { };

        public ModelLoader(string directoryPath, string[] facts, uint contextSize, bool useDatabase)
        {
            this.directoryPath = directoryPath;
            this.useDatabase = useDatabase;
            this.contextSize = contextSize;
            this.facts = facts;
            SelectedModelPath = "";
            Dt = new DataTable();
        }

        public async Task InitializeAsync(IInputHandler inputHandler)
        {
            if (!Directory.Exists(directoryPath))
            {
                OnMessage?.Invoke("The directory does not exist.");
                Environment.Exit(0);
            }

            var filePaths = Directory.GetFiles(directoryPath);
            if (filePaths.Length == 0)
            {
                OnMessage?.Invoke("No models found in the directory");
                return;
            }

            bool validModelSelected = false;
            while (!validModelSelected)
            {
                for (int i = 0; i < filePaths.Length; i++)
                {
                    OnMessage?.Invoke($"{i + 1}: {Path.GetFileName(filePaths[i])}");
                }

                OnMessage?.Invoke("\nEnter the number of the model you want to load: ");
                if (int.TryParse(await inputHandler.ReadLineAsync(), out int index) && index >= 1 && index <= filePaths.Length)
                {
                    index -= 1;
                    SelectedModelPath = filePaths[index];
                    FullModelName = Path.GetFileNameWithoutExtension(SelectedModelPath);
                    OnMessage?.Invoke($"Model selected: {FullModelName}");
                    validModelSelected = true;

                    ModelType = FullModelName.Split('-')[0].ToLower();
                }
                else
                {
                    OnMessage?.Invoke("Invalid input, please enter a number corresponding to the model list.\n");
                }
            }

            ModelParams = new ModelParams(SelectedModelPath)
            {
                ContextSize = contextSize,
                EmbeddingMode = true,
                GpuLayerCount = 16, // Assuming a default value or this can be passed as a parameter
                Threads = 8 // Assuming a default value or this can be passed as a parameter
            };

            Model = LLamaWeights.LoadFromFile(ModelParams);
            Embedder = new LLamaEmbedder(Model, ModelParams);
            OnMessage?.Invoke($"Model: {FullModelName} from {SelectedModelPath} loaded");

            if (useDatabase)
            {
                InitializeDataTable();
            }
        }

        private void InitializeDataTable()
        {
            Dt.Columns.Add("llamaEmbedding", typeof(float[]));
            Dt.Columns.Add("mistralEmbedding", typeof(float[]));
            Dt.Columns.Add("mixtralEmbedding", typeof(float[]));
            Dt.Columns.Add("phiEmbedding", typeof(float[]));
            Dt.Columns.Add("originalText", typeof(string));

            foreach (var fact in facts)
            {
                var embeddings = Embedder.GetEmbeddings(fact);
                float[]? llamaEmbedding = null, mistralEmbedding = null, mixtralEmbedding = null, phiEmbedding = null;

                switch (ModelType)
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
                        OnMessage?.Invoke($"Unsupported model type: {ModelType}");
                        break;
                }

                Dt.Rows.Add(llamaEmbedding, mistralEmbedding, mixtralEmbedding, phiEmbedding, fact);
            }
            OnMessage?.Invoke("Facts embedded!");
        }
    }

    public class ModelLoaderConsole : ModelLoader
    {
        public ModelLoaderConsole(string directoryPath, string[] facts, uint contextSize, bool useDatabase) : base(directoryPath, facts, contextSize, useDatabase)
        {
            OnMessage += Console.WriteLine;
        }
    }
}
