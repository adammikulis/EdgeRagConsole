


namespace EdgeRag
{
    public class DownloadManager
    {
        private const string mistralURLPath = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.";
        private DownloadManager()
        {
        }

        // One of the most important methods in the program, uses HttpClient to download a model
        public static async Task DownloadModelAsync(string modelType, string destinationFolder)
        {
            using (HttpClient httpClient = new HttpClient())
            {
                try
                {
                    IOManager.ClearAndPrintHeading("Download a Model");
                    int bufferSize = 8192;
                    string url = "";

                    // This lets the user choose the bit size of the model and returns a full url for downloading
                    url = SelectQuantization(modelType, url);

                    string fileName = Path.GetFileName(url);
                    IOManager.SendMessage($"Filename: {fileName}");
                    string destinationPath = Path.Combine(destinationFolder, fileName);

                    // Download the file
                    IOManager.SendMessage($"\nDownloading {fileName} from {url} to {destinationFolder}\n\n");
                    HttpResponseMessage response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();
                    await WriteStreamToFile(bufferSize, destinationPath, response);

                    IOManager.SendMessage("\nFile downloaded successfully to: " + destinationPath);
                }
                catch (Exception ex)
                {
                    IOManager.SendMessage("\nError downloading file: " + ex.Message);
                }
            }
        }

        // This method writes the downloaded model to a file while displaying to the user the percentage downloaded
        private static async Task WriteStreamToFile(int bufferSize, string destinationPath, HttpResponseMessage response)
        {
            // Used for tracking download progress
            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            long totalReadBytes = 0;
            int readBytes;
            double lastProgress = 0;

            // This sets up the stream to receive the model download (initially into a buffer)
            using (Stream contentStream = await response.Content.ReadAsStreamAsync(), fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize, true))
            {
                byte[] buffer = new byte[bufferSize];

                while ((readBytes = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    await fileStream.WriteAsync(buffer, 0, readBytes);
                    totalReadBytes += readBytes;
                    var progress = (double)totalReadBytes / totalBytes;
                    
                    // Update progress for every 1% increase or more
                    if (progress - lastProgress >= 0.01)
                    {
                        IOManager.SendMessage($"\rDownload progress: {progress * 100:0}%");
                        lastProgress = progress;
                    }
                }
            }
        }

        // This method lets the user choose how many bits they want their model to use and provides a full download url
        private static string SelectQuantization(string modelType, string url)
        {
            var quants = new List<string> { "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0" };
            IOManager.SendMessage($"\nDownloading a {modelType} model, choose a quantization (lower = smaller model):\n");
            for (int i = 0; i < quants.Count; i++)
            {
                IOManager.SendMessage($"{i + 1}. {quants[i].Split('_')[0]}\n");
            }
            var modelQuantIndex = IOManager.ReadLine();
            int index;
            if (int.TryParse(modelQuantIndex, out index) && index > 0 && index <= quants.Count)
            {
                var modelQuant = quants[index - 1];
                if (modelType == "mistral")
                {
                    // Currently only mistral is coded into the downloader
                    url = $"{mistralURLPath}{modelQuant}.gguf";
                }
            }

            return url;
        }
    }
}
