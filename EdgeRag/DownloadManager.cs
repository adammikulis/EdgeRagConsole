using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class DownloadManager
    {
        private const string mistralURLPath = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.";
        private DownloadManager()
        {
        }

        public static async Task<string> GetModelDownloadURL(string modelType)
        {
            var quants = new List<string> { "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0" };
            IOManager.SendMessage("Choose a quantization (lower = smaller model):\n");
            for (int i = 0; i < quants.Count; i++)
            {
                IOManager.SendMessage($"{i + 1}. {quants[i]}\n");
            }
            var modelQuantIndex = await IOManager.ReadLineAsync(); // Assuming this returns the index as string
            int index;
            if (int.TryParse(modelQuantIndex, out index) && index > 0 && index <= quants.Count)
            {
                var modelQuant = quants[index - 1];
                if (modelType == "mistral")
                {
                    return $"{mistralURLPath}{modelQuant}.gguf";
                }
            }
            return "";
        }

        public static async Task DownloadModelAsync(string url, string destinationFolder)
        {
            using (HttpClient httpClient = new HttpClient())
            {
                try
                {
                    int bufferSize = 8192;
                    // Extracting the file name from the URL
                    string fileName = Path.GetFileName(url);

                    // Setting the full destination path
                    string destinationPath = Path.Combine(destinationFolder, fileName);

                    // Download the file
                    IOManager.SendMessage($"\n\nDownloading {fileName} from {url} to {destinationFolder}\n\n");
                    HttpResponseMessage response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();

                    var totalBytes = response.Content.Headers.ContentLength ?? 0;
                    long totalReadBytes = 0;
                    int readBytes;
                    double lastProgress = 0;

                    using (Stream contentStream = await response.Content.ReadAsStreamAsync(),
                                  fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize, true))
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
                                IOManager.SendMessage($"\rDownload progress: {progress * 100:0.00}%");
                                lastProgress = progress;
                            }
                            
                        }
                    }

                    IOManager.SendMessage("\nFile downloaded successfully to: " + destinationPath);
                }
                catch (Exception ex)
                {
                    IOManager.SendMessage("Error downloading file: " + ex.Message);
                }
            }
        }

    }
}
