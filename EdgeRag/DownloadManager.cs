using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class DownloadManager
    {
        private DownloadManager()
        {

        }

        public static async Task DownloadFileAsync(string url, string destinationFolder)
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
                    IOManager.SendMessage($"Downloading {fileName} from {url} to {destinationFolder}");
                    HttpResponseMessage response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();

                    using (Stream contentStream = await response.Content.ReadAsStreamAsync(),
                                  fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize, true))
                    {
                        byte[] buffer = new byte[bufferSize];
                        int bytesRead;
                        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                        {
                            await fileStream.WriteAsync(buffer, 0, bytesRead);
                        }
                    }

                    IOManager.SendMessage("File downloaded successfully to: " + destinationPath);
                }
                catch (Exception ex)
                {
                    IOManager.SendMessage("Error downloading file: " + ex.Message);
                }
            }
        }
    }
}
