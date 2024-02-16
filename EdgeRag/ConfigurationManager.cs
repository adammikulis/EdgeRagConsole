using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class ConfigurationManager
    {
        private IOManager iOManager;
        private string url;
        private string destinationFolder;
        public ConfigurationManager(IOManager iOManager) 
        {
            this.iOManager = iOManager;
            string url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf";
            string destinationFolder = @"C:/ai/models";
        }

        public static void DownloadFile(string url, string destinationFolder)
        {
            using (WebClient webClient = new WebClient())
            {
                try
                {
                    string fileName = url.Substring(url.LastIndexOf("/") + 1);

                    string destinationPath = System.IO.Path.Combine(destinationFolder, fileName);

                    webClient.DownloadFile(url, destinationPath);

                    IOManager.SendMessage("File downloaded successfully to: " + destinationPath);
                }
                catch (Exception e)
                {
                    IOManager.SendMessage("Error downloading file: " + e.Message);
                }
            }
        }
    }
}
