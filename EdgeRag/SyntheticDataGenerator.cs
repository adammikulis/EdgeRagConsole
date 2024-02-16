﻿using System.Data;
using LLama;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class SyntheticDataGenerator
    {
        private IOManager iOManager;
        private ModelManager modelManager;
        private DatabaseManager databaseManager;
        private ConversationManager conversationManager;
        private DataTable vectorDatabase;
        private string jsonDbPath;
        private string json;
        private int maxTokens;
        private int questionBatchSize;
        private string modelName;
        private string embeddingColumnName;
        private long currentIncidentNumber;

        public SyntheticDataGenerator(ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            this.modelManager = modelManager;
            this.databaseManager = databaseManager;
            this.conversationManager = conversationManager;
            this.maxTokens = conversationManager.GetMaxTokens();
            this.vectorDatabase = databaseManager.GetVectorDatabase();
            this.questionBatchSize = questionBatchSize;
            modelName = modelManager.modelName;
            jsonDbPath = databaseManager.jsonDbPath;
            json = "";
            embeddingColumnName = $"{modelName}Embeddings";
        }

        public static async Task<SyntheticDataGenerator> CreateAsync(ModelManager modelManager, DatabaseManager databaseManager, ConversationManager conversationManager, int questionBatchSize)
        {
            var syntheticDataGenerator = new SyntheticDataGenerator(modelManager, databaseManager, conversationManager, questionBatchSize);
            await syntheticDataGenerator.InitializeAsync();
            return syntheticDataGenerator;
        }

        public async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                currentIncidentNumber = databaseManager.highestIncidentNumber;
            });
        }
        private string SelectRandomTheme()
        {
            string[] themes = { "a specific Apple device", "a specific Android device", "a specific Windows device", "a specific printer or copier", "a specific networking device", "a specific piece of software", "a specific piece of tech hardware" };
            Random rand = new Random();
            return themes[rand.Next(themes.Length)];
        }

        public async Task GenerateITDataPipeline(int numQuestions)
        {
            // Sets a minimum of 1 for questionBatchSize
            questionBatchSize = Math.Max(1, questionBatchSize);

            for (int i = 0; i < numQuestions; i++)
            {
                currentIncidentNumber++;
                IOManager.SendMessage($"Generating item {currentIncidentNumber}...\n");
                string selectedTheme = SelectRandomTheme();

                DataRow newRow = vectorDatabase.NewRow();
                newRow["incidentNumber"] = currentIncidentNumber;

                // Sequentially generate and set the content, passing previous content as context (this is what LangChain does)
                string incidentDetails = await conversationManager.InteractWithModelAsync($"As the user, describe a tech issue you are having with {selectedTheme}", maxTokens / 8, false);
                string supportResponse = await conversationManager.InteractWithModelAsync($"As support, work with the user to troubleshoot their issue and ask for any additional information needed for " + incidentDetails, maxTokens / 4, false);
                string userResponse = await conversationManager.InteractWithModelAsync($"As the user, troubleshoot " + incidentDetails + " with tech support's steps: " + supportResponse, maxTokens / 2, false);
                string incidentSolution = await conversationManager.InteractWithModelAsync($"As tech support, solve and summarize" + incidentDetails + " based on " + userResponse, maxTokens, false);

                // Assign generated content to the newRow
                newRow["incidentDetails"] = incidentDetails;
                newRow["supportResponse"] = supportResponse;
                newRow["userResponse"] = userResponse;
                newRow["incidentSolution"] = incidentSolution;

                // Generate embeddings for the incidentDetails
                double[] embeddings = await databaseManager.GenerateEmbeddingsAsync(incidentDetails);
                newRow[embeddingColumnName] = embeddings;

                vectorDatabase.Rows.Add(newRow);

                // Save after every questionBatchSize
                if ((i + 1) % questionBatchSize == 0 || i == numQuestions - 1)
                {
                    json = databaseManager.DataTableToJson(vectorDatabase);
                    databaseManager.SaveJsonToFile(json, jsonDbPath);
                }
            }
        }
    }
}
