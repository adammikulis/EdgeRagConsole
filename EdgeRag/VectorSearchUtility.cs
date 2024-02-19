// This utility class is for methods to determine similarity of vectors
// Based on: https://weaviate.io/blog/distance-metrics-in-vector-search
// Currently only supports Cosine Similarity but future iterations could add Euclidean, Manhattan Distance, etc
// Biggest current limitation: a short prompt will never match well against the embeddings based on longer text
// Write longer, more detailed prompts to get better matches until the asymmetry problem is resolved in a later iteration

namespace EdgeRag
{
    public class VectorSearchUtility
    {
        public static double CosineSimilarity(double[] vector1, double[] vector2)
        {
            double dotProduct = 0.0, magnitude1 = 0.0, magnitude2 = 0.0;
            int length = Math.Min(vector1.Length, vector2.Length);

            for (int i = 0; i < length; i++)
            {
                dotProduct += vector1[i] * vector2[i]; // Most deep learning calculations boil down to matrix multiplication
                magnitude1 += Math.Pow(vector1[i], 2);
                magnitude2 += Math.Pow(vector2[i], 2);
            }

            return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2)); // Returns a number from 0 - 1, with 1 being an exact match
        }
    }
}
