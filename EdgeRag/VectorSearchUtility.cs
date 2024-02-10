using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EdgeRag
{
    public class VectorSearchUtility
    {
        public static double CosineSimilarity(float[] vector1, float[] vector2)
        {
            double dotProduct = 0.0, magnitude1 = 0.0, magnitude2 = 0.0;
            int length = Math.Min(vector1.Length, vector2.Length);

            for (int i = 0; i < length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += Math.Pow(vector1[i], 2);
                magnitude2 += Math.Pow(vector2[i], 2);
            }

            return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
        }
    }
}
