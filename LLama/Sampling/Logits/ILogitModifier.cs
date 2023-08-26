using System;

namespace LLama.Sampling.Logits
{
    /// <summary>
    /// Directly modify token logits
    /// </summary>
    public interface ILogitModifier
    {
        /// <summary>
        /// Apply biases 
        /// </summary>
        /// <param name="logits">The raw logits for tokens (by index)</param>
        public void Apply(Span<float> logits);
    }
}
