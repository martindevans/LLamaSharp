using System;
using LLama.Native;

namespace LLama.Sampling.Modifiers
{
    /// <summary>
    /// Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    /// </summary>
    public class FrequencyAndPresencePenalty
        : ITokenModifier
    {
        /// <summary>
        /// Number between -2.0 and 2.0.
        /// Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        /// </summary>
        public float AlphaFrequency { get; set; }

        /// <summary>
        /// Number between -2.0 and 2.0.
        /// Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        /// </summary>
        public float AlphaPresence { get; set; }

        /// <inheritdoc />
        public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray candidates, Memory<int> lastTokens)
        {
            SamplingApi.llama_sample_frequency_and_presence_penalties(ctx, candidates, lastTokens, AlphaFrequency, AlphaPresence);
        }
    }
}
