using System;
using LLama.Native;

namespace LLama.Sampling.Modifiers
{
    /// <summary>
    /// Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    /// </summary>
    public class RepetitionPenalty
        : ITokenModifier
    {
        /// <summary>
        /// Penalty to apply
        /// </summary>
        public float Penalty { get; set; }

        /// <inheritdoc />
        public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray candidates, Memory<int> lastTokens)
        {
            SamplingApi.llama_sample_repetition_penalty(ctx, candidates, lastTokens, Penalty);
        }
    }
}
