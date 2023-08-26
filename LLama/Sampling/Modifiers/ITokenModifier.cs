using System;
using LLama.Native;
using llama_token = System.Int32;

namespace LLama.Sampling.Modifiers;

/// <summary>
/// Applies a penalty to a set of potential tokens
/// </summary>
public interface ITokenModifier
{
    /// <summary>
    /// Apply penalties
    /// </summary>
    /// <param name="ctx">Context this is running in</param>
    /// <param name="candidates">Set of possible tokens to penalise by modifying the logits</param>
    /// <param name="lastTokens">A history of the most recent tokens returned from inference</param>
    public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray candidates, Memory<llama_token> lastTokens);
}