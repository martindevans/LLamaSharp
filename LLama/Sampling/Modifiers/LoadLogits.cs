using System;
using System.Collections.Generic;
using LLama.Native;
using llama_token = System.Int32;

namespace LLama.Sampling.Modifiers;

/// <summary>
/// Overwrites the logits for certain tokens
/// </summary>
public class LoadLogits
    : ITokenModifier
{
    private readonly IReadOnlyDictionary<int, float> _source;

    /// <summary>
    /// Initializes a new instance of the OverwriteTokenLogits that will overwrite the logits of any tokens in the given dictionary
    /// </summary>
    /// <param name="source"></param>
    public LoadLogits(IReadOnlyDictionary<llama_token, float> source)
    {
        _source = source;
    }

    /// <inheritdoc />
    public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray candidates, Memory<int> lastTokens)
    {
        var modified = 0;

        var span = candidates.data.Span;
        for (var i = 0; i < span.Length; i++)
        {
            ref var item = ref span[i];
            if (_source.TryGetValue(item.id, out var logit))
            {
                item.logit = logit;
                candidates.sorted = false;
                modified++;

                if (modified == _source.Count)
                    break;
            }
        }
    }
}