using System;
using System.Collections;
using System.Collections.Generic;
using LLama.Abstractions;
using LLama.Extensions;
using LLama.Sampling.Logits;
using llama_token = System.Int32;

namespace LLama.Sampling.Bias;

/// <summary>
/// Add a value directly to the logit for tokens
/// </summary>
public class LogitBias
    : ILogitModifier, IEnumerable<KeyValuePair<llama_token, float>>
{
    private readonly Dictionary<llama_token, float> _bias;

    /// <summary>
    /// Set the bias value for a token
    /// </summary>
    /// <param name="token"></param>
    /// <returns></returns>
    public float this[llama_token token]
    {
        get => _bias.GetValueOrDefault(token, 0);
        set
        {
            if (Math.Abs(value) <= float.Epsilon)
                _bias.Remove(token);
            else
                _bias[token] = value;
        }
    }

    /// <summary>
    /// Initialize a new LogitBias which will add a bias to any tokens in the given dictionary
    /// </summary>
    /// <param name="bias"></param>
    public LogitBias(Dictionary<llama_token, float>? bias = null)
    {
        _bias = bias ?? new();
    }

    /// <summary>
    /// This method is required for collection initializers to work
    /// </summary>
    /// <param name="token"></param>
    /// <param name="bias"></param>
    private void Add(llama_token token, float bias)
    {
        this[token] = bias;
    }

    /// <inheritdoc />
    public void Apply(Span<float> logits)
    {
        // Early out if there are no biases
        if (_bias.Count == 0)
            return;

        foreach (var (token, bias) in _bias)
            logits[token] += bias;
    }

    IEnumerator<KeyValuePair<llama_token, float>> IEnumerable<KeyValuePair<llama_token, float>>.GetEnumerator()
    {
        return _bias.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable<KeyValuePair<int, float>>)this).GetEnumerator();
    }
}