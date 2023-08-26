using System;
using System.Collections;
using System.Collections.Generic;
using LLama.Sampling.Logits;
using llama_token = System.Int32;

namespace LLama.Sampling.Bias;

/// <summary>
/// Saves logit values for tokens into a dictionary
/// </summary>
public class SaveLogits
    : ILogitModifier, IEnumerable
{
    private readonly HashSet<llama_token> _save = new();
    private readonly Dictionary<int, float> _output;
    private readonly bool _clear;

    /// <summary>
    /// Initialize a new SaveLogits which will save into the given dictionary
    /// </summary>
    /// <param name="output">Dictionary to save logit values</param>
    /// <param name="clear">Indicates if the dictionary should be cleaed before writing</param>
    public SaveLogits(Dictionary<llama_token, float> output, bool clear = true)
    {
        _output = output;
        _clear = clear;
    }

    /// <summary>
    /// Add a new token to be saved
    /// </summary>
    /// <param name="token"></param>
    public void Add(llama_token token)
    {
        _save.Add(token);
    }

    /// <summary>
    /// Remove a token from being saved
    /// </summary>
    /// <param name="token"></param>
    public void Remove(llama_token token)
    {
        _save.Remove(token);
    }

    /// <inheritdoc />
    public void Apply(Span<float> logits)
    {
        if (_clear)
            _output.Clear();

        foreach (var token in _save)
            _output.Add(token, logits[token]);
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return _save.GetEnumerator();
    }
}