using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLama.Native;

namespace LLama.Batched;

/// <summary>
/// A single conversation thread that can be prompted (adding tokens from the user) or inferred (extracting a token from the LLM)
/// </summary>
public sealed class Conversation
    : IDisposable
{
    private readonly ExperimentalBatchedExecutor _batch;

    /// <summary>
    /// Unique ID for this conversation
    /// </summary>
    public LLamaSeqId ConversationId { get; }

    private (Conversation, int)? _forkedFrom;
    private bool _disposed;

    /// <summary>
    /// Total number of tokens in this conversation, cannot exceed the context length.
    /// </summary>
    public int NTokens => throw new NotImplementedException();

    internal Conversation(ExperimentalBatchedExecutor batch, LLamaSeqId id, (Conversation, int)? fork = null)
    {
        ConversationId = id;

        _batch = batch;
        _forkedFrom = fork;
    }

    /// <summary>
    /// End this conversation, freeing all resources used by it
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public void Dispose()
    {
        if (_disposed)
            throw new ObjectDisposedException("Cannot dispose conversation that has already been disposed");
        _disposed = true;

        throw new NotImplementedException("check if there are any forks using parts of this KV cache");
    }

    /// <summary>
    /// Create a copy of the current conversation.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="ObjectDisposedException"></exception>
    public Conversation Fork()
    {
        if (_disposed)
            throw new ObjectDisposedException("Cannot `Fork()` from a Conversation that has already been disposed");

        var id2 = _batch.GetNextSequenceId();

        // Assign tokens to the new sequence
        NativeApi.llama_kv_cache_seq_cp(_batch.Context.NativeHandle, ConversationId, id2, 0, NTokens);

        // Create a new conversation which references the current position in this one
        return new Conversation(_batch, id2, (this, NTokens));
    }

    /// <summary>
    /// Sample a single token from this conversation
    /// todo: when is it valid to call this?
    /// </summary>
    /// <returns></returns>
    /// <exception cref="ObjectDisposedException"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public Task<int> Sample()
    {
        if (_disposed)
            throw new ObjectDisposedException("Cannot `Sample()` from a Conversation that has already been disposed");

        throw new NotImplementedException();

        ////todo: check that logits were generated for this sequence somehow (does it matter?)

        //var logits = _parent.Context.NativeHandle.GetLogitsIth(NTokens);
        //var candidates = LLamaTokenDataArray.Create(logits);

        ////todo: introduce a way to configure per-conversation sampling
        //const int top_k = 80;
        //const float top_p = 0.8f;
        //const float temp = 0.75f;
        //candidates.TopK(_parent.Context.NativeHandle, top_k);
        //candidates.TopP(_parent.Context.NativeHandle, top_p);
        //candidates.Temperature(_parent.Context.NativeHandle, temp);
        //var new_token_id = candidates.SampleToken(_parent.Context.NativeHandle);

        //// push this new token for next evaluation
        //NTokens++;
        //_parent.Batch.LLamaBatchAdd(new_token_id, NTokens, SequenceId, true);

        //throw new NotImplementedException("Infer one single token");
    }

    /// <summary>
    /// Add tokens to this conversation
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    /// <exception cref="ObjectDisposedException"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public Task Prompt(string input)
    {
        var tokens = _batch.Context.Tokenize(input);
        return Prompt(tokens);
    }

    /// <summary>
    /// Add tokens to this conversation
    /// </summary>
    /// <param name="tokens"></param>
    /// <returns></returns>
    /// <exception cref="ObjectDisposedException"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public Task Prompt(IReadOnlyList<int> tokens)
    {
        if (_disposed)
            throw new ObjectDisposedException("Cannot `Prompt()` a Conversation that has already been disposed");

        // Add the prompt to the batch
        for (var i = 0; i < tokens.Count; i++)
            _batch.Batch.LLamaBatchAdd(tokens[i], i, ConversationId, false);

        throw new NotImplementedException("Add tokenized input");
        throw new NotImplementedException("Mark this conversation as needing logits");
    }

    /// <summary>
    /// Add a single token to this conversation
    /// </summary>
    /// <param name="token"></param>
    /// <returns></returns>
    /// <exception cref="ObjectDisposedException"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public Task Prompt(int token)
    {
        if (_disposed)
            throw new ObjectDisposedException("Cannot `Prompt()` a Conversation that has already been disposed");

        throw new NotImplementedException("Add tokenized input");
        throw new NotImplementedException("Mark this conversation as needing logits");
    }
}