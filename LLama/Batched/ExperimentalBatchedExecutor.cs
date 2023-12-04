using System;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using LLama.Abstractions;
using LLama.Exceptions;
using LLama.Native;

namespace LLama.Batched;

public sealed class ExperimentalBatchedExecutor
    : IDisposable
{
    private int _nextSequenceId;

    internal LLamaContext Context { get; }
    internal LLamaBatchSafeHandle Batch { get; }

    public LLamaWeights Model { get; }

    public ExperimentalBatchedExecutor(LLamaWeights model, IContextParams contextParams)
    {
        Model = model;

        //todo: choose n_tokens?
        var n_tokens = 128;

        //todo: what do those other 2 parameters do?
        Batch = LLamaBatchSafeHandle.Create(n_tokens, 0, 1);
        Context = model.CreateContext(contextParams);
    }

    public async Task<Conversation> Prompt(string prompt)
    {
        // Create a new conversation object
        var conversation = new Conversation(this, GetNextSequenceId());
        await conversation.Prompt(prompt);

        //// Add the prompt to the batch
        //for (var i = 0; i < prompt_tokens.Length; i++)
        //    Batch.LLamaBatchAdd(prompt_tokens[i], i, conversation.ConversationId, false);
        //Debug.Assert(Batch.NativeBatch.n_tokens == prompt_tokens.Length);

        //// llama_decode will output logits only for the last token of the prompt
        //unsafe
        //{
        //    Batch.NativeBatch.logits[Batch.NativeBatch.n_tokens - 1] = 1;
        //}

        //// Evaluate
        //var decodeResult = Context.NativeHandle.Decode(Batch);
        //if (decodeResult != 0)
        //    throw new RuntimeError("llama_decode failed");

        throw new NotImplementedException();

    }

    /// <summary>
    /// Run inference for all conversations in the batch
    /// </summary>
    public async Task Infer()
    {
        //todo: wrap this in Task.Run?
        var status = Context.NativeHandle.Decode(Batch);

        if (status < 0)
            throw new RuntimeError("Failed to decode (error)");
        if (status > 0)
            throw new RuntimeError("Failed to decode (not enough space)");

        Batch.LLamaBatchClear();

        await Task.CompletedTask;
    }

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    internal LLamaSeqId GetNextSequenceId()
    {
        checked
        {
            return (LLamaSeqId)_nextSequenceId++;
        }
    }
}