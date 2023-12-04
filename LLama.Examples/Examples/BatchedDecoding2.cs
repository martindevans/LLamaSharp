using System.Text;
using LLama.Batched;
using LLama.Common;

namespace LLama.Examples.Examples;

/// <summary>
/// This demonstrates generating multiple replies to the same prompt, with a shared cache
/// </summary>
/// <remarks>Note that this is currently using the low level API directly, future work will provide a safer C# wrapper over this!</remarks>
public class BatchedDecoding2
{
    private const int n_parallel = 8;
    private const int n_len = 32;

    public static async Task Run()
    {
        Console.Write("Please input your model path: ");
        //var modelPath = Console.ReadLine();
        var modelPath = "C:\\Users\\Martin\\Documents\\Python\\oobabooga_windows\\text-generation-webui\\models\\llama-2-7b-chat.Q5_K_M.gguf";

        var parameters = new ModelParams(modelPath);
        using var model = await LLamaWeights.LoadFromFileAsync(parameters);

        Console.WriteLine("Prompt (leave blank to select automatically):");
        var prompt = "";//Console.ReadLine();
        if (string.IsNullOrWhiteSpace(prompt))
            prompt = "Not many people know that";

        // Create an executor that can evaluate a batch of conversations together
        var executor = new ExperimentalBatchedExecutor(model, parameters);

        // Evaluate the initial prompt to create one conversation
        var start = await executor.Prompt(prompt);

        // Now fork the conversation lots of times
        var conversations = new List<Conversation> { start };
        var decoders = new List<StreamingTokenDecoder> { new(Encoding.UTF8, model) };
        while (conversations.Count < n_parallel)
        {
            conversations.Add(start.Fork());
            decoders.Add(new(Encoding.UTF8, model));
        }

        // Tell the user we're about to start
        if (n_parallel > 1)
        {
            Console.WriteLine();
            Console.WriteLine($"generating {n_parallel} sequences...");
        }

        // Keep running inference until done
        for (var i = 0; i < n_len; i++)
        {
            // Run inference for all conversations
            await executor.Infer();

            // Update each conversation
            for (var j = 0; j < conversations.Count; j++)
            {
                var conversation = conversations[j];
                var decoder = decoders[j];

                // Sample and decode one token
                var token = await conversation.Sample();
                decoder.Add(token);

                // Prompt the conversation with it's own output
                await conversation.Prompt(token);
            }
        }

        var index = 1;
        foreach (var decoder in decoders)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{index++}. {prompt}");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(decoder.Read());
        }

        Console.WriteLine("Press any key to exit demo");
        Console.ReadKey(true);
    }
}