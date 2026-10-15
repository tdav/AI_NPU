using System.Text.Json;
using AiNpu.Services;

namespace AiNpu.Tests;

// Сверка токенизатора с HF tokenizers 0.23.2 на двух схемах BPE: byte-level (LFM2) и SentencePiece-подобной
// с byte_fallback (Gemma 4). TestData/<модель>/tokenizer-cases.json: русский, английский, код, числа, эмодзи,
// CJK, спецтокены, пробелы/переводы строк, случайные смеси.
public class HfTokenizerTests
{
    private static readonly Dictionary<string, HfTokenizer> tokenizers = new()
    {
        ["lfm2"] = Load("lfm2"),
        ["gemma4"] = Load("gemma4"),
    };

    private static readonly ChatTemplate lfm2Chat = new("<|startoftext|>", "<|im_start|>system\n{content}<|im_end|>\n",
        "<|im_start|>user\n{content}<|im_end|>\n", "<|im_start|>assistant\n", "<|im_end|>\n");

    private static readonly ChatTemplate gemmaChat = new("<bos>", "<|turn>system\n{content}<turn|>\n",
        "<|turn>user\n{content}<turn|>\n", "<|turn>model\n", "<turn|>\n");

    public static IEnumerable<object[]> Cases()
    {
        foreach (string model in new[] { "lfm2", "gemma4" })
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", model, "tokenizer-cases.json")));
            int index = 0;
            foreach (JsonElement c in doc.RootElement.EnumerateArray())
            {
                yield return [model, index++, c.GetProperty("text").GetString()!, c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray()];
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Encode_MatchesHuggingFace(string model, int index, string text, int[] expected)
    {
        Assert.True(index >= 0);
        Assert.Equal(expected, tokenizers[model].Encode(text));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Decode_RestoresTextWithoutSpecialTokens(string model, int index, string text, int[] ids)
    {
        Assert.True(index >= 0);
        string expected = text;
        foreach (string special in new[] { "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<bos>", "<|turn>", "<turn|>" })
        {
            expected = expected.Replace(special, string.Empty);
        }

        Assert.Equal(expected, tokenizers[model].Decode(ids));
    }

    [Fact]
    public void ChatTemplate_Lfm2FirstTurnMatchesChatTemplate()
    {
        HfTokenizer t = tokenizers["lfm2"];
        Assert.Equal(t.Encode("<|startoftext|><|im_start|>system\nТы помощник.<|im_end|>\n<|im_start|>user\nПривет<|im_end|>\n<|im_start|>assistant\n"),
            lfm2Chat.FirstTurn(t, "Ты помощник.", "Привет"));
    }

    [Fact]
    public void ChatTemplate_GemmaFirstTurnMatchesHuggingFace()
    {
        // ids из HF tokenizers для промпта chat_template.jinja Gemma 4 (без системного промпта)
        int[] expected = [2, 105, 2364, 107, 116130, 236888, 217598, 16632, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107];
        Assert.Equal(expected, gemmaChat.FirstTurn(tokenizers["gemma4"], string.Empty, "Привет! Сколько будет 2+2?"));
    }

    [Fact]
    public void ChatTemplate_NextTurnClosesAssistantAndOpensUser()
    {
        HfTokenizer t = tokenizers["lfm2"];
        Assert.Equal(t.Encode("<|im_end|>\n<|im_start|>user\nЕщё<|im_end|>\n<|im_start|>assistant\n"), lfm2Chat.NextTurn(t, "Ещё"));
    }

    private static HfTokenizer Load(string model) => new(Path.Combine(AppContext.BaseDirectory, "TestData", model, "tokenizer.json"));
}
