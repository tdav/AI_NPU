using System.Text.Json;
using AiNpu.Services;

namespace AiNpu.Tests;

// Сверка токенизатора с HF tokenizers: TestData/tokenizer-cases.json — 173 строки (русский, английский, код,
// числа, эмодзи, CJK, спецтокены, пробелы/переводы строк, случайные смеси) и их id из tokenizers 0.23.2.
public class Lfm2TokenizerTests
{
    private static readonly Lfm2Tokenizer tokenizer = new(Path.Combine(AppContext.BaseDirectory, "TestData", "tokenizer.json"));

    public static IEnumerable<object[]> Cases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "tokenizer-cases.json")));
        int index = 0;
        foreach (JsonElement c in doc.RootElement.EnumerateArray())
        {
            yield return [index++, c.GetProperty("text").GetString()!, c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray()];
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Encode_MatchesHuggingFace(int index, string text, int[] expected)
    {
        Assert.True(index >= 0);
        Assert.Equal(expected, tokenizer.Encode(text));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Decode_RestoresTextWithoutSpecialTokens(int index, string text, int[] ids)
    {
        Assert.True(index >= 0);
        string expected = text;
        foreach (string special in new[] { "<|startoftext|>", "<|im_start|>", "<|im_end|>" })
        {
            expected = expected.Replace(special, string.Empty);
        }

        Assert.Equal(expected, tokenizer.Decode(ids));
    }

    [Fact]
    public void ChatPrompt_FirstTurnMatchesChatTemplate()
    {
        // chat_template.jinja LFM2: bos + system + user + приглашение ассистента
        List<int> ids = ChatPrompt.FirstTurn(tokenizer, "Ты помощник.", "Привет");
        Assert.Equal(tokenizer.Encode("<|startoftext|><|im_start|>system\nТы помощник.<|im_end|>\n<|im_start|>user\nПривет<|im_end|>\n<|im_start|>assistant\n"), ids);
        Assert.Equal(Lfm2Tokenizer.BosId, ids[0]);
    }

    [Fact]
    public void ChatPrompt_NextTurnClosesAssistantAndOpensUser()
    {
        Assert.Equal(tokenizer.Encode("<|im_end|>\n<|im_start|>user\nЕщё<|im_end|>\n<|im_start|>assistant\n"),
            ChatPrompt.NextTurn(tokenizer, "Ещё"));
    }
}
