namespace AiNpu.Services;

// Токены промпта по chat_template.jinja LFM2:
// <|startoftext|>[<|im_start|>system\n…<|im_end|>\n]<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…
// Сгенерированный <|im_end|> ассистента в контекст не добавляется — его закрывает следующая реплика (NextTurn),
// поэтому формат одинаков при остановке по EOS, по лимиту и кнопкой «Стоп».
public static class ChatPrompt
{
    public static List<int> FirstTurn(Lfm2Tokenizer tokenizer, string systemPrompt, string userText)
    {
        string system = string.IsNullOrWhiteSpace(systemPrompt) ? string.Empty : $"<|im_start|>system\n{systemPrompt.Trim()}<|im_end|>\n";
        return tokenizer.Encode($"<|startoftext|>{system}<|im_start|>user\n{userText}<|im_end|>\n<|im_start|>assistant\n");
    }

    public static List<int> NextTurn(Lfm2Tokenizer tokenizer, string userText) =>
        tokenizer.Encode($"<|im_end|>\n<|im_start|>user\n{userText}<|im_end|>\n<|im_start|>assistant\n");

    // Весь диалог заново (после переполнения контекста): закрытые пары реплик + новая реплика пользователя.
    public static List<int> Conversation(Lfm2Tokenizer tokenizer, string systemPrompt, IReadOnlyList<(string User, string Assistant)> turns, string userText)
    {
        if (turns.Count == 0)
        {
            return FirstTurn(tokenizer, systemPrompt, userText);
        }

        List<int> ids = FirstTurn(tokenizer, systemPrompt, turns[0].User);
        ids.AddRange(tokenizer.Encode(turns[0].Assistant));
        for (int i = 1; i < turns.Count; i++)
        {
            ids.AddRange(NextTurn(tokenizer, turns[i].User));
            ids.AddRange(tokenizer.Encode(turns[i].Assistant));
        }

        ids.AddRange(NextTurn(tokenizer, userText));
        return ids;
    }
}
