using System.Text.Json;

namespace AiNpu.Services;

// Шаблон диалога из metadata.json модели (поле "chat", пишет qnn-llm-convert по chat_template.jinja):
// bos, system/user с {content}, assistant — начало ответа, end_of_turn — конец реплики.
// Сгенерированный конец реплики в контекст не добавляется — его закрывает следующая реплика (NextTurn),
// поэтому формат одинаков при остановке по EOS, по лимиту и кнопкой «Стоп».
public sealed record ChatTemplate(string Bos, string System, string User, string Assistant, string EndOfTurn)
{
    public static ChatTemplate FromMetadata(JsonElement chat) => new(
        chat.GetProperty("bos").GetString()!,
        chat.GetProperty("system").GetString()!,
        chat.GetProperty("user").GetString()!,
        chat.GetProperty("assistant").GetString()!,
        chat.GetProperty("end_of_turn").GetString()!);

    public List<int> FirstTurn(HfTokenizer tokenizer, string systemPrompt, string userText)
    {
        string system = string.IsNullOrWhiteSpace(systemPrompt) ? string.Empty : this.System.Replace("{content}", systemPrompt.Trim());
        return tokenizer.Encode(this.Bos + system + this.User.Replace("{content}", userText) + this.Assistant);
    }

    public List<int> NextTurn(HfTokenizer tokenizer, string userText) =>
        tokenizer.Encode(this.EndOfTurn + this.User.Replace("{content}", userText) + this.Assistant);

    // Весь диалог заново (после переполнения контекста): закрытые пары реплик + новая реплика пользователя.
    public List<int> Conversation(HfTokenizer tokenizer, string systemPrompt, IReadOnlyList<(string User, string Assistant)> turns, string userText)
    {
        if (turns.Count == 0)
        {
            return this.FirstTurn(tokenizer, systemPrompt, userText);
        }

        List<int> ids = this.FirstTurn(tokenizer, systemPrompt, turns[0].User);
        ids.AddRange(tokenizer.Encode(turns[0].Assistant));
        for (int i = 1; i < turns.Count; i++)
        {
            ids.AddRange(this.NextTurn(tokenizer, turns[i].User));
            ids.AddRange(tokenizer.Encode(turns[i].Assistant));
        }

        ids.AddRange(this.NextTurn(tokenizer, userText));
        return ids;
    }
}
