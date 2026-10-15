namespace AiNpu.Services;

// Параметры генерации. Умолчания — рекомендация Liquid AI для LFM2: temperature 0.3, min_p 0.15,
// repetition_penalty 1.05. Хранятся в Preferences (экран «Настройки»).
public sealed record GenerationSettings(float Temperature, float MinP, float RepetitionPenalty, int MaxNewTokens, string SystemPrompt)
{
    public static GenerationSettings Default { get; } = new(0.3f, 0.15f, 1.05f, 1024, "Ты полезный ассистент. Отвечай на языке вопроса.");

    public static GenerationSettings Load() => new(
        Preferences.Get(nameof(Temperature), Default.Temperature),
        Preferences.Get(nameof(MinP), Default.MinP),
        Preferences.Get(nameof(RepetitionPenalty), Default.RepetitionPenalty),
        Preferences.Get(nameof(MaxNewTokens), Default.MaxNewTokens),
        Preferences.Get(nameof(SystemPrompt), Default.SystemPrompt));

    public void Save()
    {
        Preferences.Set(nameof(this.Temperature), this.Temperature);
        Preferences.Set(nameof(this.MinP), this.MinP);
        Preferences.Set(nameof(this.RepetitionPenalty), this.RepetitionPenalty);
        Preferences.Set(nameof(this.MaxNewTokens), this.MaxNewTokens);
        Preferences.Set(nameof(this.SystemPrompt), this.SystemPrompt);
    }
}
