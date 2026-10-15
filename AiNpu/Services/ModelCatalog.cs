namespace AiNpu.Services;

// Модели для NPU SM8650, собранные qnn-llm-convert (бакеты HuggingFace с metadata.json).
public sealed record ModelInfo(string Id, string Title, string Description, string Bucket)
{
    // Зеркала: hf-mirror первым — huggingface.co доступен не из всех сетей.
    public IEnumerable<string> BaseUrls =>
    [
        $"https://hf-mirror.com/buckets/{this.Bucket}/resolve/",
        $"https://huggingface.co/buckets/{this.Bucket}/resolve/",
    ];
}

public static class ModelCatalog
{
    public static readonly ModelInfo[] Models =
    [
        new("gemma-4-e4b-q4", "Gemma 4 E4B", "Google, 4B (эфф.), 42 слоя, 262k словарь; ≈2,5 ГБ", "tdavron/gemma-4-e4b-it-q4-qnn-sm8650-bucket"),
        new("lfm2-2.6b-q4", "LFM2 2.6B", "Liquid AI, 2.6B, быстрая (≈11 ток/с); ≈1,8 ГБ", "tdavron/lfm2-2.6b-q4-qnn-sm8650-v2-bucket"),
    ];

    public static ModelInfo Selected
    {
        get
        {
            string id = Preferences.Get("SelectedModel", Models[0].Id);
            return Models.FirstOrDefault(m => m.Id == id) ?? Models[0];
        }
        set => Preferences.Set("SelectedModel", value.Id);
    }
}
