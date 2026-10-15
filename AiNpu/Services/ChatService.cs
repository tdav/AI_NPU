using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace AiNpu.Services;

public sealed record GenerationStats(int PromptTokens, double PrefillSeconds, int GeneratedTokens, double DecodeSeconds, bool ContextRebuilt)
{
    public double DecodeTokensPerSecond => this.DecodeSeconds > 0 ? this.GeneratedTokens / this.DecodeSeconds : 0;
}

// Диалог с моделью на NPU (любая модель qnn-llm-convert: токенизатор, шаблон чата и EOS — из metadata.json). KV-кэш живёт между репликами: новая реплика дописывается к контексту (NextTurn),
// без повторного prefill всей истории. Контекст пересобирается заново (Conversation), если сменили системный
// промпт, после ошибки NPU или когда новая реплика с запасом под ответ не помещается в контекст —
// тогда отбрасываются самые старые пары реплик. Одновременно идёт не больше одной генерации.
public sealed class ChatService : IDisposable
{
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly List<(string User, string Assistant)> turns = [];
    private QnnLlmEngine? engine;
    private HfTokenizer? tokenizer;
    private ChatTemplate? template;
    private HashSet<int> eos = [];
    private string? contextSystemPrompt;
    private bool contextValid;

    public bool IsLoaded => this.engine is not null;

    public string? LoadedModelId { get; private set; }

    public int ContextTokens => this.engine?.NPast ?? 0;

    public int ContextLength => this.engine?.ContextLength ?? 0;

    // Секунды: токенизатор (JSON до 20 МБ) и context binary частей в NPU. Другая загруженная модель выгружается.
    // Исключение — NPU недоступен или пакет модели повреждён.
    public Task LoadAsync(string nativeLibraryDirectory, ModelManager model) => Task.Run(async () =>
    {
        await this.gate.WaitAsync();
        try
        {
            if (this.engine is not null && this.LoadedModelId == model.Model.Id)
            {
                return;
            }

            this.engine?.Dispose();
            this.engine = null;
            this.turns.Clear();
            this.contextValid = false;
            using var meta = JsonDocument.Parse(File.ReadAllText(model.MetadataPath));
            this.template = ChatTemplate.FromMetadata(meta.RootElement.GetProperty("chat"));
            this.eos = [.. meta.RootElement.GetProperty("eos").EnumerateArray().Select(e => e.GetInt32())];
            this.tokenizer = new HfTokenizer(Path.Combine(model.Directory, "tokenizer.json"));
            this.engine = QnnLlmEngine.Load(nativeLibraryDirectory, model.Directory);
            this.LoadedModelId = model.Model.Id;
        }
        finally
        {
            this.gate.Release();
        }
    });

    // Выгрузка из NPU (перед удалением файлов модели). Дождётся конца идущей генерации.
    public async Task UnloadAsync()
    {
        await this.gate.WaitAsync();
        try
        {
            this.engine?.Dispose();
            this.engine = null;
            this.LoadedModelId = null;
            this.turns.Clear();
            this.contextValid = false;
        }
        finally
        {
            this.gate.Release();
        }
    }

    public async Task NewChatAsync()
    {
        await this.gate.WaitAsync();
        try
        {
            this.turns.Clear();
            this.contextValid = false;
            this.engine?.Reset();
        }
        finally
        {
            this.gate.Release();
        }
    }

    // onText получает приращения ответа с фонового потока. Отмена завершает ответ на текущем токене
    // (сказанное остаётся в истории).
    public Task<GenerationStats> SendAsync(string userText, GenerationSettings settings, Action<string> onText, CancellationToken cancellationToken) => Task.Run(async () =>
    {
        await this.gate.WaitAsync(CancellationToken.None);
        try
        {
            QnnLlmEngine engine = this.engine ?? throw new InvalidOperationException("модель не загружена");
            HfTokenizer tokenizer = this.tokenizer!;
            ChatTemplate template = this.template!;

            // Запас под ответ: не больше половины контекста, чтобы длинный MaxNewTokens не съедал историю.
            int budget = engine.ContextLength - Math.Min(settings.MaxNewTokens, engine.ContextLength / 2);
            bool rebuild = !this.contextValid || this.contextSystemPrompt != settings.SystemPrompt || this.turns.Count == 0;
            List<int> prompt = rebuild ? [] : template.NextTurn(tokenizer, userText);
            if (!rebuild && engine.NPast + prompt.Count > budget)
            {
                rebuild = true;
            }

            bool rebuiltHistory = false;
            if (rebuild)
            {
                prompt = template.Conversation(tokenizer, settings.SystemPrompt, this.turns, userText);
                while (prompt.Count > budget && this.turns.Count > 0)
                {
                    this.turns.RemoveAt(0);
                    prompt = template.Conversation(tokenizer, settings.SystemPrompt, this.turns, userText);
                }

                if (prompt.Count > budget)
                {
                    throw new InvalidOperationException($"сообщение слишком длинное: {prompt.Count} токенов при лимите {budget}");
                }

                rebuiltHistory = this.turns.Count > 0;
                engine.Reset();
            }

            this.contextValid = false;
            var watch = Stopwatch.StartNew();
            engine.Append(prompt.ToArray());
            double prefillSeconds = watch.Elapsed.TotalSeconds;
            this.contextSystemPrompt = settings.SystemPrompt;
            this.contextValid = true;

            // UTF-8 символ может делиться между токенами — Decoder держит хвост до следующего токена.
            Decoder utf8 = Encoding.UTF8.GetDecoder();
            var answer = new StringBuilder();
            char[] chars = new char[64];
            int generated = 0;
            watch.Restart();
            try
            {
                while (generated < settings.MaxNewTokens && !cancellationToken.IsCancellationRequested && engine.NPast < engine.ContextLength)
                {
                    int token = engine.Sample(settings);
                    if (token < 0 || this.eos.Contains(token) || tokenizer.IsSpecial(token))
                    {
                        break;
                    }

                    byte[] bytes = tokenizer.DecodeBytes([token]);
                    if (chars.Length < bytes.Length + 4)
                    {
                        chars = new char[bytes.Length + 4];
                    }

                    int n = utf8.GetChars(bytes, 0, bytes.Length, chars, 0, flush: false);
                    if (n > 0)
                    {
                        string delta = new(chars, 0, n);
                        answer.Append(delta);
                        onText(delta);
                    }

                    engine.Append([token]);
                    generated++;
                }
            }
            catch
            {
                // Ошибка NPU посреди ответа: состояние кэша неизвестно — следующий вызов пересоберёт контекст.
                this.contextValid = false;
                throw;
            }

            this.turns.Add((userText, answer.ToString()));
            return new GenerationStats(prompt.Count, prefillSeconds, generated, watch.Elapsed.TotalSeconds, rebuiltHistory);
        }
        finally
        {
            this.gate.Release();
        }
    }, CancellationToken.None);

    public void Dispose()
    {
        this.engine?.Dispose();
        this.engine = null;
    }
}
