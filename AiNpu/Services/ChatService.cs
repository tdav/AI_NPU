using System.Diagnostics;
using System.Text;

namespace AiNpu.Services;

public sealed record GenerationStats(int PromptTokens, double PrefillSeconds, int GeneratedTokens, double DecodeSeconds, bool ContextRebuilt)
{
    public double DecodeTokensPerSecond => this.DecodeSeconds > 0 ? this.GeneratedTokens / this.DecodeSeconds : 0;
}

// Диалог с LFM2 на NPU. KV-кэш живёт между репликами: новая реплика дописывается к контексту (NextTurn),
// без повторного prefill всей истории. Контекст пересобирается заново (Conversation), если сменили системный
// промпт, после ошибки NPU или когда новая реплика с запасом под ответ не помещается в 4096 токенов —
// тогда отбрасываются самые старые пары реплик. Одновременно идёт не больше одной генерации.
public sealed class ChatService : IDisposable
{
    private readonly ModelManager modelManager;
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly List<(string User, string Assistant)> turns = [];
    private Lfm2Engine? engine;
    private Lfm2Tokenizer? tokenizer;
    private string? contextSystemPrompt;
    private bool contextValid;

    public ChatService(ModelManager modelManager)
    {
        this.modelManager = modelManager;
    }

    public bool IsLoaded => this.engine is not null;

    public int ContextTokens => this.engine?.NPast ?? 0;

    // ≈1 с: токенизатор (3 МБ JSON) и три context binary в NPU. Исключение — NPU недоступен.
    public Task LoadAsync(string nativeLibraryDirectory) => Task.Run(async () =>
    {
        await this.gate.WaitAsync();
        try
        {
            if (this.engine is not null)
            {
                return;
            }

            this.tokenizer ??= new Lfm2Tokenizer(this.modelManager.TokenizerPath);
            this.engine = Lfm2Engine.Load(nativeLibraryDirectory, this.modelManager.Directory);
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
            Lfm2Engine engine = this.engine ?? throw new InvalidOperationException("модель не загружена");
            Lfm2Tokenizer tokenizer = this.tokenizer!;

            // Запас под ответ: не больше половины контекста, чтобы длинный MaxNewTokens не съедал историю.
            int budget = Lfm2Engine.ContextLength - Math.Min(settings.MaxNewTokens, Lfm2Engine.ContextLength / 2);
            bool rebuild = !this.contextValid || this.contextSystemPrompt != settings.SystemPrompt || this.turns.Count == 0;
            List<int> prompt = rebuild ? [] : ChatPrompt.NextTurn(tokenizer, userText);
            if (!rebuild && engine.NPast + prompt.Count > budget)
            {
                rebuild = true;
            }

            bool rebuiltHistory = false;
            if (rebuild)
            {
                prompt = ChatPrompt.Conversation(tokenizer, settings.SystemPrompt, this.turns, userText);
                while (prompt.Count > budget && this.turns.Count > 0)
                {
                    this.turns.RemoveAt(0);
                    prompt = ChatPrompt.Conversation(tokenizer, settings.SystemPrompt, this.turns, userText);
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
                while (generated < settings.MaxNewTokens && !cancellationToken.IsCancellationRequested && engine.NPast < Lfm2Engine.ContextLength)
                {
                    int token = engine.Sample(settings);
                    if (token < 0 || tokenizer.IsSpecial(token))
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
