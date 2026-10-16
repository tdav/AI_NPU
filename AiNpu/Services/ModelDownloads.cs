namespace AiNpu.Services;

// Загрузки моделей в фоне — живут дольше страницы «Модели» (уход в чат или настройки загрузку не прерывает).
// Состояние по id модели; Changed приходит в UI-потоке.
public sealed class ModelDownloads
{
    private readonly Dictionary<string, Job> jobs = [];
    private readonly Dictionary<string, string> errors = [];

    public event Action<string>? Changed;

    public sealed class Job
    {
        public required CancellationTokenSource Cancellation { get; init; }

        public long Done { get; set; }

        public long Total { get; set; }

        // байт/с с начала текущей загрузки
        public double Speed { get; set; }
    }

    public Job? Get(string modelId) => this.jobs.GetValueOrDefault(modelId);

    public bool IsRunning(string modelId) => this.jobs.ContainsKey(modelId);

    // Последняя ошибка загрузки модели (сбрасывается при следующем старте).
    public string? Error(string modelId) => this.errors.GetValueOrDefault(modelId);

    public void Start(ModelInfo model)
    {
        if (this.jobs.ContainsKey(model.Id))
        {
            return;
        }

        var manager = new ModelManager(model);
        var job = new Job { Cancellation = new CancellationTokenSource(), Done = manager.DownloadedBytes(), Total = manager.TotalBytes };
        this.jobs[model.Id] = job;
        this.errors.Remove(model.Id);
        DeviceDisplay.Current.KeepScreenOn = true;
        this.Changed?.Invoke(model.Id);

        var started = DateTime.UtcNow;
        long startBytes = job.Done;
        var progress = new Progress<(long Done, long Total)>(p =>
        {
            job.Done = p.Done;
            job.Total = p.Total;
            double seconds = (DateTime.UtcNow - started).TotalSeconds;
            job.Speed = seconds > 1 ? (p.Done - startBytes) / seconds : 0;
            this.Changed?.Invoke(model.Id);
        });

        _ = this.RunAsync(model, manager, job, progress);
    }

    public void Pause(string modelId) => this.Get(modelId)?.Cancellation.Cancel();

    private async Task RunAsync(ModelInfo model, ModelManager manager, Job job, IProgress<(long, long)> progress)
    {
        try
        {
            // Progress<T> создан в UI-потоке — отчёты возвращаются туда же; сама загрузка — в пуле.
            await Task.Run(() => manager.DownloadAsync(progress, job.Cancellation.Token));
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            this.errors[model.Id] = ex.Message;
        }
        finally
        {
            this.jobs.Remove(model.Id);
            job.Cancellation.Dispose();
            DeviceDisplay.Current.KeepScreenOn = this.jobs.Count > 0;
            this.Changed?.Invoke(model.Id);
        }
    }
}
