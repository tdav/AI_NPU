using System.ComponentModel;
using System.Runtime.CompilerServices;
using AiNpu.Services;

namespace AiNpu.Features.Models;

// Карточка модели на экране «Модели»: состояние собирается из файлов (ModelManager) и фоновой загрузки (ModelDownloads).
public sealed class ModelItem(ModelInfo info) : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    public ModelInfo Info { get; } = info;

    public string Title => this.Info.Title;

    public string Description => this.Info.Description;

    public string Status { get; private set; } = string.Empty;

    public double Progress { get; private set; }

    public bool ShowProgress { get; private set; }

    public string ActionText { get; private set; } = string.Empty;

    public bool ShowAction { get; private set; }

    public bool ShowSelect { get; private set; }

    public bool ShowDelete { get; private set; }

    public bool IsSelected { get; private set; }

    public bool IsReady { get; private set; }

    public bool IsRunning { get; private set; }

    public Color CardStroke => this.IsSelected
        ? (Color)Application.Current!.Resources["TnGoldBase"]
        : (Color)Application.Current!.Resources["TnBorderSubtle"];

    public double CardStrokeThickness => this.IsSelected ? 2 : 1;

    private static readonly string[] ChangedProperties =
    [
        nameof(Status), nameof(Progress), nameof(ShowProgress), nameof(ActionText), nameof(ShowAction), nameof(ShowSelect),
        nameof(ShowDelete), nameof(IsSelected), nameof(IsReady), nameof(IsRunning), nameof(CardStroke), nameof(CardStrokeThickness),
    ];

    public void Refresh(ModelDownloads downloads)
    {
        var manager = new ModelManager(this.Info);
        ModelDownloads.Job? job = downloads.Get(this.Info.Id);
        string? error = downloads.Error(this.Info.Id);
        long total = manager.TotalBytes;
        long have = job?.Done ?? manager.DownloadedBytes();
        this.IsRunning = job is not null;
        this.IsReady = !this.IsRunning && manager.IsReady();
        this.IsSelected = ModelCatalog.Selected.Id == this.Info.Id;

        if (this.IsRunning)
        {
            long jobTotal = job!.Total > 0 ? job.Total : total;
            this.Status = jobTotal > 0
                ? $"Скачивание: {Gb(have)} из {Gb(jobTotal)}" + (job.Speed > 0 ? $" · {job.Speed / (1 << 20):0.0} МБ/с" : string.Empty)
                : "Подключение…";
            this.Progress = jobTotal > 0 ? (double)have / jobTotal : 0;
            this.ActionText = "Пауза";
        }
        else if (this.IsReady)
        {
            this.Status = $"Скачана · {Gb(total)}" + (manager.HasUpdate ? " · есть новая версия" : string.Empty);
            this.ActionText = manager.HasUpdate ? "Обновить" : string.Empty;
        }
        else if (have > 0 && total > 0)
        {
            this.Status = (error is null ? string.Empty : $"Ошибка: {error}\n") + $"Скачано {Gb(have)} из {Gb(total)}";
            this.Progress = (double)have / total;
            this.ActionText = "Продолжить";
        }
        else
        {
            this.Status = (error is null ? string.Empty : $"Ошибка: {error}\n") + $"Не скачана · ≈{Gb(total > 0 ? total : this.Info.ApproxBytes)}";
            this.ActionText = error is null ? "Скачать" : "Повторить";
        }

        this.ShowProgress = this.IsRunning || (!this.IsReady && have > 0 && total > 0);
        this.ShowAction = this.ActionText.Length > 0;
        this.ShowSelect = this.IsReady && !this.IsSelected;
        this.ShowDelete = !this.IsRunning && (this.IsReady || have > 0 || File.Exists(manager.MetadataPath));
        foreach (string name in ChangedProperties)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public static string Gb(long bytes) => $"{bytes / 1e9:0.00} ГБ";
}
