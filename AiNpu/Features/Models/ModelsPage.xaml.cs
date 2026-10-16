using System.Collections.ObjectModel;
using AiNpu.Services;

namespace AiNpu.Features.Models;

// Каталог моделей (ModelCatalog): скачать / пауза / продолжить, выбрать для чата, обновить до новой сборки из бакета,
// удалить с телефона. Загрузки идут в ModelDownloads и не прерываются при уходе со страницы.
public partial class ModelsPage : ContentPage
{
    // Запас места сверх размера модели: временные .part, кэш системы.
    private const long FreeSpaceReserve = 300_000_000;

    private readonly ChatService chatService;
    private readonly ModelDownloads downloads;
    private CancellationTokenSource? updateCheck;

    public ModelsPage(ChatService chatService, ModelDownloads downloads)
    {
        this.chatService = chatService;
        this.downloads = downloads;
        this.InitializeComponent();
        this.Items = new(ModelCatalog.Models.Select(m => new ModelItem(m)));
        this.BindingContext = this;
    }

    public ObservableCollection<ModelItem> Items { get; }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        this.downloads.Changed += this.OnDownloadChanged;
        ModelManager.RemoveUnknownModels();
        this.RefreshAll();
        this.updateCheck = new CancellationTokenSource();
        _ = this.CheckUpdatesAsync(this.updateCheck.Token);
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        this.downloads.Changed -= this.OnDownloadChanged;
        this.updateCheck?.Cancel();
    }

    private void OnDownloadChanged(string modelId)
    {
        this.Items.FirstOrDefault(i => i.Info.Id == modelId)?.Refresh(this.downloads);
        this.ShowFreeSpace();
    }

    private void RefreshAll()
    {
        foreach (ModelItem item in this.Items)
        {
            item.Refresh(this.downloads);
        }

        this.ShowFreeSpace();
    }

    private void ShowFreeSpace() => this.FreeSpaceLabel.Text = $"Свободно на телефоне: {ModelItem.Gb(FreeBytes())}";

    private static long FreeBytes() => new DriveInfo(FileSystem.AppDataDirectory).AvailableFreeSpace;

    // Новые сборки в бакетах (metadata.json отличается) — только для скачанных моделей; без сети тихо пропускается.
    private async Task CheckUpdatesAsync(CancellationToken cancellationToken)
    {
        foreach (ModelItem item in this.Items.Where(i => i.IsReady))
        {
            try
            {
                if (await Task.Run(() => new ModelManager(item.Info).CheckUpdateAsync(cancellationToken), cancellationToken))
                {
                    item.Refresh(this.downloads);
                }
            }
            catch (Exception) when (!cancellationToken.IsCancellationRequested)
            {
                // нет сети или бакет недоступен — обновления проверятся в следующий раз
            }
            catch (OperationCanceledException)
            {
                return;
            }
        }
    }

    private async void OnActionClicked(object? sender, EventArgs e)
    {
        if ((sender as BindableObject)?.BindingContext is not ModelItem item)
        {
            return;
        }

        if (item.IsRunning)
        {
            this.downloads.Pause(item.Info.Id);
            return;
        }

        var manager = new ModelManager(item.Info);
        if (manager.HasUpdate)
        {
            // Файлы модели заменяются — из NPU её нужно выгрузить (эмбеддинги отображены в память из файла).
            if (this.chatService.LoadedModelId == item.Info.Id)
            {
                await this.chatService.UnloadAsync();
            }

            manager.ApplyUpdate();
        }

        long total = manager.TotalBytes > 0 ? manager.TotalBytes : item.Info.ApproxBytes;
        long need = total - manager.DownloadedBytes() + FreeSpaceReserve;
        if (FreeBytes() < need)
        {
            await this.DisplayAlertAsync(
                "Мало места",
                $"Для {item.Title} нужно ещё {ModelItem.Gb(need)}, свободно {ModelItem.Gb(FreeBytes())}. Удалите ненужную модель или файлы.",
                "OK");
            item.Refresh(this.downloads);
            return;
        }

        this.downloads.Start(item.Info);
    }

    private async void OnSelectClicked(object? sender, EventArgs e)
    {
        if ((sender as BindableObject)?.BindingContext is ModelItem item)
        {
            ModelCatalog.Selected = item.Info;
            await this.Navigation.PopToRootAsync();  // в чат: он загрузит выбранную модель в NPU
        }
    }

    private async void OnDeleteClicked(object? sender, EventArgs e)
    {
        if ((sender as BindableObject)?.BindingContext is not ModelItem item || item.IsRunning)
        {
            return;
        }

        if (await this.DisplayAlertAsync("Удалить модель?", $"Файлы {item.Title} будут удалены с телефона; для чата их придётся скачать заново.", "Удалить", "Отмена"))
        {
            if (this.chatService.LoadedModelId == item.Info.Id)
            {
                await this.chatService.UnloadAsync();
            }

            new ModelManager(item.Info).Delete();
            item.Refresh(this.downloads);
            this.ShowFreeSpace();
        }
    }
}
