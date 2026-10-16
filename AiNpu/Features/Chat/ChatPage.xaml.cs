using System.Collections.ObjectModel;
using AiNpu.Features.Models;
using AiNpu.Features.Settings;
using AiNpu.Services;

namespace AiNpu.Features.Chat;

// Чат с выбранной моделью (ModelCatalog) на NPU. До чата — подготовка: проверка SoC (SM8650), модель скачана
// (иначе — экран «Модели», он же по нажатию на название в шапке), загрузка в NPU (≈1–2 с). Ответ приходит потоком; «Стоп» обрывает генерацию на текущем токене.
public partial class ChatPage : ContentPage
{
    private readonly ChatService chatService;
    private readonly IServiceProvider services;
    private CancellationTokenSource? operation;
    private readonly ModelDownloads downloads;
    private bool generating;
    private bool loading;

    public ChatPage(ChatService chatService, ModelDownloads downloads, IServiceProvider services)
    {
        this.chatService = chatService;
        this.downloads = downloads;
        this.services = services;
        this.InitializeComponent();
        this.BindingContext = this;
        this.downloads.Changed += this.OnDownloadChanged;
    }

    public ObservableCollection<ChatMessage> Messages { get; } = [];

    private static ModelManager SelectedModel => new(ModelCatalog.Selected);

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (this.loading || this.generating)
        {
            return;
        }

        ModelInfo model = ModelCatalog.Selected;
        this.TitleLabel.Text = model.Title + " ▾";
        if (this.chatService.LoadedModelId is not null && this.chatService.LoadedModelId != model.Id)
        {
            this.Messages.Clear();  // выбрана другая модель — прежний диалог к ней не относится
        }

        if (!NpuPlatform.IsSupported)
        {
            this.ShowSetup(
                "Нужен Snapdragon 8 Gen 3",
                $"Модель собрана под нейропроцессор Hexagon v75 (SM8650). На этом устройстве SoC — {NpuPlatform.SocModel}, NPU-сборка не запустится.",
                null);
            this.StatusLabel.Text = "NPU не поддерживается";
            return;
        }

        ModelManager manager = SelectedModel;
        ModelDownloads.Job? job = this.downloads.Get(model.Id);
        if (job is not null)
        {
            this.ShowSetup(
                "Скачивание " + model.Title,
                "Модель скачивается в фоне; чат откроется, когда загрузка закончится. Пауза и другие модели — в разделе «Модели».",
                "Модели");
            this.DownloadProgress.IsVisible = this.DownloadLabel.IsVisible = true;
            this.DownloadProgress.Progress = job.Total > 0 ? (double)job.Done / job.Total : 0;
            this.DownloadLabel.Text = job.Total > 0
                ? $"{FormatGb(job.Done)} из {FormatGb(job.Total)}" + (job.Speed > 0 ? $" · {job.Speed / (1 << 20):0.0} МБ/с" : string.Empty)
                : "Подключение…";
            this.StatusLabel.Text = "скачивание модели";
            return;
        }

        if (!manager.IsReady())
        {
            this.ShowSetup(
                model.Title + " не скачана",
                "Языковые модели для NPU скачиваются из HuggingFace один раз (1,8–4,7 ГБ), дальше всё работает без интернета. "
                    + "В разделе «Модели» — список, размеры, загрузка и выбор модели для чата.",
                "Модели");
            this.StatusLabel.Text = "модель не скачана";
            return;
        }

        if (this.chatService.LoadedModelId != model.Id)
        {
            await this.LoadModelAsync();
            return;
        }

        this.ShowChat();
    }

    private async Task LoadModelAsync()
    {
        this.loading = true;
        this.ShowSetup("Загрузка в NPU", "Модель загружается в нейропроцессор…", null);
        this.SetupSpinner.IsVisible = this.SetupSpinner.IsRunning = true;
        this.StatusLabel.Text = "загрузка…";
        try
        {
            await this.chatService.LoadAsync(NpuPlatform.NativeLibraryDirectory, SelectedModel);
            this.ShowChat();
        }
        catch (Exception ex)
        {
            this.ShowSetup("NPU недоступен", ex.Message, "Повторить");
            this.StatusLabel.Text = "ошибка NPU";
        }
        finally
        {
            this.loading = false;
            this.SetupSpinner.IsVisible = this.SetupSpinner.IsRunning = false;
        }
    }

    private async void OnSetupClicked(object? sender, EventArgs e)
    {
        if (SelectedModel.IsReady() && NpuPlatform.IsSupported)
        {
            await this.LoadModelAsync();
            return;
        }

        await this.OpenModelsAsync();
    }

    private async void OnTitleTapped(object? sender, TappedEventArgs e)
    {
        if (!this.generating)
        {
            await this.OpenModelsAsync();
        }
    }

    private Task OpenModelsAsync() => this.Navigation.PushAsync(this.services.GetRequiredService<ModelsPage>());

    // Пока чат ждёт выбранную модель, прогресс её фоновой загрузки виден и здесь; по окончании — загрузка в NPU.
    private void OnDownloadChanged(string modelId)
    {
        if (modelId != ModelCatalog.Selected.Id || !this.SetupView.IsVisible || this.Navigation.NavigationStack[^1] != this)
        {
            return;
        }

        this.OnAppearing();
    }

    private async void OnSendClicked(object? sender, EventArgs e)
    {
        if (this.generating)
        {
            this.operation?.Cancel();
            return;
        }

        string text = (this.InputEditor.Text ?? string.Empty).Trim();
        if (text.Length == 0 || !this.chatService.IsLoaded)
        {
            return;
        }

        // generating — до очистки поля: TextChanged иначе выключит кнопку, которая сейчас станет «Стоп»
        this.generating = true;
        this.InputEditor.Text = string.Empty;
        this.Messages.Add(new ChatMessage { IsUser = true, Text = text });
        var answer = new ChatMessage { IsUser = false, Text = "…" };
        this.Messages.Add(answer);
        this.operation = new CancellationTokenSource();
        this.SendButton.Text = "Стоп";
        this.SendButton.IsEnabled = true;
        this.NewChatButton.IsEnabled = false;
        this.StatusLabel.Text = "думает…";
        bool first = true;
        try
        {
            GenerationStats stats = await this.chatService.SendAsync(text, GenerationSettings.Load(), delta => this.Dispatcher.Dispatch(() =>
            {
                answer.Text = first ? delta.TrimStart() : answer.Text + delta;
                first = false;
            }), this.operation.Token);
            if (first)
            {
                answer.Text = this.operation.IsCancellationRequested ? "(остановлено)" : "(пустой ответ)";
            }

            answer.Meta = $"{stats.GeneratedTokens} ток · {stats.DecodeTokensPerSecond:0.0} ток/с · промпт {stats.PromptTokens} ток за {stats.PrefillSeconds:0.00} с"
                + (stats.ContextRebuilt ? " · контекст пересобран" : string.Empty);
            this.StatusLabel.Text = $"NPU · {stats.DecodeTokensPerSecond:0.0} ток/с · контекст {this.chatService.ContextTokens}/{this.chatService.ContextLength}";
        }
        catch (Exception ex)
        {
            answer.Text = "Ошибка: " + ex.Message;
            this.StatusLabel.Text = "ошибка";
        }
        finally
        {
            this.generating = false;
            this.SendButton.Text = "Отправить";
            this.SendButton.IsEnabled = !string.IsNullOrWhiteSpace(this.InputEditor.Text);
            this.NewChatButton.IsEnabled = true;
        }
    }

    private async void OnNewChatClicked(object? sender, EventArgs e)
    {
        if (this.generating)
        {
            return;
        }

        this.Messages.Clear();
        await this.chatService.NewChatAsync();
        if (this.chatService.IsLoaded)
        {
            this.StatusLabel.Text = "NPU · новый диалог";
        }
    }

    private async void OnSettingsClicked(object? sender, EventArgs e)
    {
        await this.Navigation.PushAsync(this.services.GetRequiredService<SettingsPage>());
    }

    private void OnInputTextChanged(object? sender, TextChangedEventArgs e)
    {
        if (!this.generating)
        {
            this.SendButton.IsEnabled = !string.IsNullOrWhiteSpace(e.NewTextValue);
        }
    }

    private void ShowSetup(string title, string text, string? button)
    {
        this.SetupView.IsVisible = true;
        this.MessagesView.IsVisible = this.InputBar.IsVisible = this.NewChatButton.IsVisible = false;
        this.SetupTitle.Text = title;
        this.SetupText.Text = text;
        this.DownloadProgress.IsVisible = this.DownloadLabel.IsVisible = false;
        this.SetupButton.IsVisible = button is not null;
        this.SetupButton.Text = button ?? string.Empty;
    }

    private void ShowChat()
    {
        this.SetupView.IsVisible = false;
        this.MessagesView.IsVisible = this.InputBar.IsVisible = this.NewChatButton.IsVisible = true;
        this.SendButton.IsEnabled = !string.IsNullOrWhiteSpace(this.InputEditor.Text);
        if (this.StatusLabel.Text is null or "загрузка…" or "модель не скачана")
        {
            this.StatusLabel.Text = $"NPU {NpuPlatform.SocModel} · готово";
        }
    }

    private static string FormatGb(long bytes) => $"{bytes / (double)(1L << 30):0.00} ГБ";
}
