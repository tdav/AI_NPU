using System.Collections.ObjectModel;
using AiNpu.Features.Settings;
using AiNpu.Services;

namespace AiNpu.Features.Chat;

// Чат с LFM2 на NPU. До чата — подготовка: проверка SoC (SM8650), скачивание модели (≈1,8 ГБ, с докачкой),
// загрузка в NPU (≈1 с). Ответ приходит потоком; «Стоп» обрывает генерацию на текущем токене.
public partial class ChatPage : ContentPage
{
    private readonly ModelManager modelManager;
    private readonly ChatService chatService;
    private readonly IServiceProvider services;
    private CancellationTokenSource? operation;
    private bool generating;
    private bool downloading;

    public ChatPage(ModelManager modelManager, ChatService chatService, IServiceProvider services)
    {
        this.modelManager = modelManager;
        this.chatService = chatService;
        this.services = services;
        this.InitializeComponent();
        this.BindingContext = this;
    }

    public ObservableCollection<ChatMessage> Messages { get; } = [];

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (this.downloading || this.generating)
        {
            return;
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

        if (!this.modelManager.IsReady())
        {
            long have = this.modelManager.DownloadedBytes();
            this.ShowSetup(
                "Скачать модель",
                $"LFM2-2.6B (4 бита) для NPU: {FormatGb(ModelManager.TotalBytes)} из HuggingFace. Скачивается один раз, дальше всё работает без интернета. Нужен Wi-Fi и {FormatGb(ModelManager.TotalBytes + (1L << 30))} свободного места.",
                have > 0 ? $"Продолжить ({FormatGb(have)} из {FormatGb(ModelManager.TotalBytes)})" : "Скачать");
            this.StatusLabel.Text = "модель не скачана";
            return;
        }

        if (!this.chatService.IsLoaded)
        {
            await this.LoadModelAsync();
            return;
        }

        this.ShowChat();
    }

    private async Task LoadModelAsync()
    {
        this.ShowSetup("Загрузка в NPU", "Модель загружается в нейропроцессор…", null);
        this.SetupSpinner.IsVisible = this.SetupSpinner.IsRunning = true;
        this.StatusLabel.Text = "загрузка…";
        try
        {
            await this.chatService.LoadAsync(NpuPlatform.NativeLibraryDirectory);
            this.ShowChat();
        }
        catch (Exception ex)
        {
            this.ShowSetup("NPU недоступен", ex.Message, "Повторить");
            this.StatusLabel.Text = "ошибка NPU";
        }
        finally
        {
            this.SetupSpinner.IsVisible = this.SetupSpinner.IsRunning = false;
        }
    }

    private async void OnSetupClicked(object? sender, EventArgs e)
    {
        if (this.downloading)
        {
            this.operation?.Cancel();
            return;
        }

        if (this.modelManager.IsReady())
        {
            await this.LoadModelAsync();
            return;
        }

        this.downloading = true;
        this.operation = new CancellationTokenSource();
        this.SetupButton.Text = "Пауза";
        this.DownloadProgress.IsVisible = this.DownloadLabel.IsVisible = true;
        this.DownloadLabel.Text = "Подключение…";
        this.SetupText.Text = "Скачивание… Не закрывайте приложение; при обрыве загрузка продолжится с того же места.";
        DeviceDisplay.Current.KeepScreenOn = true;
        var started = DateTime.UtcNow;
        long startBytes = this.modelManager.DownloadedBytes();
        var progress = new Progress<(long Done, long Total)>(p =>
        {
            this.DownloadProgress.Progress = (double)p.Done / p.Total;
            double seconds = (DateTime.UtcNow - started).TotalSeconds;
            double speed = seconds > 1 ? (p.Done - startBytes) / seconds : 0;
            this.DownloadLabel.Text = $"{FormatGb(p.Done)} из {FormatGb(p.Total)}" + (speed > 0 ? $" · {speed / (1 << 20):0.0} МБ/с" : string.Empty);
        });
        try
        {
            await this.modelManager.DownloadAsync(progress, this.operation.Token);
            this.downloading = false;
            await this.LoadModelAsync();
        }
        catch (OperationCanceledException)
        {
            this.downloading = false;
            this.OnAppearing();
        }
        catch (Exception ex)
        {
            this.downloading = false;
            this.ShowSetup("Ошибка загрузки", ex.Message, "Повторить");
        }
        finally
        {
            DeviceDisplay.Current.KeepScreenOn = false;
            this.DownloadProgress.IsVisible = this.DownloadLabel.IsVisible = this.downloading;
        }
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
            this.StatusLabel.Text = $"NPU · {stats.DecodeTokensPerSecond:0.0} ток/с · контекст {this.chatService.ContextTokens}/{Lfm2Engine.ContextLength}";
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
