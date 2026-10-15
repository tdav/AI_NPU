using AiNpu.Services;

namespace AiNpu.Features.Settings;

// Параметры генерации и системный промпт (Preferences, сохраняются при уходе со страницы), удаление модели.
public partial class SettingsPage : ContentPage
{
    private readonly ModelManager modelManager;
    private readonly ChatService chatService;

    public SettingsPage(ModelManager modelManager, ChatService chatService)
    {
        this.modelManager = modelManager;
        this.chatService = chatService;
        this.InitializeComponent();
        this.BindingContext = this;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        GenerationSettings s = GenerationSettings.Load();
        this.TemperatureSlider.Value = s.Temperature;
        this.MinPSlider.Value = s.MinP;
        this.PenaltySlider.Value = s.RepetitionPenalty;
        this.MaxTokensSlider.Value = s.MaxNewTokens;
        this.SystemPromptEditor.Text = s.SystemPrompt;
        this.OnSliderChanged(null, null);
        this.ModelInfo.Text = $"LFM2-2.6B q4 для NPU {NpuPlatform.RequiredSoc} (QNN 2.40), {ModelManager.TotalBytes / (double)(1L << 30):0.00} ГБ. "
            + (this.modelManager.IsReady() ? "Скачана." : "Не скачана.")
            + $" Устройство: {NpuPlatform.SocModel}.";
        this.DeleteModelButton.IsEnabled = this.modelManager.IsReady() || this.modelManager.DownloadedBytes() > 0;
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        new GenerationSettings(
            (float)Math.Round(this.TemperatureSlider.Value, 2),
            (float)Math.Round(this.MinPSlider.Value, 2),
            (float)Math.Round(this.PenaltySlider.Value, 2),
            (int)this.MaxTokensSlider.Value,
            this.SystemPromptEditor.Text ?? string.Empty).Save();
    }

    private void OnSliderChanged(object? sender, ValueChangedEventArgs? e)
    {
        this.TemperatureValue.Text = $"{this.TemperatureSlider.Value:0.00}";
        this.MinPValue.Text = $"{this.MinPSlider.Value:0.00}";
        this.PenaltyValue.Text = $"{this.PenaltySlider.Value:0.00}";
        this.MaxTokensValue.Text = $"{(int)this.MaxTokensSlider.Value}";
    }

    private void OnResetClicked(object? sender, EventArgs e)
    {
        GenerationSettings d = GenerationSettings.Default;
        this.TemperatureSlider.Value = d.Temperature;
        this.MinPSlider.Value = d.MinP;
        this.PenaltySlider.Value = d.RepetitionPenalty;
        this.MaxTokensSlider.Value = d.MaxNewTokens;
        this.SystemPromptEditor.Text = d.SystemPrompt;
    }

    private async void OnDeleteModelClicked(object? sender, EventArgs e)
    {
        if (await this.DisplayAlertAsync("Удалить модель?", "Файлы (≈1,8 ГБ) будут удалены; для чата их придётся скачать заново.", "Удалить", "Отмена"))
        {
            // Сначала выгрузка из NPU: эмбеддинги отображены в память из файла модели.
            await this.chatService.UnloadAsync();
            this.modelManager.Delete();
            this.OnAppearing();
        }
    }
}
