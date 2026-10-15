using AiNpu.Services;

namespace AiNpu.Features.Settings;

// Выбор модели (ModelCatalog), параметры генерации и системный промпт (Preferences, сохраняются при уходе
// со страницы), удаление файлов выбранной модели.
public partial class SettingsPage : ContentPage
{
    private readonly ChatService chatService;

    public SettingsPage(ChatService chatService)
    {
        this.chatService = chatService;
        this.InitializeComponent();
        this.BindingContext = this;
        this.ModelPicker.ItemsSource = ModelCatalog.Models.Select(m => m.Title).ToList();
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
        this.ModelPicker.SelectedIndex = Array.IndexOf(ModelCatalog.Models, ModelCatalog.Selected);
        this.ShowModelInfo();
    }

    private void ShowModelInfo()
    {
        var manager = new ModelManager(ModelCatalog.Selected);
        this.ModelInfo.Text = $"{ModelCatalog.Selected.Description}. "
            + (manager.IsReady() ? $"Скачана ({manager.TotalBytes / (double)(1L << 30):0.00} ГБ)." : "Не скачана.")
            + $" NPU {NpuPlatform.RequiredSoc}, устройство: {NpuPlatform.SocModel}.";
        this.DeleteModelButton.IsEnabled = manager.IsReady() || manager.DownloadedBytes() > 0 || File.Exists(manager.MetadataPath);
    }

    private void OnModelChanged(object? sender, EventArgs e)
    {
        if (this.ModelPicker.SelectedIndex >= 0)
        {
            ModelCatalog.Selected = ModelCatalog.Models[this.ModelPicker.SelectedIndex];
            this.ShowModelInfo();
        }
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
        if (await this.DisplayAlertAsync("Удалить модель?", $"Файлы {ModelCatalog.Selected.Title} будут удалены; для чата их придётся скачать заново.", "Удалить", "Отмена"))
        {
            // Сначала выгрузка из NPU: эмбеддинги отображены в память из файла модели.
            if (this.chatService.LoadedModelId == ModelCatalog.Selected.Id)
            {
                await this.chatService.UnloadAsync();
            }

            new ModelManager(ModelCatalog.Selected).Delete();
            this.OnAppearing();
        }
    }
}
