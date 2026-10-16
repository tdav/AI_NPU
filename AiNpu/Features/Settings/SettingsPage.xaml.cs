using AiNpu.Features.Models;
using AiNpu.Services;

namespace AiNpu.Features.Settings;

// Параметры генерации и системный промпт (Preferences, сохраняются при уходе со страницы); выбранная модель —
// выбор, загрузка и удаление на экране «Модели».
public partial class SettingsPage : ContentPage
{
    private readonly IServiceProvider services;

    public SettingsPage(IServiceProvider services)
    {
        this.services = services;
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
        this.ShowModelInfo();
    }

    private void ShowModelInfo()
    {
        var manager = new ModelManager(ModelCatalog.Selected);
        this.ModelInfo.Text = $"{ModelCatalog.Selected.Title}: {ModelCatalog.Selected.Description}. "
            + (manager.IsReady() ? $"Скачана ({ModelItem.Gb(manager.TotalBytes)})." : "Не скачана.")
            + $" NPU {NpuPlatform.RequiredSoc}, устройство: {NpuPlatform.SocModel}.";
    }

    private async void OnModelsClicked(object? sender, EventArgs e)
    {
        await this.Navigation.PushAsync(this.services.GetRequiredService<ModelsPage>());
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
}
