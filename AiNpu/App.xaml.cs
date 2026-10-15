using AiNpu.Features.Chat;
using Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific;

namespace AiNpu;

public partial class App : Microsoft.Maui.Controls.Application
{
    private readonly IServiceProvider services;

    public App(IServiceProvider services)
    {
        this.services = services;
        this.InitializeComponent();
        // Клавиатура поджимает окно: поле ввода чата остаётся видимым над ней.
        this.On<Microsoft.Maui.Controls.PlatformConfiguration.Android>().UseWindowSoftInputModeAdjust(WindowSoftInputModeAdjust.Resize);
    }

    protected override Window CreateWindow(IActivationState? activationState)
    {
        var navigation = new NavigationPage(this.services.GetRequiredService<ChatPage>())
        {
            BarBackgroundColor = Color.FromArgb("#2E3136"),
            BarTextColor = Color.FromArgb("#F5F2EC"),
        };
        return new Window(navigation) { Title = "AI NPU" };
    }
}
