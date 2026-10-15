using AiNpu.Features.Chat;

namespace AiNpu;

public partial class App : Microsoft.Maui.Controls.Application
{
    private readonly IServiceProvider services;

    public App(IServiceProvider services)
    {
        this.services = services;
        this.InitializeComponent();
        // Окно не сдвигается под клавиатуру (adjustPan по умолчанию уводит шапку за экран): поле ввода
        // поднимает SafeAreaEdges="SoftInput" страницы чата.
        Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific.Application.SetWindowSoftInputModeAdjust(
            this, Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific.WindowSoftInputModeAdjust.Resize);
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
