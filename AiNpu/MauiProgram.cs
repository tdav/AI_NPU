using AiNpu.Features.Chat;
using AiNpu.Features.Models;
using AiNpu.Features.Settings;
using AiNpu.Services;
using Microsoft.Extensions.Logging;

namespace AiNpu;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
                fonts.AddFont("Manrope-Regular.ttf", "Manrope");
                fonts.AddFont("Manrope-Medium.ttf", "ManropeMedium");
                fonts.AddFont("Manrope-SemiBold.ttf", "ManropeSemiBold");
                fonts.AddFont("Manrope-Bold.ttf", "ManropeBold");
                fonts.AddFont("Manrope-ExtraBold.ttf", "ManropeExtraBold");
            });

        // Движок NPU — один на процесс (модель выбирается в ModelCatalog); страница чата держит ленту диалога всё время работы.
        builder.Services.AddSingleton<ChatService>();
        builder.Services.AddSingleton<ChatPage>();
        builder.Services.AddTransient<SettingsPage>();
        builder.Services.AddSingleton<ModelDownloads>();  // загрузки моделей переживают уход со страницы «Модели»
        builder.Services.AddTransient<ModelsPage>();

#if DEBUG
        builder.Logging.AddDebug();
#endif

        return builder.Build();
    }
}
