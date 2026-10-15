using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace AiNpu.Features.Chat;

// Реплика в ленте чата. Text ответа ассистента растёт по мере генерации, Meta — строка статистики под ответом.
public sealed class ChatMessage : INotifyPropertyChanged
{
    private string text = string.Empty;
    private string meta = string.Empty;

    public event PropertyChangedEventHandler? PropertyChanged;

    public required bool IsUser { get; init; }

    public bool IsAssistant => !this.IsUser;

    public string Text
    {
        get => this.text;
        set => this.Set(ref this.text, value);
    }

    public string Meta
    {
        get => this.meta;
        set
        {
            this.Set(ref this.meta, value);
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(this.HasMeta)));
        }
    }

    public bool HasMeta => this.meta.Length > 0;

    private void Set(ref string field, string value, [CallerMemberName] string? name = null)
    {
        if (field != value)
        {
            field = value;
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
