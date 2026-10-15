using Android.OS;

namespace AiNpu.Services;

// Android-часть NPU: модель собрана только под Hexagon v75 (Snapdragon 8 Gen 3, SM8650) —
// на другом SoC context binary не загрузится, поэтому SoC проверяется до загрузки.
public static class NpuPlatform
{
    public const string RequiredSoc = "SM8650";

    public static string SocModel => OperatingSystem.IsAndroidVersionAtLeast(31) ? Build.SocModel ?? "?" : "?";

    public static bool IsSupported => SocModel.Equals(RequiredSoc, StringComparison.OrdinalIgnoreCase);

    public static string NativeLibraryDirectory => Android.App.Application.Context.ApplicationInfo!.NativeLibraryDir!;
}
