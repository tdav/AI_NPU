using System.Runtime.InteropServices;

namespace AiNpu.Services;

// Обёртка над liblfm2-npu.so (native/lfm2-npu): LFM2-2.6B на NPU, контекст 4096 токенов, prefill чанками по 128.
// Вызовы блокирующие (prefill 128 токенов ≈ 0,25 с, шаг decode ≈ 90 мс) — только из фонового потока.
// Один экземпляр на процесс: QNN HTP держит ≈1,5 ГБ разделяемой с DSP памяти.
public sealed class Lfm2Engine : IDisposable
{
    public const int ContextLength = 4096;
    private const int PrefillChunk = 128;
    private const string Library = "lfm2-npu";

    // ADSP_LIBRARY_PATH дополняется один раз на процесс: каждый вызов дописывает путь заново.
    private static int adspPathSet;

    private IntPtr handle;

    private Lfm2Engine(IntPtr handle)
    {
        this.handle = handle;
    }

    public int NPast => Lfm2NpuNPast(this.handle);

    // nativeLibraryDirectory — каталог .so приложения (там libQnnHtp.so, libQnnSystem.so и skel для DSP).
    public static Lfm2Engine Load(string nativeLibraryDirectory, string modelDirectory)
    {
        if (Interlocked.Exchange(ref adspPathSet, 1) == 0)
        {
            Lfm2NpuPrependAdspLibraryPath(nativeLibraryDirectory);
        }

        IntPtr h = Lfm2NpuCreate(nativeLibraryDirectory, modelDirectory, ContextLength, PrefillChunk);
        if (h == IntPtr.Zero)
        {
            throw new InvalidOperationException("NPU: " + Marshal.PtrToStringUTF8(Lfm2NpuCreateError()));
        }

        return new Lfm2Engine(h);
    }

    public void Reset() => Lfm2NpuReset(this.handle);

    public void Append(ReadOnlySpan<int> ids)
    {
        int rc = Lfm2NpuAppend(this.handle, ids.ToArray(), ids.Length);
        if (rc != 0)
        {
            throw new InvalidOperationException("NPU: " + Marshal.PtrToStringUTF8(Lfm2NpuLastError(this.handle)));
        }
    }

    public int Sample(GenerationSettings settings) =>
        Lfm2NpuSample(this.handle, settings.Temperature, settings.MinP, settings.RepetitionPenalty, 0);

    public void Dispose()
    {
        IntPtr h = Interlocked.Exchange(ref this.handle, IntPtr.Zero);
        if (h != IntPtr.Zero)
        {
            Lfm2NpuDestroy(h);
        }
    }

    [DllImport(Library)]
    private static extern void Lfm2NpuPrependAdspLibraryPath([MarshalAs(UnmanagedType.LPUTF8Str)] string directory);

    [DllImport(Library)]
    private static extern IntPtr Lfm2NpuCreate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string libDir,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir,
        int contextLength,
        int ar);

    [DllImport(Library)]
    private static extern IntPtr Lfm2NpuCreateError();

    [DllImport(Library)]
    private static extern IntPtr Lfm2NpuLastError(IntPtr engine);

    [DllImport(Library)]
    private static extern int Lfm2NpuNPast(IntPtr engine);

    [DllImport(Library)]
    private static extern void Lfm2NpuReset(IntPtr engine);

    [DllImport(Library)]
    private static extern int Lfm2NpuAppend(IntPtr engine, int[] ids, int count);

    [DllImport(Library)]
    private static extern int Lfm2NpuSample(IntPtr engine, float temperature, float minP, float repetitionPenalty, ulong seed);

    [DllImport(Library)]
    private static extern void Lfm2NpuDestroy(IntPtr engine);
}
