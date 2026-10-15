using System.Runtime.InteropServices;

namespace AiNpu.Services;

// Обёртка над libqnn-llm.so (native/qnn-llm): LLM на NPU по metadata.json пакета qnn-llm-convert.
// Вызовы блокирующие (prefill 128 токенов — доли секунды, шаг decode — десятки мс) — только из фонового потока.
// Один экземпляр на процесс: QNN HTP держит модель в разделяемой с DSP памяти.
public sealed class QnnLlmEngine : IDisposable
{
    private const string Library = "qnn-llm";

    // ADSP_LIBRARY_PATH дополняется один раз на процесс: каждый вызов дописывает путь заново.
    private static int adspPathSet;

    private IntPtr handle;

    private QnnLlmEngine(IntPtr handle)
    {
        this.handle = handle;
        this.ContextLength = QnnLlmContextLength(handle);
    }

    public int ContextLength { get; }

    public int NPast => QnnLlmNPast(this.handle);

    // nativeLibraryDirectory — каталог .so приложения (libQnnHtp.so, libQnnSystem.so и skel для DSP).
    public static QnnLlmEngine Load(string nativeLibraryDirectory, string modelDirectory)
    {
        if (Interlocked.Exchange(ref adspPathSet, 1) == 0)
        {
            QnnLlmPrependAdspLibraryPath(nativeLibraryDirectory);
        }

        IntPtr h = QnnLlmCreate(nativeLibraryDirectory, modelDirectory);
        if (h == IntPtr.Zero)
        {
            throw new InvalidOperationException("NPU: " + Marshal.PtrToStringUTF8(QnnLlmCreateError()));
        }

        return new QnnLlmEngine(h);
    }

    public void Reset() => QnnLlmReset(this.handle);

    public void Append(ReadOnlySpan<int> ids)
    {
        int rc = QnnLlmAppend(this.handle, ids.ToArray(), ids.Length);
        if (rc != 0)
        {
            throw new InvalidOperationException("NPU: " + Marshal.PtrToStringUTF8(QnnLlmLastError(this.handle)));
        }
    }

    public int Sample(GenerationSettings settings) =>
        QnnLlmSample(this.handle, settings.Temperature, settings.MinP, settings.RepetitionPenalty, 0);

    public void Dispose()
    {
        IntPtr h = Interlocked.Exchange(ref this.handle, IntPtr.Zero);
        if (h != IntPtr.Zero)
        {
            QnnLlmDestroy(h);
        }
    }

    [DllImport(Library)]
    private static extern void QnnLlmPrependAdspLibraryPath([MarshalAs(UnmanagedType.LPUTF8Str)] string directory);

    [DllImport(Library)]
    private static extern IntPtr QnnLlmCreate([MarshalAs(UnmanagedType.LPUTF8Str)] string libDir, [MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir);

    [DllImport(Library)]
    private static extern IntPtr QnnLlmCreateError();

    [DllImport(Library)]
    private static extern IntPtr QnnLlmLastError(IntPtr engine);

    [DllImport(Library)]
    private static extern int QnnLlmContextLength(IntPtr engine);

    [DllImport(Library)]
    private static extern int QnnLlmNPast(IntPtr engine);

    [DllImport(Library)]
    private static extern void QnnLlmReset(IntPtr engine);

    [DllImport(Library)]
    private static extern int QnnLlmAppend(IntPtr engine, int[] ids, int count);

    [DllImport(Library)]
    private static extern int QnnLlmSample(IntPtr engine, float temperature, float minP, float repetitionPenalty, ulong seed);

    [DllImport(Library)]
    private static extern void QnnLlmDestroy(IntPtr engine);
}
