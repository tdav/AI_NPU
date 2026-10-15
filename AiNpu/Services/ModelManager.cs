using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace AiNpu.Services;

// Модель LFM2-2.6B q4 для NPU SM8650 (QAIRT 2.40.0.251030): 3 context binary, эмбеддинги FP16 и tokenizer.json
// из бакета tdavron/lfm2-2.6b-q4-qnn-sm8650-bucket, всего ≈1,82 ГБ. Качается при первом запуске в
// AppDataDirectory/models/lfm2 (не в APK). Докачка через Range в <файл>.part, после загрузки — проверка SHA256,
// затем переименование. Готовность модели — наличие всех файлов нужного размера и отметки проверки SHA256.
public sealed class ModelManager
{
    // Зеркала бакета: hf-mirror первым — huggingface.co доступен не из всех сетей. Содержимое одинаково
    // (проверяется SHA256), поэтому .part докачивается с любого зеркала.
    public static readonly string[] BucketUrls =
    [
        "https://hf-mirror.com/buckets/tdavron/lfm2-2.6b-q4-qnn-sm8650-bucket/resolve/",
        "https://huggingface.co/buckets/tdavron/lfm2-2.6b-q4-qnn-sm8650-bucket/resolve/",
    ];

    // Ответ без заголовков или поток без данных дольше этого — зеркало считается недоступным, пробуется следующее.
    private static readonly TimeSpan StallTimeout = TimeSpan.FromSeconds(30);

    // Порядок — от маленьких к большим: битый мелкий файл обнаруживается до скачивания больших.
    public static readonly (string FileName, long Bytes, string Sha256)[] Files =
    [
        ("tokenizer.json", 3_296_920, "0d97ddf099eae6c0b4392bdffa1256762eb8621de54e04429e0719965036c3d7"),
        ("embed_tokens.f16", 268_435_456, "b4580a0b571d9188005a634405b881be9ce92ba8c69537f91502a9b7cebfd5a0"),
        ("lfm2_p0.bin", 470_806_528, "2392c42d51653e640a6211e3066ffc052f4cb0dee7a2e0d8fc38d898975ce2e0"),
        ("lfm2_p1.bin", 474_587_136, "b10e10581bdae3864f9465ed7d3f0140691417b14f5fb28e6b1d03ec1d9e745f"),
        ("lfm2_p2.bin", 606_752_768, "db4e5db92d7a0249576146d284cabca3dad8dd934c4d2fec6bf1f4ba968b2fe2"),
    ];

    public static long TotalBytes => Files.Sum(f => f.Bytes);

    // Один HttpClient на приложение. Общий таймаут отключён: тело 600 МБ читается дольше 100 с;
    // зависание заголовков ловит отдельный таймаут в DownloadAsync.
    private static readonly HttpClient httpClient = new() { Timeout = Timeout.InfiniteTimeSpan };

    static ModelManager()
    {
        httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("AiNpu/0.1");
    }

    public ModelManager(string? directory = null)
    {
        this.Directory = directory ?? Path.Combine(FileSystem.AppDataDirectory, "models", "lfm2");
    }

    public string Directory { get; }

    public string TokenizerPath => Path.Combine(this.Directory, "tokenizer.json");

    // Дёшево: только stat файлов и отметок, SHA256 проверяется один раз — при скачивании.
    public bool IsReady() => Files.All(f =>
    {
        var info = new FileInfo(Path.Combine(this.Directory, f.FileName));
        return info.Exists && info.Length == f.Bytes && File.Exists(VerifiedMarker(info.FullName, f.Sha256));
    });

    // Сколько байт уже лежит на диске (готовые файлы + .part) — для прогресса до начала загрузки.
    public long DownloadedBytes() => Files.Sum(f =>
    {
        string path = Path.Combine(this.Directory, f.FileName);
        if (File.Exists(path))
        {
            return f.Bytes;
        }

        var part = new FileInfo(path + ".part");
        return part.Exists ? Math.Min(part.Length, f.Bytes) : 0;
    });

    // progress: (скачано байт всего, всего байт). Отмена сохраняет .part — следующий вызов докачает.
    public async Task DownloadAsync(IProgress<(long Done, long Total)>? progress, CancellationToken cancellationToken)
    {
        System.IO.Directory.CreateDirectory(this.Directory);
        long done = 0;
        foreach (var (fileName, bytes, sha256) in Files)
        {
            string path = Path.Combine(this.Directory, fileName);
            string marker = VerifiedMarker(path, sha256);
            if (File.Exists(path) && new FileInfo(path).Length == bytes && File.Exists(marker))
            {
                done += bytes;
                progress?.Report((done, TotalBytes));
                continue;
            }

            string part = path + ".part";
            if (File.Exists(path))
            {
                // Файл без отметки проверки (прервали между переименованием и отметкой) — проверяется заново как .part.
                File.Move(path, part, overwrite: true);
            }

            long have = File.Exists(part) ? new FileInfo(part).Length : 0;
            if (have > bytes)
            {
                File.Delete(part);
                have = 0;
            }

            // Попытки по кругу зеркал; каждая продолжает .part с текущего места. Отмена пользователем
            // пробрасывается сразу, таймаут и сетевые ошибки — переход к следующему зеркалу.
            Exception? lastError = null;
            for (int attempt = 0; have < bytes && attempt < BucketUrls.Length * 3; attempt++)
            {
                string url = BucketUrls[attempt % BucketUrls.Length] + fileName;
                using var stallCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                try
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, url);
                    if (have > 0)
                    {
                        request.Headers.Range = new RangeHeaderValue(have, null);
                    }

                    stallCts.CancelAfter(StallTimeout);
                    using HttpResponseMessage response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, stallCts.Token);
                    if (have > 0 && response.StatusCode == HttpStatusCode.OK)
                    {
                        // Сервер проигнорировал Range — качаем файл целиком заново.
                        have = 0;
                    }
                    else if (response.StatusCode != HttpStatusCode.OK && response.StatusCode != HttpStatusCode.PartialContent)
                    {
                        throw new HttpRequestException($"HTTP {(int)response.StatusCode}");
                    }

                    await using Stream body = await response.Content.ReadAsStreamAsync(stallCts.Token);
                    await using var file = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
                    byte[] buffer = new byte[1 << 20];
                    int read;
                    long lastReport = 0;
                    while (true)
                    {
                        stallCts.CancelAfter(StallTimeout);
                        read = await body.ReadAsync(buffer, stallCts.Token);
                        if (read == 0)
                        {
                            break;
                        }

                        await file.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                        have += read;
                        if (have - lastReport >= 4 << 20)
                        {
                            lastReport = have;
                            progress?.Report((done + have, TotalBytes));
                        }
                    }
                }
                catch (Exception ex) when (!cancellationToken.IsCancellationRequested && ex is HttpRequestException or IOException or OperationCanceledException)
                {
                    lastError = ex is OperationCanceledException ? new TimeoutException($"{new Uri(url).Host}: нет ответа {StallTimeout.TotalSeconds:0} с") : ex;
                    have = File.Exists(part) ? new FileInfo(part).Length : 0;
                }
            }

            if (have < bytes)
            {
                throw new HttpRequestException($"{fileName}: не удалось скачать ни с одного зеркала ({lastError?.Message})", lastError);
            }

            if (new FileInfo(part).Length != bytes)
            {
                throw new IOException($"{fileName}: получено {new FileInfo(part).Length} Б вместо {bytes} Б");
            }

            string actual;
            await using (var stream = new FileStream(part, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
            {
                actual = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken));
            }

            if (actual != sha256)
            {
                File.Delete(part);
                throw new InvalidDataException($"{fileName}: SHA256 не совпадает (файл удалён, нажмите «Скачать» ещё раз)");
            }

            File.Move(part, path, overwrite: true);
            await File.WriteAllTextAsync(marker, sha256, cancellationToken);
            done += bytes;
            progress?.Report((done, TotalBytes));
        }
    }

    public void Delete()
    {
        if (System.IO.Directory.Exists(this.Directory))
        {
            System.IO.Directory.Delete(this.Directory, recursive: true);
        }
    }

    // Отметка проверки привязана к ожидаемому SHA256: новая сборка модели в приложении требует новой проверки.
    private static string VerifiedMarker(string path, string sha256) => $"{path}.{sha256[..12]}.ok";
}
