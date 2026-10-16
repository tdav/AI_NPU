using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;

namespace AiNpu.Services;

// Файлы модели qnn-llm-convert из бакета HuggingFace в AppDataDirectory/models/<id> (не в APK).
// Сначала metadata.json — в нём список файлов с размерами и SHA256 ("files"), затем сами файлы от маленьких
// к большим. Докачка через Range в <файл>.part с переключением зеркал при обрыве или таймауте; после
// загрузки — проверка SHA256 и переименование. Готовность — metadata.json + все файлы нужного размера с
// отметкой проверки SHA256.
public sealed class ModelManager
{
    // Ответ без заголовков или поток без данных дольше этого — зеркало считается недоступным, пробуется следующее.
    private static readonly TimeSpan StallTimeout = TimeSpan.FromSeconds(30);

    // Один HttpClient на приложение. Общий таймаут отключён: тело 600 МБ читается дольше 100 с.
    private static readonly HttpClient httpClient = new() { Timeout = Timeout.InfiniteTimeSpan };

    static ModelManager()
    {
        httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("AiNpu/0.2");
    }

    public ModelManager(ModelInfo model, string? root = null)
    {
        this.Model = model;
        this.Directory = Path.Combine(root ?? Path.Combine(FileSystem.AppDataDirectory, "models"), model.Id);
    }

    public ModelInfo Model { get; }

    public string Directory { get; }

    public string MetadataPath => Path.Combine(this.Directory, "metadata.json");

    // Файлы из metadata.json (без самого metadata.json и README) по возрастанию размера; пусто — metadata ещё нет.
    public IReadOnlyList<(string FileName, long Bytes, string Sha256)> Files()
    {
        if (!File.Exists(this.MetadataPath))
        {
            return [];
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(this.MetadataPath));
        return doc.RootElement.GetProperty("files").EnumerateObject()
            .Select(f => (f.Name, f.Value.GetProperty("bytes").GetInt64(), f.Value.GetProperty("sha256").GetString()!))
            .OrderBy(f => f.Item2).ToList();
    }

    public long TotalBytes => this.Files().Sum(f => f.Bytes);

    // Дёшево: только stat файлов и отметок; SHA256 проверяется один раз — при скачивании.
    public bool IsReady()
    {
        var files = this.Files();
        return files.Count > 0 && files.All(f =>
        {
            var info = new FileInfo(Path.Combine(this.Directory, f.FileName));
            return info.Exists && info.Length == f.Bytes && File.Exists(VerifiedMarker(info.FullName, f.Sha256));
        });
    }

    // Скачано байт (готовые файлы + .part) — для прогресса до начала загрузки.
    public long DownloadedBytes() => this.Files().Sum(f =>
    {
        string path = Path.Combine(this.Directory, f.FileName);
        if (File.Exists(path))
        {
            return f.Bytes;
        }

        var part = new FileInfo(path + ".part");
        return part.Exists ? Math.Min(part.Length, f.Bytes) : 0;
    });

    // progress: (скачано байт, всего байт). Отмена сохраняет .part — следующий вызов докачает.
    public async Task DownloadAsync(IProgress<(long Done, long Total)>? progress, CancellationToken cancellationToken)
    {
        System.IO.Directory.CreateDirectory(this.Directory);
        if (!File.Exists(this.MetadataPath))
        {
            string tmp = this.MetadataPath + ".part";
            File.Delete(tmp);
            await this.FetchAsync("metadata.json", tmp, 0, null, cancellationToken);
            using (JsonDocument.Parse(File.ReadAllText(tmp)))
            {
                // metadata.json должен разбираться: защита от HTML-страницы ошибки с кодом 200
            }

            File.Move(tmp, this.MetadataPath, overwrite: true);
        }

        var files = this.Files();
        long total = files.Sum(f => f.Bytes);
        long done = 0;
        foreach (var (fileName, bytes, sha256) in files)
        {
            string path = Path.Combine(this.Directory, fileName);
            string marker = VerifiedMarker(path, sha256);
            if (File.Exists(path) && new FileInfo(path).Length == bytes && File.Exists(marker))
            {
                done += bytes;
                progress?.Report((done, total));
                continue;
            }

            string part = path + ".part";
            if (File.Exists(path))
            {
                // Файл без отметки: прерванная проверка или файл прежней сборки модели (тот же размер, другой SHA256).
                if (new FileInfo(path).Length == bytes && await HashAsync(path, cancellationToken) == sha256)
                {
                    await File.WriteAllTextAsync(marker, sha256, cancellationToken);
                    done += bytes;
                    progress?.Report((done, total));
                    continue;
                }

                File.Delete(path);
            }

            long baseDone = done;
            await this.FetchAsync(fileName, part, bytes, have => progress?.Report((baseDone + have, total)), cancellationToken);

            if (await HashAsync(part, cancellationToken) != sha256)
            {
                File.Delete(part);
                throw new InvalidDataException($"{fileName}: SHA256 не совпадает (файл удалён, нажмите «Скачать» ещё раз)");
            }

            File.Move(part, path, overwrite: true);
            await File.WriteAllTextAsync(marker, sha256, cancellationToken);
            done += bytes;
            progress?.Report((done, total));
        }

        this.RemoveStaleFiles();
    }

    // metadata.json из бакета, если он отличается от скачанного (новая сборка модели).
    public string UpdatePath => Path.Combine(this.Directory, "metadata.json.new");

    public bool HasUpdate => File.Exists(this.UpdatePath);

    // Сверяет metadata.json с бакетом; новый кладётся рядом (UpdatePath) до ApplyUpdate. Только для скачанной модели.
    public async Task<bool> CheckUpdateAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(this.MetadataPath))
        {
            return false;
        }

        string tmp = this.UpdatePath + ".part";
        File.Delete(tmp);
        await this.FetchAsync("metadata.json", tmp, 0, null, cancellationToken);
        string remote = await File.ReadAllTextAsync(tmp, cancellationToken);
        using (JsonDocument.Parse(remote))
        {
            // защита от HTML-страницы ошибки с кодом 200
        }

        if (remote == await File.ReadAllTextAsync(this.MetadataPath, cancellationToken))
        {
            File.Delete(tmp);
            File.Delete(this.UpdatePath);
            return false;
        }

        File.Move(tmp, this.UpdatePath, overwrite: true);
        return true;
    }

    // Новый metadata.json вместо текущего; дальше DownloadAsync докачает только изменившиеся файлы.
    // Модель должна быть выгружена из NPU: эмбеддинги отображены в память из файлов.
    public void ApplyUpdate()
    {
        if (!this.HasUpdate)
        {
            return;
        }

        foreach (string part in System.IO.Directory.EnumerateFiles(this.Directory, "*.part"))
        {
            File.Delete(part);  // недокачанные куски прежней сборки к новой не подходят
        }

        File.Move(this.UpdatePath, this.MetadataPath, overwrite: true);
    }

    // Удаляет файлы и отметки, которых нет в текущем metadata.json (остатки прежней сборки).
    private void RemoveStaleFiles()
    {
        var keep = new HashSet<string>(StringComparer.Ordinal) { "metadata.json", Path.GetFileName(this.UpdatePath) };
        foreach (var (fileName, _, sha256) in this.Files())
        {
            keep.Add(fileName);
            keep.Add(VerifiedMarker(fileName, sha256));
        }

        foreach (string path in System.IO.Directory.EnumerateFiles(this.Directory))
        {
            if (!keep.Contains(Path.GetFileName(path)))
            {
                File.Delete(path);
            }
        }
    }

    private static async Task<string> HashAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken));
    }

    // Каталоги моделей, которых больше нет в ModelCatalog (прежние версии приложения), — место на телефоне.
    public static void RemoveUnknownModels(string? root = null)
    {
        root ??= Path.Combine(FileSystem.AppDataDirectory, "models");
        if (!System.IO.Directory.Exists(root))
        {
            return;
        }

        foreach (string dir in System.IO.Directory.EnumerateDirectories(root))
        {
            if (ModelCatalog.Models.All(m => m.Id != Path.GetFileName(dir)))
            {
                System.IO.Directory.Delete(dir, recursive: true);
            }
        }
    }

    public void Delete()
    {
        if (System.IO.Directory.Exists(this.Directory))
        {
            System.IO.Directory.Delete(this.Directory, recursive: true);
        }
    }

    // Качает файл бакета в part (докачка с текущей длины) по кругу зеркал. bytes = 0 — размер неизвестен.
    private async Task FetchAsync(string fileName, string part, long bytes, Action<long>? onProgress, CancellationToken cancellationToken)
    {
        string[] mirrors = [.. this.Model.BaseUrls];
        long have = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (bytes > 0 && have > bytes)
        {
            File.Delete(part);
            have = 0;
        }

        Exception? lastError = null;
        bool complete = bytes > 0 && have == bytes;
        for (int attempt = 0; !complete && attempt < mirrors.Length * 3; attempt++)
        {
            string url = mirrors[attempt % mirrors.Length] + fileName;
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
                    have = 0;  // сервер проигнорировал Range — файл заново
                }
                else if (response.StatusCode != HttpStatusCode.OK && response.StatusCode != HttpStatusCode.PartialContent)
                {
                    throw new HttpRequestException($"HTTP {(int)response.StatusCode}");
                }

                await using Stream body = await response.Content.ReadAsStreamAsync(stallCts.Token);
                await using var file = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
                byte[] buffer = new byte[1 << 20];
                long lastReport = 0;
                while (true)
                {
                    stallCts.CancelAfter(StallTimeout);
                    int read = await body.ReadAsync(buffer, stallCts.Token);
                    if (read == 0)
                    {
                        break;
                    }

                    await file.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                    have += read;
                    if (have - lastReport >= 4 << 20)
                    {
                        lastReport = have;
                        onProgress?.Invoke(have);
                    }
                }

                complete = bytes == 0 || have == bytes;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested && ex is HttpRequestException or IOException or OperationCanceledException)
            {
                lastError = ex is OperationCanceledException ? new TimeoutException($"{new Uri(url).Host}: нет ответа {StallTimeout.TotalSeconds:0} с") : ex;
                have = File.Exists(part) ? new FileInfo(part).Length : 0;
            }
        }

        if (!complete)
        {
            throw new HttpRequestException($"{fileName}: не удалось скачать ни с одного зеркала ({lastError?.Message})", lastError);
        }
    }

    // Отметка проверки привязана к ожидаемому SHA256: новая сборка модели требует новой проверки.
    private static string VerifiedMarker(string path, string sha256) => $"{path}.{sha256[..12]}.ok";
}
