using System.Diagnostics;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

// qnn-llm-convert.exe — обёртка Python-ядра qnnllm для Windows 11 x64.
// 1) распаковывает вшитое ядро в %LOCALAPPDATA%\qnn-llm-convert\app\<хэш>;
// 2) находит uv (PATH или скачивает в %LOCALAPPDATA%\qnn-llm-convert\uv);
// 3) создаёт venv на Python 3.10 (его ставит uv) и ставит requirements.txt — один раз на версию зависимостей;
// 4) находит QAIRT SDK (--qairt, QNN_SDK_ROOT, QAIRT_SDK_ROOT, C:\Qualcomm\AIStack\QAIRT\<версия>);
// 5) запускает python -m qnnllm с исходными аргументами и возвращает его код выхода.
Console.OutputEncoding = Encoding.UTF8;
string home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "qnn-llm-convert");
Directory.CreateDirectory(home);

try
{
    var assembly = Assembly.GetExecutingAssembly();
    string[] resources = assembly.GetManifestResourceNames().Where(n => n.StartsWith("app/", StringComparison.Ordinal)).Order().ToArray();
    using var resourceHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    foreach (string name in resources)
    {
        using Stream s = assembly.GetManifestResourceStream(name)!;
        resourceHash.AppendData(Encoding.UTF8.GetBytes(name));
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        resourceHash.AppendData(ms.ToArray());
    }

    string appHash = Convert.ToHexStringLower(resourceHash.GetHashAndReset())[..12];
    string appDir = Path.Combine(home, "app", appHash);
    if (!File.Exists(Path.Combine(appDir, ".complete")))
    {
        foreach (string name in resources)
        {
            string target = Path.Combine(appDir, name["app/".Length..].Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            using Stream s = assembly.GetManifestResourceStream(name)!;
            using FileStream f = File.Create(target);
            s.CopyTo(f);
        }

        File.WriteAllText(Path.Combine(appDir, ".complete"), appHash);
    }

    string uv = FindOnPath("uv.exe") ?? Path.Combine(home, "uv", "uv.exe");
    if (!File.Exists(uv))
    {
        Console.WriteLine("Первый запуск: скачиваю uv (менеджер Python)…");
        using var http = new HttpClient();
        byte[] zip = await http.GetByteArrayAsync("https://github.com/astral-sh/uv/releases/download/0.10.8/uv-x86_64-pc-windows-msvc.zip");
        using var archive = new ZipArchive(new MemoryStream(zip));
        Directory.CreateDirectory(Path.GetDirectoryName(uv)!);
        foreach (ZipArchiveEntry e in archive.Entries.Where(e => e.Name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)))
        {
            e.ExtractToFile(Path.Combine(Path.GetDirectoryName(uv)!, e.Name), overwrite: true);
        }
    }

    // venv привязан к содержимому requirements.txt: новые зависимости -> новый venv
    string requirements = Path.Combine(appDir, "requirements.txt");
    string reqHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(requirements)))[..12];
    string venv = Path.Combine(home, "venv-" + reqHash);
    string python = Path.Combine(venv, "Scripts", "python.exe");
    if (!File.Exists(Path.Combine(venv, ".complete")))
    {
        Console.WriteLine("Первый запуск: ставлю Python 3.10 и зависимости (≈300 МБ)…");
        Run(uv, ["venv", "--python", "3.10", "--clear", venv]);
        Run(uv, ["pip", "install", "--python", python, "-r", requirements]);
        File.WriteAllText(Path.Combine(venv, ".complete"), reqHash);
    }

    var userArgs = args.ToList();
    if (!userArgs.Contains("--qairt") && !userArgs.Contains("-h") && !userArgs.Contains("--help"))
    {
        string? qairt = Environment.GetEnvironmentVariable("QNN_SDK_ROOT") ?? Environment.GetEnvironmentVariable("QAIRT_SDK_ROOT");
        if (string.IsNullOrEmpty(qairt))
        {
            string standard = @"C:\Qualcomm\AIStack\QAIRT";
            qairt = Directory.Exists(standard) ? Directory.GetDirectories(standard).Order().LastOrDefault() : null;
        }

        if (string.IsNullOrEmpty(qairt))
        {
            Console.Error.WriteLine("Не найден QAIRT SDK. Скачайте QAIRT 2.40 (Qualcomm AI Engine Direct) и укажите --qairt <путь к qairt\\2.40.0.251030> или переменную QNN_SDK_ROOT.");
            return 2;
        }

        userArgs.AddRange(["--qairt", qairt]);
    }

    var psi = new ProcessStartInfo(python) { UseShellExecute = false, WorkingDirectory = Environment.CurrentDirectory };
    psi.ArgumentList.Add("-m");
    psi.ArgumentList.Add("qnnllm");
    foreach (string a in userArgs)
    {
        psi.ArgumentList.Add(a);
    }

    psi.Environment["PYTHONPATH"] = appDir;
    psi.Environment["PYTHONUTF8"] = "1";
    psi.Environment["PYTHONIOENCODING"] = "utf-8";
    using Process p = Process.Start(psi)!;
    await p.WaitForExitAsync();
    return p.ExitCode;
}
catch (Exception ex)
{
    Console.Error.WriteLine("Ошибка: " + ex.Message);
    return 1;
}

static string? FindOnPath(string exe) =>
    (Environment.GetEnvironmentVariable("PATH") ?? string.Empty).Split(Path.PathSeparator)
        .Select(d => Path.Combine(d.Trim(), exe)).FirstOrDefault(File.Exists);

static void Run(string exe, string[] arguments)
{
    var psi = new ProcessStartInfo(exe) { UseShellExecute = false };
    foreach (string a in arguments)
    {
        psi.ArgumentList.Add(a);
    }

    using Process p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
    {
        throw new InvalidOperationException($"{Path.GetFileName(exe)} {string.Join(' ', arguments)}: код {p.ExitCode}");
    }
}
