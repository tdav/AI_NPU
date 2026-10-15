using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AiNpu.Services;

// Токенизатор LFM2 по tokenizer.json (HF tokenizers): byte-level BPE как у GPT-2, без нормализатора.
// Порядок как в HF: сначала из текста вырезаются добавленные токены (<|im_start|> и т. п.), остальное режется
// регуляркой pre_tokenizer, каждый кусок переводится в «байтовые» символы (bytes_to_unicode) и сливается по merges.
// BOS (<|startoftext|>) сам не добавляется — промпт собирает ChatSession.
public sealed class Lfm2Tokenizer
{
    public const int BosId = 1;
    public const int ImEndId = 7;

    // Регулярка pre_tokenizer из tokenizer.json LFM2 (Split, behavior Isolated).
    private static readonly Regex pieceRegex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    private readonly Dictionary<string, int> vocab;
    private readonly string[] idToToken;
    private readonly Dictionary<(string, string), int> mergeRanks;
    private readonly Dictionary<string, int> addedTokens;
    private readonly HashSet<int> specialIds;
    private readonly Regex? addedRegex;
    private readonly char[] byteToChar = new char[256];
    private readonly Dictionary<char, byte> charToByte = [];
    private readonly Dictionary<string, int[]> cache = [];

    public Lfm2Tokenizer(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        JsonElement model = doc.RootElement.GetProperty("model");

        this.vocab = [];
        foreach (JsonProperty p in model.GetProperty("vocab").EnumerateObject())
        {
            this.vocab[p.Name] = p.Value.GetInt32();
        }

        this.mergeRanks = [];
        int rank = 0;
        foreach (JsonElement m in model.GetProperty("merges").EnumerateArray())
        {
            // merges бывают массивом пар (новый формат) или строкой "a b" (старый)
            (string a, string b) pair = m.ValueKind == JsonValueKind.Array
                ? (m[0].GetString()!, m[1].GetString()!)
                : (m.GetString()!.Split(' ')[0], m.GetString()!.Split(' ')[1]);
            this.mergeRanks.TryAdd(pair, rank++);
        }

        this.addedTokens = [];
        this.specialIds = [];
        foreach (JsonElement t in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            string content = t.GetProperty("content").GetString()!;
            int id = t.GetProperty("id").GetInt32();
            this.addedTokens[content] = id;
            if (t.GetProperty("special").GetBoolean())
            {
                this.specialIds.Add(id);
            }
        }

        int maxId = Math.Max(this.vocab.Values.Max(), this.addedTokens.Count > 0 ? this.addedTokens.Values.Max() : 0);
        this.idToToken = new string[maxId + 1];
        foreach (var (token, id) in this.vocab)
        {
            this.idToToken[id] = token;
        }

        foreach (var (token, id) in this.addedTokens)
        {
            this.idToToken[id] = token;
        }

        // Самые длинные добавленные токены — первыми, чтобы <|im_start|> не перехватывался более коротким.
        if (this.addedTokens.Count > 0)
        {
            string alternation = string.Join("|", this.addedTokens.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape));
            this.addedRegex = new Regex(alternation, RegexOptions.Compiled);
        }

        // bytes_to_unicode из GPT-2: печатные байты остаются собой, остальные сдвигаются в 256+.
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            char c = printable ? (char)b : (char)(256 + n++);
            this.byteToChar[b] = c;
            this.charToByte[c] = (byte)b;
        }
    }

    public int VocabSize => this.idToToken.Length;

    public bool IsSpecial(int id) => this.specialIds.Contains(id);

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        int pos = 0;
        if (this.addedRegex is not null)
        {
            foreach (Match m in this.addedRegex.Matches(text))
            {
                this.EncodeOrdinary(text[pos..m.Index], ids);
                ids.Add(this.addedTokens[m.Value]);
                pos = m.Index + m.Length;
            }
        }

        this.EncodeOrdinary(text[pos..], ids);
        return ids;
    }

    // Байты токенов (для потокового вывода: UTF-8 символ может делиться между токенами).
    // Специальные токены пропускаются, обычные добавленные — выводятся текстом.
    public byte[] DecodeBytes(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (id < 0 || id >= this.idToToken.Length || this.idToToken[id] is null || this.specialIds.Contains(id))
            {
                continue;
            }

            string token = this.idToToken[id];
            if (this.addedTokens.ContainsKey(token))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(token));
                continue;
            }

            foreach (char c in token)
            {
                if (this.charToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
            }
        }

        return [.. bytes];
    }

    public string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString(this.DecodeBytes(ids));

    private void EncodeOrdinary(string text, List<int> ids)
    {
        if (text.Length == 0)
        {
            return;
        }

        foreach (Match piece in pieceRegex.Matches(text))
        {
            if (this.cache.TryGetValue(piece.Value, out int[]? cached))
            {
                ids.AddRange(cached);
                continue;
            }

            byte[] utf8 = Encoding.UTF8.GetBytes(piece.Value);
            var symbols = new List<string>(utf8.Length);
            foreach (byte b in utf8)
            {
                symbols.Add(this.byteToChar[b].ToString());
            }

            // BPE: каждый раз сливается пара с наименьшим рангом (все её вхождения слева направо).
            while (symbols.Count > 1)
            {
                int bestRank = int.MaxValue;
                (string, string) best = default;
                for (int i = 0; i + 1 < symbols.Count; i++)
                {
                    if (this.mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                    {
                        bestRank = r;
                        best = (symbols[i], symbols[i + 1]);
                    }
                }

                if (bestRank == int.MaxValue)
                {
                    break;
                }

                var merged = new List<string>(symbols.Count);
                for (int i = 0; i < symbols.Count; i++)
                {
                    if (i + 1 < symbols.Count && symbols[i] == best.Item1 && symbols[i + 1] == best.Item2)
                    {
                        merged.Add(best.Item1 + best.Item2);
                        i++;
                    }
                    else
                    {
                        merged.Add(symbols[i]);
                    }
                }

                symbols = merged;
            }

            var pieceIds = new int[symbols.Count];
            for (int i = 0; i < symbols.Count; i++)
            {
                // В byte-level BPE каждый одиночный байт есть в словаре; отсутствие токена — испорченный tokenizer.json.
                pieceIds[i] = this.vocab.TryGetValue(symbols[i], out int id)
                    ? id
                    : throw new InvalidDataException($"токена «{symbols[i]}» нет в словаре");
            }

            if (this.cache.Count < 50_000)
            {
                this.cache[piece.Value] = pieceIds;
            }

            ids.AddRange(pieceIds);
        }
    }
}
