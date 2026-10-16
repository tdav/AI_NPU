using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AiNpu.Services;

// BPE-токенизатор по tokenizer.json (HF tokenizers). Поддержаны две схемы, которых хватает для LLM onnx-community:
// 1) byte-level (GPT-2: LFM2, Qwen, Llama 3): [NFC] + pre_tokenizer Split(regex) + ByteLevel, байты -> bytes_to_unicode;
// 2) SentencePiece-подобная (Gemma, Llama 2, Mistral): normalizer Replace(" " -> "▁") / Prepend("▁"),
//    BPE по символам, неизвестные символы -> байтовые токены <0xNN> (byte_fallback).
// Как в HF, добавленные токены (<|im_start|>, <bos>, …) вырезаются из текста до нормализации.
public sealed class HfTokenizer
{
    private readonly Dictionary<string, int> vocab;
    private readonly string[] idToToken;
    private readonly Dictionary<(string, string), int> mergeRanks;
    private readonly Dictionary<string, int> addedTokens;
    private readonly HashSet<int> specialIds;
    private readonly Regex? addedRegex;
    private readonly bool byteLevel;
    private readonly bool byteFallback;
    private readonly Regex? splitRegex;
    private readonly bool splitMergedWithPrevious;
    private readonly string? splitString;
    private readonly List<(string From, string To)> normalizerReplace = [];
    private readonly string? normalizerPrepend;
    private readonly bool normalizerNfc;
    private readonly List<(string From, string To)> decoderReplace = [];
    private readonly char[] byteToChar = new char[256];
    private readonly Dictionary<char, byte> charToByte = [];
    private readonly Dictionary<string, int[]> cache = [];

    public HfTokenizer(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        JsonElement root = doc.RootElement;
        JsonElement model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
        {
            throw new NotSupportedException("поддерживается только BPE-токенизатор");
        }

        this.byteFallback = model.TryGetProperty("byte_fallback", out JsonElement bf) && bf.ValueKind == JsonValueKind.True;
        this.vocab = [];
        foreach (JsonProperty p in model.GetProperty("vocab").EnumerateObject())
        {
            this.vocab[p.Name] = p.Value.GetInt32();
        }

        this.mergeRanks = [];
        int rank = 0;
        foreach (JsonElement m in model.GetProperty("merges").EnumerateArray())
        {
            (string a, string b) pair;
            if (m.ValueKind == JsonValueKind.Array)
            {
                pair = (m[0].GetString()!, m[1].GetString()!);
            }
            else
            {
                string s = m.GetString()!;
                int space = s.IndexOf(' ', 1);
                pair = (s[..space], s[(space + 1)..]);
            }

            this.mergeRanks.TryAdd(pair, rank++);
        }

        this.addedTokens = [];
        this.specialIds = [];
        foreach (JsonElement t in root.GetProperty("added_tokens").EnumerateArray())
        {
            string content = t.GetProperty("content").GetString()!;
            int id = t.GetProperty("id").GetInt32();
            this.addedTokens[content] = id;
            if (t.GetProperty("special").GetBoolean())
            {
                this.specialIds.Add(id);
            }
        }

        // Нормализатор: Replace / Prepend, в том числе внутри Sequence.
        foreach (JsonElement n in Flatten(root.GetProperty("normalizer"), "normalizers"))
        {
            switch (n.GetProperty("type").GetString())
            {
                case "Replace":
                    this.normalizerReplace.Add((n.GetProperty("pattern").GetProperty("String").GetString()!, n.GetProperty("content").GetString()!));
                    break;
                case "Prepend":
                    this.normalizerPrepend = n.GetProperty("prepend").GetString();
                    break;
                case "NFC":
                    this.normalizerNfc = true;
                    break;
                default:
                    throw new NotSupportedException("normalizer " + n.GetProperty("type").GetString());
            }
        }

        // Предтокенизатор: Split (regex/строка) и ByteLevel.
        foreach (JsonElement p in Flatten(root.GetProperty("pre_tokenizer"), "pretokenizers"))
        {
            switch (p.GetProperty("type").GetString())
            {
                case "Split":
                    JsonElement pattern = p.GetProperty("pattern");
                    string behavior = p.GetProperty("behavior").GetString()!;
                    if (pattern.TryGetProperty("Regex", out JsonElement rx))
                    {
                        this.splitRegex = new Regex(rx.GetString()!, RegexOptions.Compiled);
                    }
                    else
                    {
                        this.splitString = pattern.GetProperty("String").GetString();
                        this.splitMergedWithPrevious = behavior == "MergedWithPrevious";
                    }

                    break;
                case "ByteLevel":
                    this.byteLevel = true;
                    if (p.TryGetProperty("use_regex", out JsonElement ur) && ur.ValueKind == JsonValueKind.True)
                    {
                        this.splitRegex ??= new Regex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled);
                    }

                    break;
                default:
                    throw new NotSupportedException("pre_tokenizer " + p.GetProperty("type").GetString());
            }
        }

        foreach (JsonElement d in Flatten(root.GetProperty("decoder"), "decoders"))
        {
            if (d.GetProperty("type").GetString() == "Replace")
            {
                this.decoderReplace.Add((d.GetProperty("pattern").GetProperty("String").GetString()!, d.GetProperty("content").GetString()!));
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

        if (this.addedTokens.Count > 0)
        {
            string alternation = string.Join("|", this.addedTokens.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape));
            this.addedRegex = new Regex(alternation, RegexOptions.Compiled);
        }

        // bytes_to_unicode из GPT-2: печатные байты остаются собой, остальные сдвигаются в 256+.
        int shift = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            char c = printable ? (char)b : (char)(256 + shift++);
            this.byteToChar[b] = c;
            this.charToByte[c] = (byte)b;
        }
    }

    public int VocabSize => this.idToToken.Length;

    public bool IsSpecial(int id) => this.specialIds.Contains(id);

    public int? TokenId(string token) =>
        this.addedTokens.TryGetValue(token, out int a) ? a : this.vocab.TryGetValue(token, out int v) ? v : null;

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        int pos = 0;
        bool first = true;
        if (this.addedRegex is not null)
        {
            foreach (Match m in this.addedRegex.Matches(text))
            {
                this.EncodeOrdinary(text[pos..m.Index], ids, first);
                first = false;
                ids.Add(this.addedTokens[m.Value]);
                pos = m.Index + m.Length;
            }
        }

        this.EncodeOrdinary(text[pos..], ids, first);
        return ids;
    }

    // Байты токенов: потоковый вывод собирает UTF-8 символ, разделённый между токенами.
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
            }
            else if (this.byteLevel)
            {
                foreach (char c in token)
                {
                    if (this.charToByte.TryGetValue(c, out byte b))
                    {
                        bytes.Add(b);
                    }
                }
            }
            else if (token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>')
            {
                bytes.Add(Convert.ToByte(token.Substring(3, 2), 16));
            }
            else
            {
                foreach (var (from, to) in this.decoderReplace)
                {
                    token = token.Replace(from, to, StringComparison.Ordinal);
                }

                bytes.AddRange(Encoding.UTF8.GetBytes(token));
            }
        }

        return [.. bytes];
    }

    public string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString(this.DecodeBytes(ids));

    private void EncodeOrdinary(string text, List<int> ids, bool atStart)
    {
        if (text.Length == 0)
        {
            return;
        }

        if (this.normalizerNfc)
        {
            text = text.Normalize(NormalizationForm.FormC);
        }

        foreach (var (from, to) in this.normalizerReplace)
        {
            text = text.Replace(from, to, StringComparison.Ordinal);
        }

        if (this.normalizerPrepend is not null && atStart)
        {
            text = this.normalizerPrepend + text;
        }

        foreach (string piece in this.Split(text))
        {
            if (this.cache.TryGetValue(piece, out int[]? cached))
            {
                ids.AddRange(cached);
                continue;
            }

            var symbols = new List<string>();
            if (this.byteLevel)
            {
                foreach (byte b in Encoding.UTF8.GetBytes(piece))
                {
                    symbols.Add(this.byteToChar[b].ToString());
                }
            }
            else
            {
                // по символам Юникода (суррогатная пара — один символ)
                for (int i = 0; i < piece.Length; i += char.IsSurrogatePair(piece, i) ? 2 : 1)
                {
                    symbols.Add(piece.Substring(i, char.IsSurrogatePair(piece, i) ? 2 : 1));
                }
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

            var pieceIds = new List<int>(symbols.Count);
            foreach (string s in symbols)
            {
                if (this.vocab.TryGetValue(s, out int id))
                {
                    pieceIds.Add(id);
                }
                else if (this.byteFallback)
                {
                    foreach (byte b in Encoding.UTF8.GetBytes(s))
                    {
                        pieceIds.Add(this.vocab[$"<0x{b:X2}>"]);
                    }
                }
                else
                {
                    throw new InvalidDataException($"токена «{s}» нет в словаре");
                }
            }

            if (this.cache.Count < 50_000 && piece.Length < 64)
            {
                this.cache[piece] = [.. pieceIds];
            }

            ids.AddRange(pieceIds);
        }
    }

    private IEnumerable<string> Split(string text)
    {
        if (this.splitRegex is not null)
        {
            foreach (Match m in this.splitRegex.Matches(text))
            {
                yield return m.Value;
            }
        }
        else if (this.splitString is not null && this.splitMergedWithPrevious)
        {
            int start = 0;
            for (int i = text.IndexOf(this.splitString, StringComparison.Ordinal); i >= 0; i = text.IndexOf(this.splitString, start, StringComparison.Ordinal))
            {
                yield return text[start..(i + this.splitString.Length)];
                start = i + this.splitString.Length;
            }

            if (start < text.Length)
            {
                yield return text[start..];
            }
        }
        else
        {
            yield return text;
        }
    }

    private static IEnumerable<JsonElement> Flatten(JsonElement element, string listKey)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            yield break;
        }

        if (element.TryGetProperty(listKey, out JsonElement list))
        {
            foreach (JsonElement e in list.EnumerateArray())
            {
                foreach (JsonElement inner in Flatten(e, listKey))
                {
                    yield return inner;
                }
            }
        }
        else
        {
            yield return element;
        }
    }
}
