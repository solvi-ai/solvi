// Byte-level BPE tokenizer for ModernBERT (tokenizer.json), written to return CHARACTER OFFSETS for every token.
// transformers.js does not expose offset mappings, and the extractor needs them to cite exact spans, so this is a small,
// self-contained reimplementation of what Hugging Face `tokenizers` does for this tokenizer:
//   NFC normalizer -> added-token split (leftmost-longest) -> ByteLevel pre-tokenizer (GPT-2 regex) -> BPE (merge ranks).
// Offsets follow `tokenizers` exactly: a token covers the characters its bytes came from (a token that holds only part of a
// multi-byte character still covers that whole character). Two offset systems are returned:
//   offsets   - Python string indices (Unicode code points): what solvi's Quote(start, end) and the Python extractor use;
//   offsets16 - JavaScript string indices (UTF-16 code units): for highlighting in the page.
// The input must already be NFC-normalized (the app normalizes every document once, on input).

const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const map = new Array(256);
  bs.forEach((b, i) => { map[b] = String.fromCharCode(cs[i]); });
  return map;
}

export class ByteLevelBPE {
  constructor(tokenizerJson) {
    const t = tokenizerJson;
    const m = t.model;
    if (m.type !== "BPE") throw new Error("expected a BPE tokenizer");
    this.vocab = new Map(Object.entries(m.vocab));
    this.ranks = new Map();
    m.merges.forEach((mg, i) => {
      const [a, b] = Array.isArray(mg) ? mg : mg.split(" ");
      this.ranks.set(a + "\u0000" + b, i);
    });
    this.byteMap = bytesToUnicode();
    this.added = new Map();                         // first char -> [{content, id}] sorted longest first
    for (const a of t.added_tokens || []) {
      if (!a.content) continue;
      const k = a.content[0];
      if (!this.added.has(k)) this.added.set(k, []);
      this.added.get(k).push({ content: a.content, id: a.id });
      this.vocab.set(a.content, a.id);
    }
    for (const lst of this.added.values()) lst.sort((x, y) => y.content.length - x.content.length);
    const special = (name) => (t.added_tokens || []).find((a) => a.content === name)?.id;
    this.cls = special("[CLS]"); this.sep = special("[SEP]"); this.pad = special("[PAD]");
    this.cache = new Map();
    this.enc = new TextEncoder();
  }

  bpe(word) {                                        // word: string of byte-level chars -> array of token strings
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + "\u0000" + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      parts = parts.slice(0, at).concat([parts[at] + parts[at + 1]], parts.slice(at + 2));
    }
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, parts);
    return parts;
  }

  // text (NFC) -> {ids, offsets: [[cpStart, cpEnd]], offsets16: [[u16Start, u16End]]}
  encode(text) {
    // per-UTF-16-index maps: byte offset of the code unit's character, and the code point index
    const n16 = text.length;
    const u16ToByte = new Int32Array(n16 + 1);
    const u16ToCp = new Int32Array(n16 + 1);
    const byteCp = [], byteU16s = [], byteU16e = [];
    let b = 0, cp = 0;
    for (let i = 0; i < n16;) {
      const c = text.codePointAt(i);
      const w = c > 0xffff ? 2 : 1;
      const nb = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
      for (let k = 0; k < w; k++) { u16ToByte[i + k] = b; u16ToCp[i + k] = cp; }
      for (let k = 0; k < nb; k++) { byteCp.push(cp); byteU16s.push(i); byteU16e.push(i + w); }
      b += nb; cp += 1; i += w;
    }
    u16ToByte[n16] = b; u16ToCp[n16] = cp;
    const ids = [], offsets = [], offsets16 = [];
    const pushPiece = (s16, e16) => {               // one pre-token -> BPE tokens with offsets
      const piece = text.slice(s16, e16);
      const bytes = this.enc.encode(piece);
      let word = "";
      for (const x of bytes) word += this.byteMap[x];
      let pos = u16ToByte[s16];
      for (const tok of this.bpe(word)) {
        const id = this.vocab.get(tok);
        if (id === undefined) throw new Error("token not in vocab: " + tok);
        const bs = pos, be = pos + tok.length;      // one byte-level char per byte
        ids.push(id);
        offsets.push([byteCp[bs], byteCp[be - 1] + 1]);
        offsets16.push([byteU16s[bs], byteU16e[be - 1]]);
        pos = be;
      }
    };
    const pretok = (s16, e16) => {
      if (e16 <= s16) return;
      const seg = text.slice(s16, e16);
      GPT2_SPLIT.lastIndex = 0;
      let m;
      while ((m = GPT2_SPLIT.exec(seg)) !== null) {
        if (m[0].length === 0) { GPT2_SPLIT.lastIndex++; continue; }
        pushPiece(s16 + m.index, s16 + m.index + m[0].length);
      }
    };
    // added tokens (runs of spaces, special tokens): leftmost-longest match, the rest goes through the pre-tokenizer
    let start = 0, i = 0;
    while (i < n16) {
      const cands = this.added.get(text[i]);
      let match = null;
      if (cands) for (const a of cands) if (text.startsWith(a.content, i)) { match = a; break; }
      if (match) {
        pretok(start, i);
        ids.push(match.id);
        offsets.push([u16ToCp[i], u16ToCp[i + match.content.length]]);
        offsets16.push([i, i + match.content.length]);
        i += match.content.length;
        start = i;
      } else i++;
    }
    pretok(start, n16);
    return { ids, offsets, offsets16 };
  }
}
