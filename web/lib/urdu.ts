// Roman Urdu -> tiny-voca token ids. A line-by-line port of urdu/frontend.py
// (roman_to_ipa, _ipa_word, pieces) and tiny_voca.text.tokenize. Keep the two in
// sync: scripts/parity.py writes the Python ids, scripts/parity.mjs checks these.

const N = "̃";
export const PAD = "<pad>", BOS = "<bos>", EOS = "<eos>", SP = "<sp>", UNK = "<unk>";
const SPECIALS = [PAD, BOS, EOS, SP, UNK];
const PUNCTUATION = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"];

const IGNORED = new Set("ˈˌ.|‿͜͡");
const APPROX: Record<string, string> = { "ɹ": "ɹ", "ɡ": "ɡ", g: "ɡ", "ə": "ə", "ʌ": "ʌ", "ɐ": "ɐ", "ɑ": "ɑː", "ɔ": "ɔ", "ʊ": "ʊ", "ɪ": "ɪ", i: "i", "ɛ": "ɛ" };
const NASAL_BASE: Record<string, string> = { "ə": "ɐ", "ʌ": "ɐ", "ɐ": "ɐ", "ɪ": "i", "ʊ": "u", "ɑ": "ɑ" };

const LEXICON: Record<string, string> = {
  hai: "hɛ", hay: "hɛ", he: "hɛ", hain: "hẽ", hein: "hẽ", hen: "hẽ",
  mein: "mẽ", main: "mẽ", men: "mẽ", me: "mẽ", nahi: "nahiː", nahin: "nahĩ", nahee: "nahiː",
  aur: "ɔr", or: "ɔr", ka: "kaː", ki: "ki", ke: "keː", ko: "koː", se: "seː", ne: "neː",
  par: "par", pe: "peː", yeh: "jeː", ye: "jeː", woh: "ʋoː", wo: "ʋoː", to: "toː", bhi: "bʰiː",
  hi: "hiː", is: "ɪs", us: "ʊs", in: "ɪn", un: "ʊn", kya: "kjaː", kyun: "kjũ", kyon: "kjũ",
  kuch: "kʊcʰ", kuchh: "kʊcʰ", tha: "tʰaː", thi: "tʰiː", the: "tʰeː", thay: "tʰeː", ho: "hoː",
  hoon: "hũ", hun: "hũ", hum: "ham", tum: "tʊm", aap: "aːp", ap: "aːp", ek: "eːk", do: "doː",
  jo: "ɟoː", jab: "ɟab", tak: "tak", allah: "allaːh", mera: "meːraː", meri: "meːriː",
  mere: "meːreː", apna: "apnaː", apni: "apniː", apne: "apneː", sab: "sab", ab: "ab",
};
// Same as _LEXICON_EXTRA in frontend.py: fixes for Roman spellings, safe to extend.
export const LEXICON_EXTRA: Record<string, string> = {
  chai: "caːeː", chaye: "caːeː", chae: "caːeː", chaaye: "caːeː", chaey: "caːeː",
};
const RULES: [string, string][] = [
  ["chh", "cʰ"], ["Th", "ʈʰ"], ["Dh", "ɖʰ"], ["Rh", "ɽʰ"],
  ["kh", "kʰ"], ["gh", "ɡʰ"], ["ch", "c"], ["sh", "ʃ"], ["th", "tʰ"], ["dh", "dʰ"], ["ph", "pʰ"],
  ["bh", "bʰ"], ["jh", "ɟʰ"], ["zh", "ʒ"], ["ck", "k"],
  ["aa", "aː"], ["ee", "iː"], ["ii", "iː"], ["oo", "uː"], ["ai", "ɛ"], ["ay", "eː"], ["ey", "eː"],
  ["ei", "ɛ"], ["au", "ɔ"], ["ou", "ɔ"], ["ao", "aːoː"],
  ["T", "ʈ"], ["D", "ɖ"], ["R", "ɽ"],
];
const SINGLE: Record<string, string> = { a: "a", e: "eː", i: "ɪ", o: "oː", u: "ʊ", g: "ɡ", j: "ɟ", y: "j", w: "ʋ", v: "ʋ", c: "k", x: "ks", r: "r" };
const VOWEL_LETTERS = new Set("aeiou");
const isAlpha = (c: string) => /\p{L}/u.test(c);
const has = (o: Record<string, string>, k: string) => Object.prototype.hasOwnProperty.call(o, k);

export function romanToIpa(word: string): string {
  const lower = word.toLowerCase();
  if (has(LEXICON_EXTRA, lower) && !/[TDR]/.test(word.slice(1))) return LEXICON_EXTRA[lower];
  if (has(LEXICON, lower) && !/[TDR]/.test(word)) return LEXICON[lower];
  if (word.length > 1 && /\p{Lu}/u.test(word) && word === word.toUpperCase()) word = word.toLowerCase();
  word = word.slice(0, 1).toLowerCase() + word.slice(1);
  if (has(LEXICON, word.toLowerCase()) && !/[TDR]/.test(word)) return LEXICON[word.toLowerCase()];
  word = [...word.replaceAll("'", "")].map((c) => ("TDR".includes(c) ? c : c.toLowerCase())).join("");
  const out: string[] = [];
  let i = 0, last = "";
  while (i < word.length) {
    const rule = RULES.find(([letters]) => word.startsWith(letters, i));
    if (rule) {
      out.push(rule[1]);
      i += rule[0].length;
      last = "";
      continue;
    }
    const ch = word[i], final = i === word.length - 1;
    let ipa: string;
    if (ch === "a" && final && word.length > 1) ipa = "aː";
    else if (ch === "i" && final && word.length > 1) ipa = "iː";
    else ipa = has(SINGLE, ch) ? SINGLE[ch] : isAlpha(ch) ? ch : "";
    if (ipa && !(!VOWEL_LETTERS.has(ch) && ch === last)) out.push(ipa); // "achha": a doubled consonant once
    last = ch;
    i += 1;
  }
  return out.join("");
}


// ---- numbers -> spoken Urdu (port of expand_numbers in urdu/frontend.py) ---------------------
const W = ("sifar ek do teen chaar paanch chhe saat aath nau das " +
  "gyarah barah terah chaudah pandrah solah satrah atharah unnees " +
  "bees ikkees baaees teyees chaubees pachchees chhabbees sattaees atthaees untees " +
  "tees iktees battees taitees chauntees paintees chhattees saintees artees untaalees " +
  "chaalees iktaalees bayaalees taintaalees chauwaalees paintaalees chhiyaalees saintaalees artaalees unchaas " +
  "pachaas ikyaavan baavan tirpan chauvan pachpan chhappan sattaavan atthaavan unsath " +
  "saath iksath baasath tirsath chausath painsath chhiyaasath sarsath arsath unhattar " +
  "sattar ikhattar bahattar tihattar chauhattar pachhattar chhihattar satattar athattar unaasi " +
  "assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sataasi athaasi navaasi " +
  "nabbe ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve atthaanve ninnaanve").split(" ");
const SCALES: [number, string][] = [[1e9, "arab"], [1e7, "karoR"], [1e5, "lakh"], [1000, "hazaar"], [100, "sau"]];
const D = "[0-9\\u0660-\\u0669\\u06F0-\\u06F9\\u0966-\\u096F]";
const NUM = new RegExp(`${D}{1,3}(?:,${D}{3})+(?:\\.${D}+)?|${D}{1,2}(?:,${D}{2})+,${D}{3}(?:\\.${D}+)?|${D}+(?:\\.${D}+)?`, "g");

export function numberWords(n: number): string {
  if (n < 100) return W[n];
  for (const [size, name] of SCALES) {
    if (n >= size) {
      const q = Math.floor(n / size), r = n % size;
      return numberWords(q) + " " + name + (r ? " " + numberWords(r) : "");
    }
  }
  return "";
}

function digit(c: string): string {
  const o = c.charCodeAt(0);
  for (const zero of [0x30, 0x660, 0x6f0, 0x966]) if (o >= zero && o < zero + 10) return String(o - zero);
  return c;
}

export function expandNumbers(text: string): string {
  return text.replace(NUM, (m) => {
    const s = [...m].map(digit).join("");
    const [wholeRaw, frac] = s.split(".");
    const whole = wholeRaw.replaceAll(",", "");
    let words = (whole.length > 1 && whole[0] === "0") || whole.length > 9
      ? [...whole].map((d) => W[+d]).join(" ")
      : numberWords(Number(whole));
    if (frac) words += " poaint " + [...frac].map((d) => W[+d]).join(" ");
    return " " + words + " ";
  });
}

export class Frontend {
  ids: Map<string, number>;
  phones: Set<string>;

  constructor(symbols: string[]) {
    this.ids = new Map(symbols.map((s, i) => [s, i]));
    this.phones = new Set(symbols.filter((s) => !SPECIALS.includes(s) && !PUNCTUATION.includes(s)));
  }

  private resolve(base: string, long: boolean, aspirated: boolean, nasal: boolean): string[] {
    const L = long ? "ː" : "", H = aspirated ? "ʰ" : "";
    if (base === "g") base = "ɡ";
    const candidates: string[] = [];
    if (nasal) for (const b of [base, NASAL_BASE[base] ?? base]) candidates.push(b + N + L, b + N);
    candidates.push(base + H + L, base + H);
    if (!H && !L) candidates.push(APPROX[base] ?? base);
    if (long) candidates.push((APPROX[base] ?? base) + "ː", base === "ɑ" ? "ɑː" : "");
    for (const c of candidates) if (this.phones.has(c)) return [c];
    if (aspirated) return [...this.resolve(base, long, false, nasal), "h"];
    return [base];
  }

  ipaWord(word: string): string[] {
    const units: [string, boolean, boolean, boolean][] = [];
    for (const char of word.normalize("NFD")) {
      if (IGNORED.has(char)) continue;
      const lastUnit = units[units.length - 1];
      if ("ː:".includes(char) && lastUnit) lastUnit[1] = true;
      else if (char === "ʰ" && lastUnit) lastUnit[2] = true;
      else if (char === N && lastUnit) lastUnit[3] = true;
      else if (/\p{M}/u.test(char)) continue;
      else units.push([char, false, false, false]);
    }
    return units.flatMap(([b, l, a, n]) => this.resolve(b, l, a, n));
  }

  // [isPunctuation, value]: value is a mark, or a list of words (each a list of phones)
  pieces(text: string): [boolean, string | string[][]][] {
    if (text.startsWith("ipa:")) {
      const words = text.slice(4).split(/\s+/).filter(Boolean).map((w) => this.ipaWord(w)).filter((w) => w.length);
      return [[false, words]];
    }
    const result: [boolean, string | string[][]][] = [];
    let words: string[][] = [];
    for (const token of expandNumbers(normalizeText(text)).replaceAll("-", " ").match(/[A-Za-z']+|[,.?!;:"()—…]/g) ?? []) {
      if (isAlpha(token[0]) || token[0] === "'") {
        const phones = this.ipaWord(romanToIpa(token));
        if (phones.length) words.push(phones);
      } else {
        if (words.length) result.push([false, words]);
        words = [];
        result.push([true, token]);
      }
    }
    if (words.length) result.push([false, words]);
    return result;
  }

  private canonical(phone: string): string {
    if (this.phones.has(phone)) return phone;
    const stripped = [...phone.normalize("NFD")].filter((c) => !/\p{M}/u.test(c)).join("");
    for (const c of [stripped, stripped.replaceAll("ː", ""), stripped + "ː"])
      if (c !== phone && this.phones.has(c)) return c;
    return UNK;
  }

  // tiny_voca.text.tokenize (stress is always 0 for Urdu)
  phonemize(text: string): { symbols: string[]; ids: number[]; stress: number[] } {
    const symbols = [BOS];
    for (const [punctuation, value] of this.pieces(text)) {
      if (punctuation) {
        if (symbols[symbols.length - 1] !== value) symbols.push(value as string);
        continue;
      }
      for (const word of value as string[][]) {
        const last = symbols[symbols.length - 1];
        if (this.phones.has(last) || last === UNK) symbols.push(SP);
        for (const phone of word) symbols.push(this.canonical(phone));
      }
    }
    symbols.push(EOS);
    return { symbols, ids: symbols.map((s) => this.ids.get(s)!), stress: symbols.map(() => 0) };
  }
}

const QUOTES: Record<string, string> = { "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
  "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"', "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")" };

export function normalizeText(text: string): string {
  text = [...text.normalize("NFKC")].map((c) => QUOTES[c] ?? c).join("");
  text = text.replace(/\.{3,}/g, "…").replace(/-{2,}|\s-\s/g, "—");
  return text.replace(/\s+/g, " ").trim();
}
