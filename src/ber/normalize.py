"""Deterministic text normalization (polars expressions, vectorized).

Several representations are kept per field:
  name / addr          original string
  *_tok                tokens: lowercase, Latin accents folded, punctuation -> space,
                       Devanagari tokens transliterated, field-specific abbreviations expanded
  *_norm               " ".join(*_tok)
All Indic script blocks (U+0900-U+0DFF: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
Telugu, Kannada, Malayalam) are preserved by cleaning (their vowel signs are Unicode marks, so only
Latin combining marks U+0300-U+036F are stripped).
"""
import re

import polars as pl

# Business-name abbreviations (language knowledge, not entity data).
NAME_ABBREV = {
    "ltd": "limited", "pvt": "private", "corp": "corporation", "inc": "incorporated", "co": "company",
    "cos": "companies", "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "assoc": "associates", "bros": "brothers", "mgmt": "management", "grp": "group", "hldgs": "holdings",
    "ent": "enterprises",
}
# Address abbreviations: English street types/directions, Indian and French address shorthand.
ADDR_ABBREV = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square", "ste": "suite", "apt": "apartment",
    "fl": "floor", "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount",
    "nr": "near", "opp": "opposite", "hno": "house", "h": "house", "no": "number", "dist": "district",
    "tq": "taluk", "tal": "taluk", "vill": "village", "po": "post", "ps": "police",
    "kh": "khasra", "sec": "sector", "ph": "phase", "gf": "ground", "flr": "floor",
    "r": "rue", "che": "chemin", "chem": "chemin", "imp": "impasse", "rte": "route",
    "fbg": "faubourg", "qu": "quai", "pte": "porte", "crs": "cours",
}
DEVA = r"[\u0900-\u0dff]"  # any Indic script block
_DEVANAGARI = r"[\u0900-\u097f]"

# Generic Devanagari -> Latin character rules, used only for tokens absent from the learned dictionary.
_V = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
      "ऑ": "o", "ऋ": "ri"}
_M = {"ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o",
      "ृ": "ri", "ं": "n", "ँ": "n", "ः": "h", "्": "", "़": ""}
_C = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
              "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h".split()))


def romanize(word: str) -> str:
    """Rule-based romanization of a Devanagari token (inherent 'a' after bare consonants)."""
    out = []
    for j, ch in enumerate(word):
        if ch in _C:
            out.append(_C[ch])
            nxt = word[j + 1] if j + 1 < len(word) else ""
            if nxt not in _M:
                out.append("a")
        else:
            out.append(_M.get(ch, _V.get(ch, ch)))
    s = "".join(out)
    return s[:-1] if s.endswith("a") and len(s) > 3 else s


def clean(expr: pl.Expr) -> pl.Expr:
    return (
        expr.fill_null("")
        .str.to_lowercase()
        .str.normalize("NFKD")
        .str.replace_all(r"[\u0300-\u036f]", "")
        .str.replace_all(r"<null>|\bnull\b|\bn/a\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^a-z0-9\u0900-\u0dff]+", " ")
        .str.strip_chars()
    )


def raw_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Cleaned tokens before transliteration/abbreviation (used to learn the transliteration map)."""
    split = lambda c: clean(pl.col(c)).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return df.with_columns(split("name").alias("name_tok"), split("addr").alias("addr_tok"))


def _deva_map(df: pl.DataFrame, translit: dict) -> dict:
    toks = pl.concat([df["name_tok"].explode(), df["addr_tok"].explode()]).drop_nulls().unique()
    deva = toks.filter(toks.str.contains(DEVA)).to_list()
    # learned dictionary first; generic romanization only exists for Devanagari; other scripts stay raw
    return {t: translit.get(t) or (romanize(t) if re.search(_DEVANAGARI, t) else t) for t in deva}


def add_normalized(df: pl.DataFrame, translit: dict | None = None) -> pl.DataFrame:
    """Add token lists and normalized strings for name and address (df must come from raw_tokens)."""
    m = _deva_map(df, translit or {})
    df = df.with_columns(
        pl.col("name_tok").list.eval(pl.element().replace(m).replace(NAME_ABBREV)),
        pl.col("addr_tok").list.eval(pl.element().replace(m).replace(ADDR_ABBREV)),
        clean(pl.col("country")).alias("country_norm"),
        pl.col("name").str.contains(DEVA).alias("name_deva"),
    )
    return df.with_columns(pl.col("name_tok").list.join(" ").alias("name_norm"),
                           pl.col("addr_tok").list.join(" ").alias("addr_norm"))
