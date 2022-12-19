import lzma
from pathlib import Path
import unicodedata
from collections import defaultdict, namedtuple

from typing import Dict, List, Set, Optional, Generator, TextIO, Tuple
import csv

from tqdm import tqdm
import smart_open


def _handle_xz(file_obj, mode):
    return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)


smart_open.register_compressor(".xz", _handle_xz)


def deaccent(text: str) -> str:
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.

    Return input string with accents removed, as unicode.

    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'

    """

    res = []

    for c in text:
        if c in "їйЇЙ":
            res.append(c)
        else:
            norm = unicodedata.normalize("NFD", c)
            res.append(
                unicodedata.normalize(
                    "NFC",
                    "".join(ch for ch in norm if unicodedata.category(ch) != "Mn"),
                )
            )

    return "".join(res)


def preprocess_lemma(s: str) -> str:
    return deaccent(s).lower().strip(" 0123456789-")


def fix_nulls(fp_in: TextIO) -> Generator[str, None, None]:
    for line in fp_in:
        yield line.replace("\0", "")


Lemma = namedtuple("Lemma", ["lemma", "pos", "freq"])
Relation = namedtuple("Relation", ["word_left", "word_right", "relation"])

POS_ADJ: str = "ADJ"
POS_ADV: str = "ADV"
POS_NOUN: str = "NOUN"
POS_VERB: str = "VERB"

REL_ANTONYM: str = "antonym"
REL_RANDOM: str = "random"
REL_SYNONYM: str = "synonym"
REL_CO_HYPONYMS: str = "co-hyponyms"
REL_HYPERNYM_HYPONYM: str = "hypernym-hyponym"
REL_CO_INSTANCES: str = "co-instances"
REL_HYPERNYM_INSTANCE: str = "hypernym-instance"


class LemmaDictionary:
    def __init__(self, csv_source: Path, pos_whitelist: List[str]) -> None:
        # lemma -> Lemma objs
        self._lemmas: Dict[str, Set[Lemma]] = defaultdict(set)
        # pos -> lemma -> Lemma objs
        self._by_pos: Dict[str, Dict[str, Set[Lemma]]] = defaultdict(lambda: defaultdict(set))

        self._total: int = 0
        self._total_unfiltered: int = 0

        with smart_open.open(csv_source, "rt") as fp_in:
            r = csv.DictReader(fix_nulls(fp_in))

            for entry in tqdm(r):
                self._total_unfiltered += 1
                if entry["pos"] not in pos_whitelist:
                    continue

                word = preprocess_lemma(entry["lemma"])

                lemma = Lemma(lemma=word, pos=entry["pos"], freq=entry.get("freq_in_corpus", 0))
                self._lemmas[word].add(lemma)
                self._by_pos[entry["pos"]][word].add(lemma)

                self._total += 1

    def __str__(self) -> str:
        nl: str = "\n"
        tab: str = "\t"

        return f"""
        Total lemmas (filtered): {self._total}
        Total lemmas (w. ignored): {self._total_unfiltered}
        Unique lemmas: {len(self._lemmas)}
        POS tags:\t{len(self._by_pos)}
        Composition:\n{nl.join(f'{tab}{pos}: {len(lemmas)}' for pos, lemmas in self._by_pos.items() )}
        """


class RelationDictionary:
    def __init__(
        self, csv_source: Path, rel_whitelist: List[str], rel_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        self._by_rel: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._all_rels: Set[Relation] = set()

        self._rel_mapping: Dict = {}
        if rel_mapping is not None:
            self._rel_mapping = rel_mapping

        self._total_unfiltered: int = 0
        self._total: int = 0

        with smart_open.open(csv_source, "rt") as fp_in:
            r = csv.DictReader(fix_nulls(fp_in))

            for entry in tqdm(r):
                self._total_unfiltered += 1

                if entry["relation"] not in rel_whitelist:
                    continue

                resolved_rel: str = self._rel_mapping.get(entry["relation"], entry["relation"])

                self._by_rel[resolved_rel].add((entry["word_left"], entry["word_right"]))
                self._all_rels.add(
                    Relation(word_left=entry["word_left"], word_right=entry["word_right"], relation=resolved_rel)
                )
                self._total += 1

    def __str__(self) -> str:
        nl: str = "\n"
        tab: str = "\t"

        return f"""
        Total relations (filtered): {self._total}
        Total lemmas (w. ignored): {self._total_unfiltered}
        Unique lemmas: {len(self._all_rels)}
        Rel types:\t{len(self._by_rel)}
        Composition:\n{nl.join(f'{tab}{rel}: {len(lemmas)}' for rel, lemmas in self._by_rel.items() )}
        """


class RelationDataset:
    def __init__(
        self,
        pos_whitelist: List[str] = [POS_ADJ, POS_ADV, POS_NOUN, POS_VERB],
        rel_whitelist: List[str] = [
            REL_ANTONYM,
            REL_RANDOM,
            REL_SYNONYM,
            REL_CO_HYPONYMS,
            REL_HYPERNYM_HYPONYM,
            REL_CO_INSTANCES,
            REL_HYPERNYM_INSTANCE,
        ],
    ) -> None:
        # Here we store lemma dictionaries obtained from different sources.
        # Key of the dict is the handle of the dictionary
        self.lemma_dicts: Dict[str, LemmaDictionary] = {}
        self._pos_whitelist = pos_whitelist
        self._rel_whitelist = rel_whitelist
        self.rel_dicts: Dict[str, RelationDictionary] = {}

    def add_lemma_dict(self, dict_handle: str, csv_source: Path) -> None:
        self.lemma_dicts[dict_handle] = LemmaDictionary(csv_source, pos_whitelist=self._pos_whitelist)

    def add_rel_dict(self, dict_handle: str, csv_source: Path) -> None:
        self.rel_dicts[dict_handle] = RelationDictionary(csv_source, rel_whitelist=self._rel_whitelist)


if __name__ == "__main__":
    rd = RelationDataset()
    # rd.add_lemma_dict("ubertext_freq", Path("dictionaries/lemmas/ubertext_freq.csv.xz"))
    rd.add_lemma_dict("vesum", Path("dictionaries/lemmas/vesum.csv.xz"))
    print(rd.lemma_dicts["vesum"])

    rd.add_lemma_dict("ulif", Path("dictionaries/lemmas/ulif.csv.xz"))
    print(rd.lemma_dicts["ulif"])

    rd.add_rel_dict("ulif_synonyms", Path("dictionaries/relations/ulif_synonyms.csv.xz"))
    print(rd.rel_dicts["ulif_synonyms"])

    rd.add_rel_dict("wn_wikidata", Path("dictionaries/relations/wn_wikidata.csv.xz"))
    print(rd.rel_dicts["wn_wikidata"])
