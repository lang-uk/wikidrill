import re
import lzma
from pathlib import Path
import unicodedata
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Optional, Generator, TextIO, Tuple, Callable, Union, Iterable
import csv
from functools import reduce

from tqdm import tqdm
import smart_open
import pandas as pd


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


def normalize_apostrophes(s: str) -> str:
    # Kudos to Andriy Rysin for the regex
    s = re.sub(
        r"(?iu)([бвгґдзкмнпрстфхш])[\"\u201D\u201F\u0022\u2018\u2032\u0313\u0384\u0092´`?*]([єїюя])", r"\1'\2", s
    )
    # TODO: verify with Andriy, wtf
    # s = re.sub(r"(?iu)[´`]([аеєиіїоуюя])", '\u0301\\1', s)
    return s


def preprocess_lemma(s: str) -> str:
    return deaccent(normalize_apostrophes(s)).lower().strip(" 0123456789-")


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
REL_HYPONYM_HYPERNYM: str = "hyponym-hypernym"
REL_CO_INSTANCES: str = "co-instances"
REL_HYPERNYM_INSTANCE: str = "hypernym-instance"
REL_INSTANCE_HYPERNYM: str = "instance-hypernym"

INVERSE_RELATIONS: List[set[str]] = [
    {REL_HYPERNYM_INSTANCE, REL_INSTANCE_HYPERNYM},
    {REL_HYPERNYM_HYPONYM, REL_HYPONYM_HYPERNYM},
]

ALL_RELATIONS: set[str] = {
    REL_ANTONYM,
    REL_RANDOM,
    REL_SYNONYM,
    REL_CO_HYPONYMS,
    REL_HYPERNYM_HYPONYM,
    REL_HYPONYM_HYPERNYM,
    REL_CO_INSTANCES,
    REL_HYPERNYM_INSTANCE,
    REL_INSTANCE_HYPERNYM,
}

ASSYMETRIC_RELATIONS: set[str] = reduce(set.union, INVERSE_RELATIONS)
SYMMETRIC_RELATIONS: set[str] = ALL_RELATIONS - ASSYMETRIC_RELATIONS

FLIPPED_MAPPING = {
    rel: (rel if rel in SYMMETRIC_RELATIONS else "-".join(rel.split("-")[::-1])) for rel in ALL_RELATIONS
}


class LemmaDictionary:
    def __init__(self, source: Union[Path, Iterable], pos_whitelist: List[str]) -> None:
        # lemma -> Lemma objs
        self._lemmas: Dict[str, Set[Lemma]] = defaultdict(set)
        # pos -> lemma -> Lemma objs
        self._by_pos: Dict[str, Dict[str, Set[Lemma]]] = defaultdict(lambda: defaultdict(set))
        # lemma -> combined freqs over the all poses
        self._freqs: Dict[str, float] = {}

        self._total: int = 0
        self._total_unfiltered: int = 0

        if isinstance(source, Path):
            fp_in = smart_open.open(source, "rt")
            dataset: Union[csv.DictReader, Iterable] = csv.DictReader(fix_nulls(fp_in))
        else:
            dataset = source

        for entry in tqdm(dataset):
            self._total_unfiltered += 1

            if isinstance(entry, Lemma):
                entry = entry._asdict()

            if entry["pos"] not in pos_whitelist:
                continue

            word = preprocess_lemma(entry["lemma"])

            lemma = Lemma(lemma=word, pos=entry["pos"], freq=float(entry.get("freq_in_corpus", 0)))
            self._lemmas[word].add(lemma)
            self._by_pos[entry["pos"]][word].add(lemma)
            self._freqs[word] = self._freqs.get(word, 0) + lemma.freq

            self._total += 1

        if isinstance(source, Path):
            fp_in.close()

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
    """
    Most of the methods of this class are chainable so you can easily combine them
    like this:
        .filter(my_func1).filter(my_func2).flip().remap({REL_ANTONYM: REL_UNKNOWN})

    mind the memory usage tho
    """

    def __init__(
        self,
        source: Union[Path, Iterable],
        rel_whitelist: Optional[List[str]] = None,
        rel_mapping: Optional[Dict[str, str]] = None,
        sort_lemmas: bool = True,
    ) -> None:
        """
        source is either a CSV file of at least three columns (word_left, word_right, relation)
        or an iterable over same tuples/dicts/Relations.

        rel_whitelist allow you to drop some relations you don't need
        rel_mapping allow you to map relations in the file/iterable to other (for example mark all
        antonyms as random).
        sort_lemmas is turned on by default and will sort all lemmas in symmetrical relations by default
        to eliminate duplicates like райдуга->веселка and веселка-райдуга
        """
        self._by_rel: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._all_rels: Set[Relation] = set()

        self._rel_mapping: Dict = {}
        if rel_mapping is not None:
            self._rel_mapping = rel_mapping

        self._rel_whitelist: List[str] = []
        if rel_whitelist is not None:
            self._rel_whitelist = rel_whitelist

        self._total_unfiltered: int = 0
        self._total: int = 0

        if isinstance(source, Path):
            fp_in = smart_open.open(source, "rt")
            dataset: Union[csv.DictReader, Iterable] = csv.DictReader(fix_nulls(fp_in))
        else:
            dataset = source

        for entry in tqdm(dataset):
            self._total_unfiltered += 1

            if isinstance(entry, Relation):
                entry = entry._asdict()
            elif isinstance(entry, (tuple, list)):
                # Booo, lame
                entry = Relation(*entry)._asdict()

            if self._rel_whitelist and entry["relation"] not in self._rel_whitelist:
                continue

            resolved_rel: str = self._rel_mapping.get(entry["relation"], entry["relation"])

            words = list(map(preprocess_lemma, [entry["word_left"], entry["word_right"]]))
            if resolved_rel not in ASSYMETRIC_RELATIONS and sort_lemmas:
                words = sorted(words)

            self._by_rel[resolved_rel].add((words[0], words[1]))
            self._all_rels.add(Relation(word_left=words[0], word_right=words[1], relation=resolved_rel))
            self._total += 1

        if isinstance(source, Path):
            fp_in.close()

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

    def to_csv(self, output_file: Union[Path, str]) -> None:
        """
        Saves the relationship dictionary into the csv file
        """
        with Path(output_file).open("w") as fp_out:
            w = csv.DictWriter(fp_out, fieldnames=["word_left", "word_right", "relation"])
            w.writeheader()

            for rel_entry in self._all_rels:
                w.writerow(rel_entry._asdict())

    def remap(self, mapping: Dict[str, str]) -> "RelationDictionary":
        """
        Returns the copy of the existing dictionary with relations mapped through the mapping
        Useful when you need to declare some relations as REL_RANDOM
        """
        return RelationDictionary(self._all_rels, rel_mapping=mapping, sort_lemmas=False)

    def flip(self) -> "RelationDictionary":
        """
        Return the copy of the existing dictionary with lemmas swapped and relations remapped
        (synonyms will remain synonyms, while assymetric relations will be replaced with their
        inverses)
        """

        return RelationDictionary(
            [(word_right, word_left, relation) for word_left, word_right, relation in self._all_rels],
            rel_mapping=FLIPPED_MAPPING,
            sort_lemmas=False,
        )

    def add_flipped(self) -> "RelationDictionary":
        """
        Return the union of existing dictionary and a flipped one
        """
        return self.union(self.flip(), sort_lemmas=False)

    def filter(self, filter_func: Callable, sort_lemmas: bool) -> "RelationDictionary":
        """
        Returns a filtered copy of the current dictionary
        """

        pairs_to_export: Set[Tuple[str, str, str]] = set()

        for rel_type, rel_pairs in self._by_rel.items():
            rel_pairs = filter_func(rel_pairs)

            for rel_pair in rel_pairs:
                pairs_to_export.add(rel_pair + (rel_type,))

        return RelationDictionary(pairs_to_export, sort_lemmas=sort_lemmas)

    def intersect(self, other_dict: "RelationDictionary", sort_lemmas: bool) -> "RelationDictionary":
        """
        Returns an intersection of two dictionaries
        """

        return RelationDictionary(self._all_rels.intersection(other_dict._all_rels), sort_lemmas=sort_lemmas)

    def union(self, other_dict: "RelationDictionary", sort_lemmas: bool) -> "RelationDictionary":
        """
        Returns a union of two dictionaries
        """

        return RelationDictionary(self._all_rels.union(other_dict._all_rels), sort_lemmas=sort_lemmas)

    def compose(
        self,
        composition: Dict[str, int],
        order_by: Optional[Callable] = None,
    ) -> "RelationDictionary":
        """
        Also function can apply optional sorting (for example by frequency) and maintain required
        dataset composition (e.g composition={REL_ANTONYM: 3500, REL_SYNONYM: 3500} will export up
        to 3500 antonyms and synonyms). You might use -1 to export all relations we got

        Caveat: sorting is not preserved during the export of the resulting dictionary. It only
        helps to pick top-n pairs of each class"""

        # Optional sorting (by lemma popularity)
        if order_by:
            pairs_to_filter: List[Tuple[str, str, str]] = sorted(self._all_rels, key=order_by, reverse=True)
        else:
            pairs_to_filter = list(self._all_rels)

        current_composition: defaultdict = defaultdict(int)
        pairs_to_export: List[Tuple[str, str, str]] = []

        for word_left, word_right, relation in pairs_to_filter:
            if relation not in composition:
                continue

            if current_composition[relation] < composition[relation] or composition[relation] == -1:
                current_composition[relation] += 1
                pairs_to_export.append((word_left, word_right, relation))

        return RelationDictionary(pairs_to_export, sort_lemmas=False)


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

    def add_lemma_dict(self, dict_handle: str, csv_source: Path) -> LemmaDictionary:
        self.lemma_dicts[dict_handle] = LemmaDictionary(csv_source, pos_whitelist=self._pos_whitelist)

        return self.lemma_dicts[dict_handle]

    def add_rel_dict(self, dict_handle: str, csv_source: Path) -> RelationDictionary:
        self.rel_dicts[dict_handle] = RelationDictionary(source=csv_source, rel_whitelist=self._rel_whitelist)

        return self.rel_dicts[dict_handle]

    def iter_relation_sources(self) -> Generator[Tuple[str, set], None, None]:
        for dict_handle, rel_dict in self.rel_dicts.items():
            for rel_type, rel_pairs in rel_dict._by_rel.items():
                yield f"{dict_handle}/{rel_type}", rel_pairs

    def apply_filter(self, pairs: set, freq_dict_handle: str, min_freq: float = 0):
        """
        Helper function to drop the pairs from the dataset which doesn't appear in the given
        lemma dict (or infrequent ones)
        """
        assert freq_dict_handle in self.lemma_dicts, "Invalid dict handle, exiting"

        freqs = self.lemma_dicts[freq_dict_handle]._freqs

        return set(
            [
                pair
                for pair in pairs
                if pair[0] in freqs and pair[1] in freqs and freqs[pair[0]] >= min_freq and freqs[pair[1]] >= min_freq
            ]
        )

    def order_by_freq(self, pair: Tuple[str, str, str], freq_dict_handle: str):
        """
        Helper function to sort the relations dict by the popularity of the relation
        (where popularity it the avg of freqs of both lemmas)
        """
        assert freq_dict_handle in self.lemma_dicts, "Invalid dict handle, exiting"

        freqs = self.lemma_dicts[freq_dict_handle]._freqs

        return (freqs.get(pair[0], 0) + freqs.get(pair[1], 0)) / 2  # Well, / 2 is not necessary here

    def overlap_matrix(self, filter_func: Optional[Callable] = None) -> pd.DataFrame:
        """
        Calculates an overlap matrix over the different datasets and relations to examine
        the number of wordpairs that overlap between dataset's relations
        """
        if filter_func is None:
            all_rels: Dict[str, Set] = {handle: rels for handle, rels in self.iter_relation_sources()}
        else:
            all_rels = {handle: filter_func(rels) for handle, rels in self.iter_relation_sources()}

        handles: list[str] = list(all_rels.keys())
        rels: list[Set] = list(all_rels.values())

        return pd.DataFrame(
            [[len(rel1.intersection(rel2)) for rel2 in rels] for rel1 in rels], index=handles, columns=handles
        )

    def combine_relations(
        self,
        rel_dicts: List[Union[str, RelationDictionary]],
    ) -> RelationDictionary:
        """
        Combines different relation dictionaries, dropping wordpairs that has more than one relation
        (usually happens due to noisy datasets, when, for example black and white appears as synonyms
        and antonyms in two different datasets).
        """
        resolved_rel_dicts: List[RelationDictionary] = [
            self.rel_dicts[rel_dict] if isinstance(rel_dict, str) else rel_dict for rel_dict in rel_dicts
        ]
        all_pairs: Set[Tuple[str, str, str]] = set()

        # Keeping track how many different relations has a wordpair
        occurences: defaultdict = defaultdict(set)
        for rel_dict in resolved_rel_dicts:
            for rel_type, rel_pairs in rel_dict._by_rel.items():
                for rel_pair in rel_pairs:
                    # Here we are sorting all the pairs despite the fact that
                    # some relations are asymmetric
                    occurences[tuple(sorted(rel_pair))].add(rel_type)
                    all_pairs.add(rel_pair + (rel_type,))

        pairs_to_export: List[Tuple[str, str, str]] = []
        for word_left, word_right, relation in all_pairs:
            # Set of relations where that word pair was used
            used_in_rels: set[str] = occurences[tuple(sorted([word_left, word_right]))]

            # If word pair was used more than in one relation (and those aren't inverse to each other)
            # we drop that pair without doubt. E.g antonym/synonym is not ok,
            # hypernym-hyponym/hyponym-hypernym is ok (TODO: still might be an issue)

            if len(used_in_rels) > 1 and (used_in_rels not in INVERSE_RELATIONS):
                continue

            pairs_to_export.append((word_left, word_right, relation))

        return RelationDictionary(pairs_to_export, sort_lemmas=False)


if __name__ == "__main__":
    # Below is the recepy to take 4 different rel datasets, filter them through
    # the vesum lemma dict (to remove all non-lemmas, rare words or phrases)
    # and then combine them into the balanced dataset of four classes where only N most
    # popular word pairs are included (according to ubertext_freq lemma frequency dictionary)

    # To speedup the computation and reduce the memory footprint a shorter version of the
    # ubertext freq dictionary called ubertext_freq.lean.csv.xz might be used

    rd = RelationDataset()
    rd.add_lemma_dict("ubertext_freq", Path("dictionaries/lemmas/ubertext_freq.csv.xz"))

    rd.add_lemma_dict("vesum", Path("dictionaries/lemmas/vesum.csv.xz"))

    ulif_synonyms = rd.add_rel_dict("ulif_synonyms", Path("dictionaries/relations/ulif_synonyms.csv.xz"))
    web_synonyms = rd.add_rel_dict("web_synonyms", Path("dictionaries/relations/web_synonyms.csv.xz"))

    # Here we are only taking synonym pairs that can be found in both web and ulif
    top_synonyms = ulif_synonyms.intersect(web_synonyms, sort_lemmas=True).add_flipped()

    web_antonyms = rd.add_rel_dict("web_antonyms", Path("dictionaries/relations/web_antonyms.csv.xz")).add_flipped()
    wn_wikidata = rd.add_rel_dict("wn_wikidata", Path("dictionaries/relations/wn_wikidata.csv.xz")).add_flipped()

    combined_dataset = rd.combine_relations(
        [top_synonyms, web_antonyms, wn_wikidata],
    ).filter(filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="vesum"), sort_lemmas=False)

    print(combined_dataset)

    one_vs_one_dataset = combined_dataset.compose(
        {
            REL_ANTONYM: 10000,
            REL_SYNONYM: 10000,
            REL_CO_HYPONYMS: 10000,
            REL_HYPERNYM_HYPONYM: 10000,
            REL_HYPONYM_HYPERNYM: 10000,
        },
        order_by=lambda x: rd.order_by_freq(x, freq_dict_handle="ubertext_freq"),
    )

    print(one_vs_one_dataset)
    one_vs_one_dataset.to_csv("/tmp/5cls.10000cap.csv")

    one_vs_all_dataset = combined_dataset.remap(
        {
            REL_ANTONYM: REL_RANDOM,
            REL_CO_HYPONYMS: REL_RANDOM,
            REL_HYPERNYM_HYPONYM: REL_RANDOM,
            REL_HYPONYM_HYPERNYM: REL_RANDOM,
        }
    ).compose(
        {REL_SYNONYM: 75000, REL_RANDOM: 75000},
        order_by=lambda x: rd.order_by_freq(x, freq_dict_handle="ubertext_freq"),
    )

    print(one_vs_all_dataset)
    one_vs_all_dataset.to_csv("/tmp/synonym-vs-random.75000cap.csv")
