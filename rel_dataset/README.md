# rel_dataset.py — swiss army knife to work on word-pair datasets

## General concept
rel_dataset (a better name is needed) operates on the following entities:
 - LemmaDictionary — dictionary of lemmas with optional frequency and pos tag. POS tags are aligned with the UD tagset
 - RelationDictionary — dictionary that consists of two lemmas and the relation between them. Probably the most important class. It can load dictionaries from the CSV or from an iterable, it can filter relations/map them, combine two dictionaries using union/intersect, and flip the relations, and all that operations are chainable (see below)
 - Lemma — one lemma with optional freq/pos
 - Relation — one relation between `word_left` and `word_right`
 - All possible relations are available under `ALL_RELATIONS`
 - RelationDataset — a set of tools to control the zoo of dictionaries and perform various operations on the dictionaries. This class might be refactored someday and disappear at all


## Important caveats:
 - Some relations are symmetrical (synonym, antonym), while some are asymmetrical (hypernym-hyponym). I.e., swapping two lemmas in the synonym will yield a pair of synonyms. Swapping hypernym and hyponym will yield different relation
 - By default, words in symmetrical relations are sorted, so duplicates like "car-auto" and "auto-car" will be normalized. This behavior can be disabled by passing the `sort_lemmas` param to the constructor or chainable methods like `intersect` or `union`
 - Sometimes, it's good to add also reverse relations. To do so, you can use the `flip` method and then make sure that you don't sort your lemmas
 — You might use the `intersect` method to find the common subset of synonyms from different but noisy sources. In this way, you'd probably end up with a cleaner dataset of synonyms
 - You might (and should) filter your relations against lemma dictionaries, maybe with a threshold on the min frequency.
 - `combine_relations` method will drop the word-pairs which appear in two different relations (again, this happens because of the noisy data)


## Data
The package comes with some datasets like antonyms, and synonyms crawled from the web, synonyms of the `ulif` dictionary, lemmas from `ulif`, `vesum` and `ubertext.freq` (this one has POS tags and frequency). `ubertext_freq.lean` is a smaller version of the `ubertext_freq` with only top 99999 lemmas. `wn_wikidata` is a broader dataset of relations collected from Princeton wordnet and wikidata/Wikipedia.

## Recepies

### Toy dataset of 5 classes:

```python
    rd = RelationDataset()
    # Load frequency dict to sort wordpairs by popularity
    rd.add_lemma_dict("ubertext_freq", Path("dictionaries/lemmas/ubertext_freq.csv.xz"))
    # Load vesum dict to remove OOV words from dataset
    rd.add_lemma_dict("vesum", Path("dictionaries/lemmas/vesum.csv.xz"))

    # Read two dictionaries of synonyms
    ulif_synonyms = rd.add_rel_dict("ulif_synonyms", Path("dictionaries/relations/ulif_synonyms.csv.xz"))
    web_synonyms = rd.add_rel_dict("web_synonyms", Path("dictionaries/relations/web_synonyms.csv.xz"))

    # Here we are only taking synonym pairs that can be found in both web and ulif
    # to filter noisy pairs that only appears in one of the dataset
    # .add_flipped adds the flipped copy of the dictionary
    top_synonyms = ulif_synonyms.intersect(web_synonyms, sort_lemmas=True).add_flipped()

    # Antonyms
    web_antonyms = rd.add_rel_dict("web_antonyms", Path("dictionaries/relations/web_antonyms.csv.xz")).add_flipped()

    # hypernym-hyponym, co-hyponym, hypernym-instance and co-instance
    wn_wikidata = rd.add_rel_dict("wn_wikidata", Path("dictionaries/relations/wn_wikidata.csv.xz")).add_flipped()

    # Combine all of them together while dropping ambigous relations
    combined_dataset = rd.combine_relations(
        [top_synonyms, web_antonyms, wn_wikidata],
    # And filtering out OOV words
    ).filter(filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="vesum"), sort_lemmas=False)

    # Printing intermediate statistics
    print(combined_dataset)

    # Preparing the final dataset: up to 100 pairs of each type of relations
    one_vs_one_dataset = combined_dataset.compose(
        {
            REL_ANTONYM: 100,
            REL_SYNONYM: 100,
            REL_CO_HYPONYMS: 100,
            REL_HYPERNYM_HYPONYM: 100,
            REL_HYPONYM_HYPERNYM: 100,
        },
        # Also, let those 100 be the most popular ones
        order_by=lambda x: rd.order_by_freq(x, freq_dict_handle="ubertext_freq"),
    )

    print(one_vs_one_dataset)
    # Saving the dataset to the file
    one_vs_one_dataset.to_csv("/tmp/5cls.toy.csv")

    one_vs_all_dataset = combined_dataset.remap(
        {
            REL_ANTONYM: REL_RANDOM,
            REL_CO_HYPONYMS: REL_RANDOM,
            REL_HYPERNYM_HYPONYM: REL_RANDOM,
            REL_HYPONYM_HYPERNYM: REL_RANDOM,
        }
    ).compose(
        {REL_SYNONYM: 100, REL_RANDOM: 100},
        order_by=lambda x: rd.order_by_freq(x, freq_dict_handle="ubertext_freq"),
    )

    print(one_vs_all_dataset)
    one_vs_all_dataset.to_csv("/tmp/synonym-vs-random.toy.csv")
```

### Build confusion matrix between word-pair datasets

```python
    # The recepy below allows to build a confusion matrix between different datasets using different
    # filtering strategies. That can show how many different word pairs has more than one relation
    # in different dataset (i.e considered as a synonym AND antonym by different sources)

    with pd.ExcelWriter("/tmp/overlap_matrix.xlsx") as pd_writer:
        overlap_df = rd.overlap_matrix()
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="unfiltered rels")

        overlap_df = rd.overlap_matrix(filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="ulif"))
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (ulif)")

        overlap_df = rd.overlap_matrix(filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="vesum"))
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (vesum)")

        overlap_df = rd.overlap_matrix(
            filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="ubertext_freq", min_freq=1e-6)
        )
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (freqs, 1e-6)")

        overlap_df = rd.overlap_matrix(
            filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="ubertext_freq", min_freq=5e-6)
        )
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (freqs, 5e-6)")

        overlap_df = rd.overlap_matrix(
            filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="ubertext_freq", min_freq=5e-7)
        )
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (freqs, 5e-7)")

        overlap_df = rd.overlap_matrix(
            filter_func=lambda x: rd.apply_filter(x, freq_dict_handle="ubertext_freq", min_freq=1e-7)
        )
        print(overlap_df)
        overlap_df.to_excel(pd_writer, sheet_name="filtered rels (freqs, 1e-7)")
```
