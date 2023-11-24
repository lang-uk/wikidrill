from typing import List, Dict, Union, Literal, Optional, Callable
import re
import json
import argparse
from random import seed, shuffle, choice
import pathlib

from tqdm import tqdm
import jinja2
import pandas as pd
from more_itertools import unique_everseen
from utils import get_hypernyms, get_hyponyms, get_cohyponyms, get_instance_hyponyms


env = jinja2.Environment(autoescape=False)
seed(42)


def num_to_str_jinja_filter(input: int) -> str:
    """
    Convert number to string in ukrainian (only for 1-9)
    """
    return {
        1: "один",
        2: "два",
        3: "три",
        4: "чотири",
        5: "п'ять",
        6: "шість",
        7: "сім",
        8: "вісім",
        9: "дев'ять",
        10: "десять",
        11: "одинадцять",
        12: "дванадцять",
        13: "тринадцять",
        14: "чотирнадцять",
        15: "п'ятнадцять",
        16: "шістнадцять",
        17: "сімнадцять",
        18: "вісімнадцять",
        19: "дев'ятнадцять",
        20: "двадцять",
        21: "двадцять один",
        22: "двадцять два",
    }.get(input, "багато")


def ukr_plural(value: Union[str, int, float], *args) -> str:
    """
    Pluralize ukrainian words according to the numerical
    """

    value = int(value) % 100
    rem = value % 10
    if value > 4 and value < 20:
        return args[2]
    elif rem == 1:
        return args[0]
    elif rem > 1 and rem < 5:
        return args[1]
    else:
        return args[2]


def ukr_plural_jinja_filter(value: Union[str, int, float], args: str) -> str:
    """
    Pluralize ukrainian words according to the numerical
    """
    return ukr_plural(value, *args.split(","))


env.filters["num_to_str"] = num_to_str_jinja_filter
env.filters["ukr_plural"] = ukr_plural_jinja_filter


def render_template(
    anchor_word: str, related_words: List[str], template: str, rel_meta: Dict
) -> str:
    """
    Render template with jinja2
    :param anchor_word: string
    :param related_words: list
    :param template: string
    :return: string
    """

    return env.from_string(template).render(
        anchor_word=anchor_word,
        related_words=related_words,
        related_words_len=len(related_words),
        rel_meta=rel_meta,
    )


TEMPLATES: Dict[str, List[str]] = {
    "hypernyms": [
        # "{{ related_words_len|num_to_str }} {{ related_words_len|ukr_plural('гіперонім,гіпероніма,гіперонімів') }} до {{ anchor_word }}",
        "Згенеруй мені {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("гіперонім,гіпероніма,гіперонімів") }} до слова "{{ anchor_word }}".',
        "Запропонуй мені {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("гіперонім,гіпероніма,гіперонімів") }} до поняття "{{ anchor_word }}".',
        'Надай мені декілька гіперонімів до слова "{{ anchor_word }}".',
        'Надай мені {{ related_words_len }} {{ related_words_len|ukr_plural("гіперонім,гіпероніма,гіперонімів") }} до слова "{{ anchor_word }}".',
        'Які слова є гіперонімами "{{ anchor_word }}"?',
        'Які слова є гіперонімами поняття "{{ anchor_word }}"?',
        'Які загальні поняття описують слово "{{ anchor_word }}"?',
        'Які ще слова належать до більш загального поняття, ніж "{{ anchor_word }}"?',
        'Які слова описують більш загальне поняття, ніж "{{ anchor_word }}"?',
        'Які інші терміни можна використовувати замість слова "{{ anchor_word }}" в більш загальному контексті?',
        'Які інші терміни можна використовувати замість "{{ anchor_word }}" в загальному контексті?',
        'Які є загальні поняття, до яких можна віднести "{{ anchor_word }}"?',
        'Які терміни відносяться до вищого рівня абстракції в порівнянні з "{{ anchor_word }}"?',
        'Які є терміни, що належать до більш широкої категорії, ніж "{{ anchor_word }}"?',
        'Які інші слова можна використовувати замість "{{ anchor_word }}" в більш загальному контексті?',
        'Чи є ще загальні категорії, до яких можна віднести "{{ anchor_word }}"?',
        'З якими іншими поняттями можна пов\'язати "{{ anchor_word }}"?',
        'Чи є ще більш абстрактні терміни, ніж "{{ anchor_word }}"?',
        'Чи є ще слова, що належать до більш широкої категорії, ніж "{{ anchor_word }}"?',
    ],
    "co-hyponyms": [
        # "{{ related_words_len|num_to_str }} {{ related_words_len|ukr_plural('когіпонім,когіпоніма,когіпонімів') }} до {{ anchor_word }}",
        "Згенеруй мені {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("когіпонім,когіпоніма,когіпонімів") }} до слова "{{ anchor_word }}".',
        "Запропонуй мені {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("когіпонім,когіпоніма,когіпонімів") }} до поняття "{{ anchor_word }}".',
        'Згенеруй мені когіпоніми до слова "{{ anchor_word }}"',
        'Запропонуй мені когіпоніми до слова "{{ anchor_word }}"',
        'Які інші слова належать до когіпонімів "{{ anchor_word }}"?',
        'Які є інші слова, що належать до когіпонімів "{{ anchor_word }}"?',
        'Які інші терміни можна використовувати як когіпоніми "{{ anchor_word }}"?',
        'Які інші поняття пов\'язані з "{{ anchor_word }}" на одному рівні абстракції?',
        'Які ще слова можна використовувати як когіпоніми до "{{ anchor_word }}"?',
        'Чи є ще терміни, які вказують на подібне поняття, що й "{{ anchor_word }}"?',
        'Які інші терміни можна використовувати як когіпоніми до "{{ anchor_word }}"?',
        'Які ще поняття пов\'язані з "{{ anchor_word }}" на одному рівні абстракції?',
        'Які ще слова є частиною загального поняття, до якого належить "{{ anchor_word }}"?',
    ],
    "hyponyms": [
        # "{{ related_words_len|num_to_str }} {{ related_words_len|ukr_plural('гіпонім,гіпоніма,гіпонімів') }} до {{ anchor_word }}",
        "Згенеруй {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("гіпонім,гіпоніма,гіпонімів") }} до слова "{{ anchor_word }}".',
        "Запропонуй мені {{ related_words_len|num_to_str }} "
        '{{ related_words_len|ukr_plural("гіпонім,гіпоніма,гіпонімів") }} до поняття "{{ anchor_word }}".',
        'Запропонуй гіпоніми до слова-гіпероніма "{{ anchor_word }}"',
        'Які слова належать до гіпонімів "{{ anchor_word }}"?',
        'Які поняття є більш конкретними, ніж "{{ anchor_word }}"?',
        'Які інші терміни можуть бути використані, щоб позначити деталізацію поняття "{{ anchor_word }}"?',
        'Які ще слова можна використовувати, щоб позначити менші елементи в рамках поняття "{{ anchor_word }}"?',
        'Які інші слова належать до гіпонімів поняття "{{ anchor_word }}"?',
        'Які терміни використовуються, щоб позначити детальніші елементи "{{ anchor_word }}"?',
        'Які інші терміни можна використовувати, щоб описати елементи "{{ anchor_word }}" більш докладно?',
        'Які є інші слова, які вказують на більш конкретні підкатегорії "{{ anchor_word }}"?',
        'Чи є ще слова, що належать до гіпонімів поняття "{{ anchor_word }}"?',
        'Які інші терміни використовуються, щоб позначити детальніші елементи, які складають "{{ anchor_word }}"?',
        'Які ще слова вказують на більш конкретні підкатегорії, що входять до "{{ anchor_word }}"?',
    ],
}


def load_titles(fname: pathlib.Path) -> pd.DataFrame:
    """
    Load titles from csv file and filter them.
    """
    filtered_uk: pd.DataFrame = pd.read_csv(fname, sep=";").loc[
        lambda x: x["rel"] == "pwn31_to_uk_wiki"
    ]
    filtered_uk = filtered_uk.loc[
        ~filtered_uk["title"].isin(
            ["Фізичне тіло", "Тверде тіло", "Матерія (фізика)", "Суще"]
        )
    ]

    filtered_uk = filtered_uk.dropna().reset_index(drop=True)

    pattern = r"\([^)]*\)"

    filtered_uk["title"] = filtered_uk["title"].apply(
        lambda x: re.sub(pattern, "", x).strip()
    )

    # Temporary filter bigrams and trigrams
    filtered_uk = filtered_uk[filtered_uk["title"].apply(lambda x: len(x.split())) < 2]

    filtered_uk = filtered_uk[
        filtered_uk["title"].str.match(r".*[^\x00-\xFF]")
    ].reset_index(drop=True)
    filtered_uk["title"] = filtered_uk["title"].str.lower()

    return filtered_uk


def get_titles(wn_dict: pd.DataFrame, titles: List[str]) -> List[List[str]]:
    """
    Get translated titles from the dictionary.
    """
    return (
        wn_dict.set_index("id_from")
        .reindex(titles)["title"]
        .dropna()
        .drop_duplicates()
        .reset_index()
        .values.tolist()
    )


def generate_hypernyms(wn_dict: pd.DataFrame, levels: int = 5) -> List[Dict]:
    """
    Generate hypernyms for each word in the dictionary up to level 5.
    """

    data = []

    for _, row in tqdm(wn_dict.iterrows(), total=len(wn_dict)):
        res = {}
        hypernyms, query_type = get_hypernyms(row["id_from"])

        if not get_titles(wn_dict, hypernyms):
            continue

        all_hypernyms = hypernyms[:]

        for _ in range(levels - 1):
            new_hypernyms = []
            for hypernym in all_hypernyms:
                indirect, _ = get_hypernyms(hypernym)
                new_hypernyms.extend(indirect)
            all_hypernyms.extend(new_hypernyms)
        all_hypernyms = list(unique_everseen(all_hypernyms))

        if all_hypernyms:
            hypernym_titles = get_titles(wn_dict, all_hypernyms)
            if hypernym_titles and (row["title"] not in hypernym_titles):
                res["query"] = row["title"]
                res["id"] = row["id_from"]
                res["query_type"] = query_type
                res["relation"] = "hypernym"
                res["related_words"] = hypernym_titles
                data.append(res)

    return data


def generate_hyponyms(wn_dict: pd.DataFrame) -> List[Dict]:
    """
    Generate hyponyms for each word in the dictionary.
    """

    data = []

    for _, row in tqdm(wn_dict.iterrows(), total=len(wn_dict)):
        res = {}
        hyponyms = list(
            set(get_hyponyms(row["id_from"]))
            | set(get_instance_hyponyms(row["id_from"]))
        )

        if hyponyms:
            hyponym_titles = get_titles(wn_dict, hyponyms)
            if not hyponym_titles:
                continue

            if hyponym_titles and (row["title"] not in hyponym_titles):
                res["query"] = row["title"]
                res["id"] = row["id_from"]
                res["query_type"] = "Concept"
                res["relation"] = "hyponym"
                res["related_words"] = hyponym_titles
                data.append(res)

    return data


def generate_cohyponyms(wn_dict: pd.DataFrame) -> List[Dict]:
    """
    Generate co-hyponyms for each word in the dictionary.
    """

    data = []

    for _, row in tqdm(wn_dict.iterrows(), total=len(wn_dict)):
        res = {}
        hyponyms = get_cohyponyms(row["id_from"])

        if hyponyms:
            cohyponym_titles = get_titles(wn_dict, hyponyms)
            if not cohyponym_titles:
                continue

            if cohyponym_titles and (row["title"] not in cohyponym_titles):
                res["query"] = row["title"]
                res["id"] = row["id_from"]
                res["query_type"] = "Concept"
                res["relation"] = "co-hyponym"
                res["related_words"] = cohyponym_titles
                data.append(res)

    return data


def turn_into_instructions(
    relations: List[Dict],
    rel_type: str,
    strategy: Literal["all", "random", "first"] = "all",
    meta: Optional[Dict] = None,
    shuffle_words: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Turn relations into instructions.
    relations: List[Dict] - list of relations (anchor word, related words, relation type)
    rel_type: str - type of the relation (hypernym, hyponym, co-hyponym)
    strategy: str - strategy for choosing the template (all, random, first)
    meta: Dict - dictionary of meta information for the anchor words in the task, used for the enrichment
    shuffle_words: bool - whether to shuffle the related words
    limit: int - limit the number of related words to use
    """
    instructions: List[Dict] = []
    if meta is None:
        meta = {}

    for rel in relations:
        if strategy == "first":
            rel_templates = [TEMPLATES[rel_type][0]]
        elif strategy == "random":
            rel_templates = [choice(TEMPLATES[rel_type])]
        else:
            rel_templates = TEMPLATES[rel_type]

        for rel_template in rel_templates:
            res: Dict = {}
            rel_meta: Dict = {}

            related_words = rel["related_words"]

            if rel["id"] in meta:
                rel_meta = res["meta"] = meta[rel["id"]]
                related_words = []

                if rel_meta["dataset"] == "test":
                    inverse_split_classes: List[str] = ["training", "trial"]
                else:
                    inverse_split_classes = ["test"]

                for rw in rel["related_words"]:
                    if meta.get(rw[0], {}).get("dataset", None) in inverse_split_classes:
                        pass
                        # print(
                        #     f"Skipping {rw[1]} ({rw[0]}) because for the instruction on "
                        #     + f"{rel['query']} ({rel['id']}) it is in the training set"
                        # )
                    else:
                        related_words.append(rw)

            if limit:
                related_words = related_words[:limit]

            res["instruction"] = render_template(
                anchor_word=rel["query"],
                related_words=related_words,
                template=rel_template,
                rel_meta=rel_meta,
            )
            res["input"] = ""
            if shuffle_words:
                shuffle(related_words)

            res["output"] = ", ".join(rw[1] for rw in related_words)

            instructions.append(res)

    return instructions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=pathlib.Path, help="Path to the file with the dictionary"
    )
    parser.add_argument(
        "output_path",
        type=pathlib.Path,
        help="Path to the file with generated instructions",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy for choosing the template",
        choices=["all", "random", "first"],
        default="first",
    )
    parser.add_argument(
        "--rel-type",
        type=str,
        help="Type of the relation",
        nargs="+",
        choices=["hypernyms", "hyponyms", "co-hyponyms"],
        default=[
            "hypernyms",
        ],
    )
    parser.add_argument(
        "--meta-path",
        type=pathlib.Path,
        help="Path to the file with meta information for the anchor words",
    )
    parser.add_argument(
        "--shuffle-related-words",
        help="Whether to shuffle the related words",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--limit-related-words",
        type=int,
        help="Limit the number of related words",
        default=10,
    )
    cli_args = parser.parse_args()

    rel_type_task: Dict[str, Callable] = {
        "hypernyms": generate_hypernyms,
        "hyponyms": generate_hyponyms,
        "co-hyponyms": generate_cohyponyms,
    }

    meta_dict: Optional[Dict] = {}

    wn_titles_dict: pd.DataFrame = load_titles(pathlib.Path(cli_args.input_path))
    if cli_args.meta_path:
        with open(cli_args.meta_path, encoding="utf-8") as fp_in:
            meta_dict = json.load(fp_in)
    else:
        meta_dict = None

    with cli_args.output_path.open("w") as fp_out:
        for rel_type in cli_args.rel_type:
            for instruction in turn_into_instructions(
                rel_type_task[rel_type](wn_titles_dict),
                rel_type,
                strategy=cli_args.strategy,
                meta=meta_dict,
                shuffle_words=cli_args.shuffle_related_words,
                limit=cli_args.limit_related_words,
            ):
                fp_out.write(json.dumps(instruction, ensure_ascii=False) + "\n")
