import glob
import csv
import os
from typing import List, Optional, Tuple, Set
from functools import cache

from tqdm import tqdm
from extractor import extract_from_page
import wn


def title_by_url(url):
    """
    Returns title of the Wikipedia page
    :param url: string
    :return: string
    """
    page_ison = extract_from_page(url)["json"]
    if "title" not in page_ison:
        return
    page_title = page_ison["title"]
    return page_title


def merge_csv(folder_name, out_file, header_list, delimiter):
    """
    Merge multiple csv files
    :param folder_name: string
    :param out_file: string
    :param header_list: list
    :param delimiter: string
    :return: None
    """
    files = os.path.join(f"./{folder_name}/", "*.csv")
    files = glob.glob(files)
    files.sort()
    with open(f"./{out_file}.csv", "w") as file_out:
        writer = csv.writer(file_out, delimiter=delimiter)
        writer.writerow(header_list)
        for filename in tqdm(files):
            with open(filename, "r") as f_out:
                line = [list(elem.strip().split(";")) for elem in f_out]
                writer.writerows(line)

@cache
def get_hyponyms(synset_id: str) -> List[str]:
    """
    Returns hyponyms for given synset id.
    :param synset_id: string
    :return: list of synset ids
    """

    try:
        rels = wn.synset(synset_id).relations()
    except wn.Error as e:
        print(f"{synset_id} gave us some troubles: {e}")
        return []

    if "hyponym" not in rels:
        return []
    return [el.id for el in rels["hyponym"]]

@cache
def get_instance_hyponyms(synset_id: str) -> List[str]:
    """
    Returns instance hyponyms for given synset id.
    :param synset_id: string
    :return: list of synset ids
    """
    try:
        rels = wn.synset(synset_id).relations()
    except wn.Error as e:
        print(f"{synset_id} gave us some troubles: {e}")
        return []

    if "instance_hyponym" not in rels:
        return []

    return [el.id for el in wn.synset(synset_id).relations()["instance_hyponym"]]

@cache
def get_hypernyms(synset_id: str) -> Tuple[List[str], Optional[str]]:
    """
    Returns hypernyms for given synset id.
    :param synset_id: string
    :return: list
    """
    try:
        rels = wn.synset(synset_id).relations()
    except wn.Error as e:
        print(f"{synset_id} gave us some troubles: {e}")

        return [], None

    if "hypernym" in rels:
        return [el.id for el in rels["hypernym"]], "Concept"
    if "instance_hypernym" in rels:
        return [el.id for el in rels["instance_hypernym"]], "Entity"
    return [], None

@cache
def get_cohyponyms(synset_id: str) -> List[str]:
    """
    Returns cohyponyms for given synset id.
    :param synset_id: string
    :return: list of synset ids
    """
    rels = wn.synset(synset_id).relations()

    hypernyms = [el.id for el in rels.get("hypernym", [])] + [el.id for el in rels.get("instance_hypernym", [])]

    res: Set[str] = set()
    for hypernym in hypernyms:
        res.update(get_hyponyms(hypernym))
        res.update(get_instance_hyponyms(hypernym))

    return list(res - set([synset_id]))
