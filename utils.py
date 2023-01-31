import glob
import csv
from tqdm import tqdm
import os
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


def get_hyponyms(synset_id):
    """
    Returns hyponyms for given synset id.
    :param synset_id: string
    :return: list
    """
    if "hyponym" not in wn.synset(synset_id).relations():
        return []
    return [el.id for el in wn.synset(synset_id).relations()["hyponym"]]


def get_instance_hyponyms(synset_id):
    """
    Returns instance hyponyms for given synset id.
    :param synset_id: string
    :return: list
    """
    if "instance_hyponym" not in wn.synset(synset_id).relations():
        return []
    return [el.id for el in wn.synset(synset_id).relations()["instance_hyponym"]]
