import csv
import pandas as pd
import wn
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import merge_csv, title_by_url
import os
import shutil

UK = "pwn31_to_uk_wiki"
EN = "pwn31_to_en_wiki"
ILI = "pwn31_to_ili"


def write_one(filename, key, value):
    """
    :param filename: string
    :param key: int
    :param value: list
    :return: None
    """
    with open(filename, "w") as file_out:
        writer = csv.writer(file_out, delimiter=";")
        if (UK in value) or (EN in value):
            writer.writerow(value + [title_by_url(value[1])])
        else:
            writer.writerow(value + [wn.synset(value[0]).lemmas()[0]])


if __name__ == "__main__":
    # read and filter main dataframe
    pwn_friends = pd.read_csv("./data/pwn_friends.csv")
    df = pwn_friends[
        pwn_friends["rel"].isin(["pwn31_to_uk_wiki", "pwn31_to_en_wiki", "pwn31_to_ili"])
    ].sort_values(by=["rel"], ascending=[False])
    df = df.drop_duplicates(subset="id_from").reset_index(drop=True)
    dct = df.set_index(df.index).T.to_dict("list")
    # dir for titled csv files
    path_dir = os.getcwd() + "/titled/"
    os.mkdir(path_dir)
    # parallel loop
    res = Parallel(n_jobs=-1)(
        delayed(write_one)(f"{path_dir}{k}.csv", k, v)
        for k, v in tqdm(dct.items(), total=len(dct))
    )

    # merge and clean-up
    header = ["id_from", "id_to", "rel", "title"]
    merge_csv("titled", "data/titled_pwn", header, ";")
    shutil.rmtree(path_dir)
