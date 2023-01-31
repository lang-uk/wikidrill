import glob
import json
import os
import shutil
import pandas as pd
import wn
from anytree import Node, RenderTree
from joblib import Parallel, delayed
from tqdm import tqdm

UK = "pwn31_to_uk_wiki"
EN = "pwn31_to_en_wiki"
ILI = "pwn31_to_ili"


def get_title_node(wd_id, df):
    """
    Returns title for the node: Ukrainian, English or ILI.
    :param wd_id: string
    :param df: DataFrame
    :return: string
    """
    title_uk = df[((df.id_from == wd_id) & (df.rel == UK))].title
    title_en = df[((df.id_from == wd_id) & (df.rel == EN))].title
    if title_uk.any():
        title = title_uk.iloc[0]
    elif title_en.any():
        title = f"*{title_en.iloc[0]}*"
    else:
        lemma = wn.synset(wd_id).lemmas()
        title = f"*{lemma[0] if lemma else ''}*"
        # title = f"*{df[((df.id_from == wd_id) & (df.rel == ILI))].title.iloc[0]}*"
    return title


def get_all_hyponyms(synset_id):
    """
    Returns hyponyms and instance hyponyms for given synset id.
    :param synset_id: string
    :return: list
    """
    return [el.id for el in wn.synset(synset_id).hyponyms()]


def get_name(node):
    """
    :param node: dict
    :return: iterator
    """
    return next((v["title"] for v in node.values() if "title" in v), None)


def get_ili(synset_id):
    """
    Returns ili for synset.
    :param synset_id: string
    :return: string
    """
    ili = wn.synset(synset_id).ili
    return ili.id if ili else "ili-in"


def save_tree(tree, filename, idx):
    """
    Creates image of tree and writes its to file.
    :param tree: Node
    :param filename: string
    :param idx: int
    :return: None
    """
    if tree.height < 2:
        return
    with open(filename, "w") as fp_out:
        fp_out.write(f"{idx}) Tree height: {tree.height}\n")
        for pre, fill, node in RenderTree(tree):
            fp_out.write("%s%s" % (pre, get_name(node.name)))
            fp_out.write("\n")


def find_path_with_most_pairs(node, path, path_with_most_pairs, max_pairs):
    """
    Finds paths in the tree with the most pairs of nodes that are connected by an edge without Ukrainian translation.
    :param node: anytree.Node
    :param path: List[str]
    :param path_with_most_pairs: Dict[str, Any]
    :param max_pairs: int
    :return: Tuple[List[Dict[str, Any]], int]
    """
    path = path + [node]
    if "*" in get_name(node.name) and not node.is_leaf:
        pairs = []
        for nd in path:
            if ("*" in get_name(nd.name)) and (not nd.is_leaf):
                curr_pairs = [
                    (get_name(nd.name), get_name(child.name))
                    for child in nd.children
                    if "*" not in get_name(child.name)
                ]
                if curr_pairs:
                    pairs.extend(curr_pairs)
        num_pairs = len(pairs)
        res_path = [node.name for node in path]
        res_path = {k: v for d in res_path for k, v in d.items()}
        num_gaps = len(
            [
                value["title"]
                for key, value in res_path.items()
                if "*" in value.get("title", "")
            ]
        )
        if num_pairs > max_pairs:
            max_pairs = num_pairs
            path_with_most_pairs = {
                "generated_pairs": num_pairs,
                "gaps_to_fill": num_gaps,
                "path": res_path,
            }

    for child in node.children:
        path_with_most_pairs, max_pairs = find_path_with_most_pairs(
            child, path, path_with_most_pairs, max_pairs
        )
    return path_with_most_pairs, max_pairs


def build_tree(index, row, json_filename, tree_dir, titled_df):
    """
    Builds tree for given hypernym.
    :param index: int
    :param row: Series
    :param json_filename: string
    :param tree_dir: string
    :param titled_df: DataFrame
    :return: None
    """
    visited, queue = set(), []
    X = row["id_from"]
    visited.add(X)
    queue.append(X)
    start = row["title"]
    first = {X: {"ili": get_ili(X), "title": start, }}
    titles = {}
    titles.update(first)
    tree = Node(first)
    nodes = {X: tree}
    hyponyms = {}
    while queue:
        synset_id = queue.pop(0)
        if synset_id not in hyponyms:
            hyponyms[synset_id] = get_all_hyponyms(synset_id)
        if synset_id in titles:
            page_title = titles[synset_id]["title"]
        else:
            page_title = get_title_node(synset_id, titled_df)
        titles[synset_id] = {
            "ili": get_ili(synset_id),
            "title": page_title,
        }
        if page_title == start:
            parent_node = tree
        else:
            parent_node = nodes.get(synset_id)
            if not parent_node:
                continue
        for idx, elem in enumerate(hyponyms[synset_id]):
            if elem not in visited:
                visited.add(elem)
                queue.append(elem)
            if elem in titles:
                hyp_title = titles[elem]["title"]
            else:
                hyp_title = get_title_node(elem, titled_df)
            hyp_elem = {elem: {"ili": get_ili(elem), "title": hyp_title, }}
            titles.update(hyp_elem)
            node = Node(hyp_elem, parent=parent_node)
            nodes[elem] = node
    save_tree(tree, f"./{tree_dir}/{X}.txt", index)

    paths_with_most_pairs, max_pairs = find_path_with_most_pairs(tree, [], {}, 0)
    result = {"title": start}
    if paths_with_most_pairs.get("generated_pairs", 0) > paths_with_most_pairs.get(
            "gaps_to_fill", 0
    ):
        result.update(paths_with_most_pairs)
    else:
        return

    with open(json_filename, "w") as fp:
        json.dump(result, fp, ensure_ascii=False)


if __name__ == "__main__":
    titled_pwn = pd.read_csv("../data/titled_pwn.csv", sep=";")
    pos1 = titled_pwn.loc[titled_pwn.id_from == "omw-en31-10364746-n"].index[0]
    titled_pwn.loc[pos1, 'title'] = "nan_"
    pos2 = titled_pwn.loc[titled_pwn.id_from == "omw-en31-02510539-s"].index[0]
    titled_pwn.loc[pos2, 'title'] = "null_"
    titled_pwn.dropna(inplace=True)
    filtered_uk = titled_pwn.loc[titled_pwn["rel"] == "pwn31_to_uk_wiki"]
    filtered_uk.sort_values(by="id_from", inplace=True)
    filtered_uk.reset_index(drop=True, inplace=True)

    # dir for titled csv files
    path_dir_trees = os.getcwd() + "/trees/"
    path_dir_stats = os.getcwd() + "/stats/"
    os.mkdir(path_dir_trees)
    os.mkdir(path_dir_stats)

    res = Parallel(n_jobs=-1)(
        delayed(build_tree)(index, row, f"./stats/{index}.json", "trees", titled_pwn)
        for index, row in tqdm(filtered_uk[::-1].iterrows(), total=len(filtered_uk))
    )

    # Use glob to find all JSON files in the given folder
    json_files = glob.glob("./stats/*.json")

    data = []
    # Iterate over the JSON files
    for json_file in json_files:
        with open(json_file, "r") as f:
            # Load the JSON data from the file
            file_data = json.load(f)
            # Update the merged data with the new data
            data.append(file_data)

    # Write the merged data to a new file
    with open("../path_with_gaps.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)

    shutil.rmtree(path_dir_stats)
