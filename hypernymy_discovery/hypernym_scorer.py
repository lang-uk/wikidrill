import numpy as np
import argparse
import sys


def mean_reciprocal_rank(r):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    r = np.asarray(r).nonzero()[0]
    return 1. / (r[0] + 1) if r.size else 0.


def precision_at_k(r, k, n):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return (np.mean(r) * k) / min(k, n)


def average_precision(r, n):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1, n) for k in range(r.size)]
    # Modified from the first version (removed "if r[k]"). All elements (zero and nonzero) are taken into account
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(r, n):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return average_precision(r, n)


def get_hypernyms(line, is_gold=True):
    if is_gold:
        valid_hyps = line.strip().split('\t')
        return valid_hyps
    else:
        linesplit = line.strip().split('\t')
        cand_hyps = []
        for hyp in linesplit[:limit]:
            hyp_lower = hyp.lower()
            if hyp_lower not in cand_hyps:
                cand_hyps.append(hyp_lower)
        return cand_hyps


def overlap_coefficient(gt_hyps, pred_hyps):
    intersection = set(gt_hyps) & set(pred_hyps)
    return len(intersection) / len(gt_hyps) if len(gt_hyps) > 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evaluation metrics on hypernym prediction')
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('gold', type=str, help='Path to gold file')
    parser.add_argument('predictions', type=str, help='Path to predictions file')
    parser.add_argument('--entity_type', type=str, choices=['Concept', 'Entity', 'All'], default='All',
                        help='Type of entities to include in score calculation (default: All)')
    args = parser.parse_args()

    limit = 6
    gold = args.gold
    predictions = args.predictions
    data = args.data_file

    with open(data, 'r') as fdata, open(gold, 'r') as fgold, open(predictions, 'r') as fpredictions:
        datas = fdata.readlines()
        goldls_all = fgold.readlines()
        predls_all = fpredictions.readlines()

    if len(goldls_all) != len(predls_all):
        sys.exit('ERROR: Number of lines in gold and output files differ')

    typ = args.entity_type

    if typ == "All":
        goldls = goldls_all
        predls = predls_all
    else:
        indices = [idx for idx, elem in enumerate(datas) if elem.split('\t')[1].strip() == typ]
        goldls = [goldls_all[idx] for idx in indices]
        predls = [predls_all[idx] for idx in indices]

    all_scores = []
    scores_names = ['MOC', 'MRR', 'MAP', 'P@1', 'P@3', 'P@6']

    for goldline, predline in zip(goldls, predls):
        avg_pat1 = []
        avg_pat2 = []
        avg_pat3 = []
        gold_hyps = get_hypernyms(goldline, is_gold=True)
        pred_hyps = get_hypernyms(predline, is_gold=False)
        gold_hyps_n = len(gold_hyps)

        r = [0] * limit

        for j, pred_hyp in enumerate(pred_hyps):
            if j < gold_hyps_n and pred_hyp in gold_hyps:
                r[j] = 1

        avg_pat1.append(precision_at_k(r, 1, gold_hyps_n))
        avg_pat2.append(precision_at_k(r, 3, gold_hyps_n))
        avg_pat3.append(precision_at_k(r, 6, gold_hyps_n))

        mrr_score_numb = mean_reciprocal_rank(r)
        map_score_numb = mean_average_precision(r, gold_hyps_n)
        moc_score = overlap_coefficient(gold_hyps, pred_hyps)
        avg_pat1_numb = np.mean(avg_pat1)
        avg_pat2_numb = np.mean(avg_pat2)
        avg_pat3_numb = np.mean(avg_pat3)

        scores_results = [moc_score, mrr_score_numb, map_score_numb, avg_pat1_numb, avg_pat2_numb, avg_pat3_numb]
        all_scores.append(scores_results)

    for k, score_name in enumerate(scores_names):
        scores = [score_list[k] for score_list in all_scores]
        avg_score = np.round(np.mean(scores) * 100, 2)  

        print(f"{score_name}: {avg_score}")
