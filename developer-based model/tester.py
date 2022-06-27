"""
Computes the BLEU
using the COCO metrics scripts
"""
import argparse
from bleu.bleu import Bleu
import glob


def load_textfiles(references, hypothesis):
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs")
    return refs, hypo


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def main():

    # Feed in the directory where the hypothesis summary and true summary is stored


    hyp_file = glob.glob(args.hyp)
    ref_file = glob.glob(args.ref)

    BLEU_1 = 0.
    BLEU_2 = 0.
    BLEU_3 = 0.
    BLEU_4 = 0.
    num_files = 0
    for reference_file, hypothesis_file in zip(ref_file, hyp_file):
        num_files += 1
        with open(reference_file) as rf:
            reference = rf.readlines()

        with open(hypothesis_file) as hf:
            hypothesis = hf.readlines()

        ref, hypo = load_textfiles(reference, hypothesis)
        score_map = score(ref, hypo)
        BLEU_1 += score_map['Bleu_1']
        BLEU_2 += score_map['Bleu_2']
        BLEU_3 += score_map['Bleu_3']
        BLEU_4 += score_map['Bleu_4']


    BLEU_1 = BLEU_1/num_files
    BLEU_2 = BLEU_2/num_files
    BLEU_3 = BLEU_3/num_files
    BLEU_4 = BLEU_4/num_files

    print('Average Metric Score for All Review Summary Pairs:')
    print('Bleu - 1gram:', BLEU_1)
    print('Bleu - 2gram:', BLEU_2)
    print('Bleu - 3gram:', BLEU_3)
    print('Bleu - 4gram:', BLEU_4)
    f4 = open("bleu_log.txt","a",encoding="utf-8")
    f4.write("-------------" + " ".join(args.global_step) + "-------------" + "\n")
    f4.write("Average Metric Score for All Review Summary Pairs:" + "\n")
    f4.write("Bleu - 1gram:" + str(BLEU_1) + "\n")
    f4.write("Bleu - 2gram:" + str(BLEU_2) + "\n")
    f4.write("Bleu - 3gram:" + str(BLEU_3) + "\n")
    f4.write("Bleu - 4gram:" + str(BLEU_4) + "\n")
    f4.close()

    return BLEU_1,BLEU_2,BLEU_3, BLEU_4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='')
    parser.add_argument('--hyp', type=str, default='')
    parser.add_argument('--global_step', type=str, default='')

    args = parser.parse_args()
    main()
