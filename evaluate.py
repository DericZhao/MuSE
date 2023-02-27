from nltk.tokenize import word_tokenize
from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score
import os
from models import config
import csv
from rouge import Rouge
import numpy as np


def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores


def read_file(file_name, dec_type="Greedy"):
    f = open(f"./results/{file_name}.txt", "r", encoding="utf-8")

    refs = []
    cands = []
    dec_str = f"{dec_type}:"

    for i, line in enumerate(f.readlines()):
        if i == 1:
            _, ppl, _, acc = line.strip("EVAL	Loss	PPL	Accuracy").split()
            print(f"PPL: {ppl}\tAccuracy: {float(acc)*100}%")
        if line.startswith(dec_str):
            exp = line.strip(dec_str).strip("\n")
            cands.append(exp)
        if line.startswith("Ref:"):
            ref = line.strip("Ref:").strip("\n")
            refs.append(ref)

    return refs, cands, float(ppl), float(acc)


if __name__ == "__main__":
    if not os.path.exists(config.report_path):
        os.makedirs(config.report_path)
    files = [
        'emotion_flow'
    ]

    with open(config.report_path + "/Report_Greedy1.csv", "w", newline='') as reportFile:
        fp_write = csv.writer(reportFile)
        fp_write.writerow(
            ['Model', 'PPL', 'Distinct-1', 'Distinct-2', 'Accuracy', 'Bleu', 'Rouge-l', 'P-Bert', 'R-Bert', 'F-Bert', 'CHRF'])
        for f in files:
            print(f"Evaluating {f}")
            refs, cands, p, a = read_file(f, dec_type="Greedy")

            dist_1, dist_2 = calc_distinct(cands)

            bleu = BLEU().corpus_score(cands, [refs])
            print(f"***** BLEU: {bleu.score} *****")

            scores = Rouge().get_scores(cands, refs)
            print(f"***** Rouge-f : {np.mean([score['rouge-l']['f'] for score in scores]) * 100} *****")

            P_Bert, R_Bert, F_Bert = score(cands, refs, lang="en", rescale_with_baseline=True)
            print(f"***** Bert_P: {P_Bert.mean() * 100} *****")
            print(f"***** Bert_R: {R_Bert.mean() * 100} *****")
            print(f"***** Bert_F: {F_Bert.mean() * 100} *****")

            chrf = CHRF().corpus_score(cands, [refs])
            print(f"***** CHRF: {chrf.score} *****")

            result = ['#' + str(f),
                      round(p, 4),
                      round(dist_1 * 100, 4) , round(dist_2 * 100, 4),
                      round(a, 4),
                      round(bleu.score, 4),
                      round(np.mean([score['rouge-l']['f'] for score in scores]) * 100, 4),
                      round(float(P_Bert.mean())* 100, 4), round(float(R_Bert.mean()) * 100, 4), round(float(F_Bert.mean()) * 100, 4),
                      round(chrf.score, 4)]
            fp_write.writerow(result)

    # with open(config.report_path + "/Report_Beam.csv", "w", newline='') as reportFile:
    #     fp_write = csv.writer(reportFile)
    #     fp_write.writerow(
    #         ['Model', 'PPL', 'Distinct-1', 'Distinct-2', 'Accuracy', 'Bleu', 'P-Bert', 'R-Bert', 'F-Bert', 'CHRF', 'TER'])
    #     for f in files:
    #         print(f"Evaluating {f}")
    #         refs, cands, p, a = read_file(f, dec_type="Beam")
    #
    #         dist_1, dist_2 = calc_distinct(cands)
    #
    #         bleu = BLEU().corpus_score(cands, [refs])
    #
    #         print(f"***** BLEU: {bleu.score} *****")
    #
    #         P_Bert, R_Bert, F_Bert = score(cands, refs, lang="en", rescale_with_baseline=True)
    #         print(f"***** Bert_P: {P_Bert.mean() * 100} *****")
    #         print(f"***** Bert_R: {R_Bert.mean() * 100} *****")
    #         print(f"***** Bert_F: {F_Bert.mean() * 100} *****")
    #
    #         chrf = CHRF().corpus_score(cands, [refs])
    #         print(f"***** CHRF: {chrf.score} *****")
    #
    #         ter = TER().corpus_score(cands, [refs])
    #         print(f"***** TER: {ter.score} *****")
    #
    #         result = [f,
    #                   round(p, 4),
    #                   round(dist_1 * 100, 4), round(dist_2 * 100, 4),
    #                   round(a, 4),
    #                   round(bleu.score, 4),
    #                   round(float(P_Bert.mean()) * 100, 4), round(float(R_Bert.mean()) * 100, 4),
    #                   round(float(F_Bert.mean()) * 100, 4),
    #                   round(chrf.score, 4),
    #                   round(ter.score, 4)]
    #         fp_write.writerow(result)
