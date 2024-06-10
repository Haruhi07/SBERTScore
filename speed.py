import os
import json
import pandas as pd
import evaluate
from tqdm import tqdm
from evaluation_utils import sbert_scorer

result_dir = "results.json"
evaluation_models_dir = "eval_ckpt"
summary_dir = "generated_summaries"
metric_list = ["ROUGE-1-F", "ROUGE-2-F", "ROUGE-L-F", "SBERTScore-P"]
summarisers_to_eval = "margin-bart-large-cnn"
summary_file_suffix = ".json" # {"id": id, "summary": summary}

data_df = pd.read_json("cnn_test.json")


json_dir = f"{summary_dir}/{summarisers_to_eval+summary_file_suffix}"
with open(json_dir, "r") as fp:
    sum_df = pd.read_json(json_dir)

main_df = pd.merge(data_df, sum_df, on="id")

source = main_df["source"].to_list()
reference = main_df["reference"].to_list()
summary = main_df["summary"].to_list()

import sys
sys.path.append("./evaluation_utils/QuestEval")
from questeval.questeval_metric import QuestEval
questeval = QuestEval(task="summarization", do_weighter=True)

list_references = [[ref] for ref in reference]
score = questeval.corpus_questeval(
    hypothesis=summary, 
    sources=source,
    list_references=list_references,
    batch_size=32,
)
print(score["corpus_score"])
main_df["QuestEval"] = score["ex_level_scores"]

rouge_scorer = evaluate.load('rouge')
sbert_scorer = sbert_scorer.SBERT_Scorer()

rouge_score = rouge_scorer.compute(predictions=summary, references=reference, use_stemmer=True)
main_df["ROUGE-1-F"] = rouge_score["rouge1"]
main_df["ROUGE-2-F"] = rouge_score["rouge2"]
main_df["ROUGE-L-F"] = rouge_score["rougeLsum"]

print(sum(main_df["ROUGE-1-F"])/len(main_df)) # 44.05 43.37
print(sum(main_df["ROUGE-2-F"])/len(main_df)) # 21.07 20.27
print(sum(main_df["ROUGE-L-F"])/len(main_df)) # 37.36 40.39

sbert_score, _, __= sbert_scorer.compute(summaries=summary, targets=source)
main_df["SBERTScore-P"] = sbert_score

print(sum(main_df["SBERTScore-P"])/len(main_df))

from summac.model_summac import SummaCConv, SummaCZS

model_zs = SummaCZS(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

summac_score = []
for doc, summ in zip(source, tqdm(summary)):
    summac_score.append(model_zs.score([doc], [summ])["scores"][0])


model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

summac_score = []
for doc, summ in zip(source, tqdm(summary)):
    summac_score.append(model_conv.score([doc], [summ])["scores"][0])