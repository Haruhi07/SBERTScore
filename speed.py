import os
import json
import time
import pandas as pd
import evaluate
from bert_score import BERTScorer
from evaluation_utils import sbert_scorer

result_dir = "results.json"
evaluation_models_dir = "eval_ckpt"
summary_dir = "generated_summaries"
metric_list = ["ROUGE-1-F", "ROUGE-2-F", "ROUGE-L-F", "SBERTScore-P"]

df = pd.read_csv('aggre_fact_final_main-100.csv')
df = df[:1000]

source = df["doc"].to_list()
summary = df["summary"].to_list()

rouge_scorer = evaluate.load('rouge')
scorer = sbert_scorer.SBERT_Scorer()
bert_scorer = BERTScorer(lang='en')

print("BERT")
st = time.time()
bert_score, _, __= bert_scorer.score(summary, source, batch_size=1)
ed = time.time()
running_time = ed - st
print(running_time)

print("SBERT")
st = time.time()
sbert_score, _, __= scorer.compute(summaries=summary, targets=source)
ed = time.time()

running_time = ed - st
df["SBERTScore-P"] = sbert_score

import sys
sys.path.append("./evaluation_utils/QuestEval")
from questeval.questeval_metric import QuestEval
from tqdm import tqdm
questeval = QuestEval(task="summarization", do_weighter=True)
print("QuestEval")
st = time.time()
score = questeval.corpus_questeval(
    hypothesis=summary, 
    sources=source,
    batch_size=32,
)
ed = time.time()
running_time = ed - st
print(10066*60/running_time)
df["QuestEval"] = score["ex_level_scores"]

import sys
sys.path.append("./evaluation_utils/summac")
from summac.model_summac import SummaCZS, SummaCConv
from tqdm import tqdm
print("SummaCConv")
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

summac_score = []
st = time.time()
for doc, summ in zip(source, tqdm(summary)):
    summac_score.append(model_conv.score([doc], [summ])["scores"][0])
ed = time.time()

running_time = ed - st
df["SummaC-Conv"] = summac_score

print("SummaCZS")
st = time.time()
model_conv = SummaCZS(models=["anli"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean", imager_load_cache=False)

summac_score = []
for doc, summ in zip(source, tqdm(summary)):
    summac_score.append(model_conv.score([doc], [summ])["scores"][0])
ed = time.time()

running_time = ed - st
df["SummaC-ZS"] = summac_score

print(f"SummaC-Conv Speed: {10066*60/running_time}")