import nltk
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


class SBERT_Scorer:
    def __init__(self, ckpt_name="all-roberta-large-v1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(ckpt_name)

    def compute(self, summaries, targets):
        def tokenize_sent(text):
            max_doc_sents = 100
            sentences = nltk.tokenize.sent_tokenize(text)
            ret = [sent for sent in sentences if len(sent)>10]
            if max_doc_sents != -1 and len(ret) > max_doc_sents:
                return ret[:max_doc_sents]
            else:
                return ret
            
        precision = []
        recall = []
        F_measure = []
        
        for summary, target in zip(summaries, tqdm(targets)):
            summary_sent_list = tokenize_sent(summary)
            target_sent_list = tokenize_sent(target)

            sent_list = summary_sent_list + target_sent_list

            sent_embedding = self.model.encode(sent_list)

            summary_sent_embedding = sent_embedding[: len(summary_sent_list), ...]
            target_sent_embedding = sent_embedding[len(summary_sent_list): , ...]

            cos_similarity_table = util.cos_sim(summary_sent_embedding, target_sent_embedding)

            P = sum(torch.max(cos_similarity_table, dim=1)[0]) / len(summary_sent_list)
            R = sum(torch.max(cos_similarity_table, dim=0)[0]) / len(target_sent_embedding)
            F1 = 2*P*R/(P+R)

            precision.append(P.item())
            recall.append(R.item())
            F_measure.append(F1.item())
        return (precision, recall, F_measure)
