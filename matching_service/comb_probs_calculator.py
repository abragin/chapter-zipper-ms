from .lr_probs_calculator import get_lr_probs, handle_edge_cases
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import torch.nn.functional as F

model_nm = 'abragin/bert_book_zipper'
tokz = BertTokenizer.from_pretrained(model_nm)
model = AutoModelForSequenceClassification.from_pretrained(
        model_nm, num_labels=4)

@lru_cache(maxsize = 2000)
def nn_model_v1(txt_source, txt_target):
    source_toks = tokz.tokenize(txt_source)
    target_toks = tokz.tokenize(txt_target)
    if len(source_toks) + len(target_toks) > 509: # 512 - (start_token + sep_token + end_token)
        source_len = max(254, 509 - len(target_toks))
        target_len = 509 - source_len
        model_inp = tokz(
            tokz.convert_tokens_to_string(source_toks[:source_len]) +
            tokz.sep_token +
            txt_target,
            truncation = True,
            return_tensors="pt"
        )
        with torch.inference_mode():
            probs = F.softmax(model(**model_inp).logits[0], dim=0).numpy()
        return {
            'consistent': 1 - probs[3],
            'inconsistent': probs[3]

        }
    else:
        model_inp = tokz(
            txt_source + tokz.sep_token + txt_target,
            truncation = True,
            return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**model_inp).logits[0]
            probs = F.softmax(logits[:4], dim=0).numpy()
        return {
            'match': probs[0],
            'source_ahead': probs[1],
            'target_ahead': probs[2],
            'consistent': 1 - probs[3],
            'inconsistent': probs[3]
        }

thresholds = {
    'compl_match': 0.41,
    'compl_sa_ta': 0.31,
    'incompl_match': 0.87,
    'incompl_sa_ta': 0.62

}

def combine_probs(current_probs, next_probs, lr_probs):
    resp_incons = {
        'match': 0.0,
        'source_ahead': 0.0,
        'target_ahead': 0.0,
        'inconsistent': 1.0
    }
    if current_probs["inconsistent"] > 0.9:
        return resp_incons
    if 'match' in current_probs:
        probs = {
            'match': 0.22 * current_probs['match'] + 0.26 * lr_probs['match'] + 0.6 * next_probs['consistent'] - 0.06,
            'source_ahead': 0.42 * current_probs['source_ahead'] + 0.42 * lr_probs['source_ahead'] + 0.17 * next_probs['inconsistent'] - 0.02,
            'target_ahead': 0.42 * current_probs['target_ahead'] + 0.42 * lr_probs['target_ahead'] + 0.17 * next_probs['inconsistent'] - 0.02
        }
        probs['inconsistent'] = 0.0
        decision = max(probs, key=probs.get)
        if decision == 'match':
            if probs[decision] < thresholds['compl_match']:
                return resp_incons
        else:
            if probs[decision] < thresholds['compl_sa_ta']:
                return resp_incons
    else:
        probs = {
            'match': 0.68 * lr_probs['match'] + 0.64 * next_probs['consistent'] - 0.32,
            'source_ahead': 0.9 * lr_probs['source_ahead'] + 0.26 * next_probs['inconsistent'] - 0.03,
            'target_ahead': 0.9 * lr_probs['target_ahead'] + 0.26 * next_probs['inconsistent'] - 0.03,
        }
        probs['inconsistent'] = 0.0
        decision = max(probs, key=probs.get)
        if decision == 'match':
            if probs[decision] < thresholds['incompl_match']:
                return resp_incons
        else:
            if probs[decision] < thresholds['incompl_sa_ta']:
                return resp_incons
    return probs


def get_combined_probs(ps_source, ps_target):
    res = handle_edge_cases(ps_source, ps_target)
    if res:
        return res
    probs_current = nn_model_v1(ps_source[0], ps_target[0])
    probs_next = nn_model_v1(ps_source[1], ps_target[1])
    lr_probs = get_lr_probs(ps_source, ps_target)
    res = combine_probs(probs_current, probs_next, lr_probs)
    return res
