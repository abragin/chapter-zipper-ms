from transformers import AutoModelForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import torch
import functools
import os


model_nm = 'abragin/bert_book_zipper'
tokz = BertTokenizer.from_pretrained(model_nm)
paragraph_sep_token = "[PST]"
paragraph_separator = f" {paragraph_sep_token} "
before_after_sep_token = "[BAST]"
before_after_sep =  f" {before_after_sep_token} "
sep = f" {tokz.sep_token} "
if torch.cuda.is_available() and not os.getenv('FORCE_CPU'):
    torch_device = 'cuda'
else:
    torch_device = 'cpu'
model = AutoModelForSequenceClassification.from_pretrained(
        model_nm, num_labels=2).to(torch_device)

def shorten_by_tok(ps, cut_start, max_len=126):
    s = paragraph_sep_token.join(ps)
    s_words = s.split()
    if cut_start:
        s_short = ' '.join(s_words[-max_len:])
    else:
        s_short = ' '.join(s_words[:max_len])
    tokens = tokz.tokenize(s_short)
    if cut_start:
        tokens = tokens[-max_len:]
        i = 0
        while tokens[i][:2] == "##":
            i += 1
        tokens = tokens[i:]
    else:
        tokens = tokens[:max_len]
    return tokz.convert_tokens_to_string(tokens)

def build_input(ps_source_before, ps_source_after, ps_target_before, ps_target_after):
    ml = 126
    s_b = shorten_by_tok(ps_source_before, cut_start=True, max_len=ml)
    t_b = shorten_by_tok(ps_target_before, cut_start=True, max_len=ml)
    s_a = shorten_by_tok(ps_source_after, cut_start=False, max_len=ml)
    t_a = shorten_by_tok(ps_target_after, cut_start=False, max_len=ml)
    return (s_b + before_after_sep + s_a + sep + t_b + before_after_sep + t_a)

def get_match_prob(ps_source_before, ps_source_after, ps_target_before, ps_target_after):
    input_text = build_input(ps_source_before, ps_source_after, ps_target_before, ps_target_after)
    return get_match_prob_(input_text)

@functools.lru_cache(maxsize = 2000)
def get_match_prob_(input_text):
    ipt = tokz(input_text, return_tensors="pt", padding=True).to(torch_device)
    with torch.inference_mode():
        probs = F.softmax(model(**ipt).logits, dim=1)[0]
        prob_match = probs[1]
    return prob_match

def next_paragraph_match_nn(ps_source_before, ps_source_after, ps_target_before, ps_target_after):
    threshold = 0.5
    total_p_left = len(ps_source_after) + len(ps_target_after)
    max_len = 8
    si_ti_s = [
        (si, ti)
        for total_len in range(2, max_len+1)
        for si in range(1, total_len)
        if (si < len(ps_source_after)) and ((ti := total_len - si) < len(ps_target_after))
            ]
    for si, ti in si_ti_s:
        p_s_before = ps_source_before + ps_source_after[:si]
        p_s_after = ps_source_after[si:]
        p_t_before = ps_target_before + ps_target_after[:ti]
        p_t_after = ps_target_after[ti:]
        match_prob = get_match_prob(p_s_before, p_s_after, p_t_before, p_t_after)
        if match_prob > threshold:
            return (si, ti)
    if max_len >= len(ps_source_after) + len(ps_target_after):
        return (len(ps_source_after), len(ps_target_after))

def prev_paragraph_match_nn(ps_source_before, ps_source_after, ps_target_before, ps_target_after):
    threshold = 0.5
    total_p_left = len(ps_source_before) + len(ps_target_before)
    max_len = 8
    si_ti_s = [
        (si, ti)
        for total_len in range(2, max_len+1)
        for si in range(1, total_len)
        if (si < len(ps_source_before)) and ((ti := total_len - si) < len(ps_target_before))
            ]
    for si, ti in si_ti_s:
        p_s_before = ps_source_before[:-si]
        p_s_after = ps_source_before[-si:] + ps_source_after
        p_t_before = ps_target_before[:-ti]
        p_t_after = ps_target_before[-ti:] + ps_target_after
        match_prob = get_match_prob(p_s_before, p_s_after, p_t_before, p_t_after)
        if match_prob > threshold:
            return (si, ti)
    if max_len >= len(ps_source_before) + len(ps_target_before):
        return (len(ps_source_before), len(ps_target_before))
