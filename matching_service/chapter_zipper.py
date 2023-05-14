from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .lr_probs_calculator import get_lr_probs
from .comb_probs_calculator import get_combined_probs


class Matcher(object):
    def __init__(self, probs_func):
        self.probs_func = probs_func

    def __call__(self, ps_source, ps_target, source_pointer=1,
            target_pointer=1):
        source_txt = ' '.join(ps_source[:source_pointer])
        target_txt = ' '.join(ps_target[:target_pointer])
        ps_source_current = [source_txt] + ps_source[source_pointer:]
        ps_target_current = [target_txt] + ps_target[target_pointer:]
        probs = self.probs_func(ps_source_current, ps_target_current)
        decision = max(probs, key=probs.get)
        if decision == "match":
            return (source_pointer, target_pointer)
        elif decision == "source_ahead":
            return self(
                ps_source, ps_target, source_pointer, target_pointer+1
            )
        elif decision == "target_ahead":
            return self(
                ps_source, ps_target, source_pointer+1, target_pointer
            )
        else:
            return None

match_combined = Matcher(get_combined_probs)
match_lr = Matcher(get_lr_probs)

def match_chapter(source_ps, target_ps):
    source_pointer = 0
    target_pointer = 0
    connections = []
    first_inconsistent_connection = None
    next_match_fn = match_combined
    while (source_pointer < len(source_ps)) or (target_pointer < len(target_ps)):
        match = next_match_fn(
                source_ps[source_pointer:],
                target_ps[target_pointer:]
                )
        if not match:
            first_inconsistent_connection = len(connections) - 1
            next_match_fn = match_lr
            match = next_match_fn( source_ps[source_pointer:],
                    target_ps[target_pointer:])
        source_pointer += match[0]
        target_pointer += match[1]
        connections.append((source_pointer, target_pointer))
    return {
            "connections": connections[:-1],
            "first_inconsistent_connection": first_inconsistent_connection
            }
