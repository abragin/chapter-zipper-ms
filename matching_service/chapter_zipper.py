from .next_match_lr import next_paragraph_match_lr
from .next_match_nn import next_paragraph_match_nn


def match_chapter(source_ps, target_ps):
    source_pointer = 0
    target_pointer = 0
    connections = []
    unverified_connections = []
    while (source_pointer < len(source_ps)) or (target_pointer < len(target_ps)):
        match_lr = next_paragraph_match_lr(
                source_ps[source_pointer:],
                target_ps[target_pointer:]
        )
        match_nn = next_paragraph_match_nn(
                source_ps[:source_pointer],
                source_ps[source_pointer:],
                target_ps[:target_pointer],
                target_ps[target_pointer:]
        )
        if not match_nn:
            return {
                    "connections": connections,
                    "unverified_connections": unverified_connections
                    }
        if match_nn != match_lr:
            unverified_connections.append(len(connections))

        source_pointer += match_nn[0]
        target_pointer += match_nn[1]
        connections.append((source_pointer, target_pointer))
    return {
            "connections": connections[:-1],
            "unverified_connections": unverified_connections
            }
