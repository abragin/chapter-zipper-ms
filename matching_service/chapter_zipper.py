from .next_match_lr import next_paragraph_match_lr
from .next_match_nn import next_paragraph_match_nn, prev_paragraph_match_nn


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
            res_backwards = backward_chapter_match(
                    source_ps[source_pointer:],
                    target_ps[target_pointer:]
            )
            connections += [
                (c[0] + source_pointer, c[1] + target_pointer)
                for c in res_backwards['connections']
            ]
            unverified_connections += [
                uc + source_pointer
                for uc in res_backwards['unverified_connections_source_id']
            ]
            return {
                    "connections": connections,
                    "unverified_connections_source_id": unverified_connections
                    }
        if match_nn != match_lr:
            unverified_connections.append(source_pointer)

        source_pointer += match_nn[0]
        target_pointer += match_nn[1]
        connections.append((source_pointer, target_pointer))
        if match_nn != match_lr:
            unverified_connections.append(source_pointer)
    return {
            "connections": connections[:-1],
            "unverified_connections_source_id": unverified_connections
            }


def backward_chapter_match(source_ps, target_ps):
    source_pointer = len(source_ps) - 1
    target_pointer = len(target_ps) - 1
    connections = []
    unverified_connections = []
    while (source_pointer > 0) or (target_pointer > 0):
        match_lr = next_paragraph_match_lr(
                source_ps[:source_pointer][::-1],
                target_ps[:target_pointer][::-1]
        )
        match_nn = prev_paragraph_match_nn(
                source_ps[:source_pointer],
                source_ps[source_pointer:],
                target_ps[:target_pointer],
                target_ps[target_pointer:]
        )
        if not match_nn:
            return {
                    "connections": connections[::-1],
                    "unverified_connections_source_id": unverified_connections[::-1]
                    }

        source_pointer -= match_nn[0]
        target_pointer -= match_nn[1]
        connections.append((source_pointer, target_pointer))
        if match_nn != match_lr:
            unverified_connections.append(source_pointer)
    return {
            "connections": connections[:-1][::-1],
            "unverified_connections_source_id": unverified_connections[::-1]
            }
