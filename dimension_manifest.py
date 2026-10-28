"""Frozen, shared task subsets for the balanced dimension-count ablation.

Pairs were matched by within-category evaluation-MSE rank (easy -> hard) using
an earlier plain-weighted checkpoint. The ablation uses unweighted MSE.
Subset sizes are numbers of individual target dimensions, not number of pairs.
"""

PAIR_BY_RANK = {
    1:  ("P2",  "E3"),
    2:  ("P7",  "D8"),
    3:  ("P6",  "F7+"),
    4:  ("P5",  "A13"),
    5:  ("P8",  "A5+"),
    6:  ("P3",  "I9"),
    7:  ("A6",  "E4+"),
    8:  ("A3−", "D1b"),
    9:  ("F9",  "E1"),
    10: ("D11", "D6+"),
    11: ("P1",  "D1a"),
    12: ("D9",  "E2"),
    13: ("A5−", "A3+"),
    14: ("P10", "I7"),
    15: ("P4",  "D4+"),
    16: ("D3−", "I4"),
    17: ("F13", "D7+"),
    18: ("A2",  "A12"),
    19: ("E4−", "D5+"),
    20: ("D5−", "A11"),
    21: ("D6−", "D3+"),
    22: ("I8",  "I1"),
    23: ("F7−", "I3+"),
}

# Pair ranks are nested. Each selected rank contributes exactly one target
# from Hostility and one from Deliberation.
PAIR_RANKS_BY_SIZE = {
    2:  (12,),
    4:  (12, 23),
    8:  (1, 6, 12, 23),
    16: (1, 4, 6, 9, 12, 15, 19, 23),
    32: (1, 2, 4, 6, 7, 9, 10, 12, 13, 15, 17, 19, 20, 21, 22, 23),
}

DIMENSION_SETS = {
    size: {
        "hostility": [PAIR_BY_RANK[r][0] for r in ranks],
        "deliberation": [PAIR_BY_RANK[r][1] for r in ranks],
        "pair_ranks": list(ranks),
    }
    for size, ranks in PAIR_RANKS_BY_SIZE.items()
}

# Validate dimensions and nesting at import time.
for _size, _subset in DIMENSION_SETS.items():
    assert len(_subset["hostility"]) == _size // 2
    assert len(_subset["deliberation"]) == _size // 2
    assert len(set(_subset["hostility"] + _subset["deliberation"])) == _size
for _small, _large in ((2, 4), (4, 8), (8, 16), (16, 32)):
    assert set(DIMENSION_SETS[_small]["hostility"]).issubset(DIMENSION_SETS[_large]["hostility"])
    assert set(DIMENSION_SETS[_small]["deliberation"]).issubset(DIMENSION_SETS[_large]["deliberation"])
