"""One rule for "these two detections are the same part".

Two poses are duplicates when their object centres nearly coincide *and*
their orientations agree. Both tests are needed: two parts stacked flat sit
only a thickness apart, but then their orientations differ. The learned and
the geometric detector, the fold evaluation and the submission merger all
apply this one rule, so a duplicate means the same thing everywhere.
"""

import numpy as np

#: Two detections this close (object centres, metres) AND this aligned
#: (degrees) are duplicates. Both tests: two parts stacked flat sit only a
#: thickness apart, but then their orientations differ.
NMS_DIST = 0.009
NMS_ANGLE_DEG = 30.0



def is_duplicate(R_a: np.ndarray, t_a: np.ndarray,
                 R_b: np.ndarray, t_b: np.ndarray) -> bool:
    """The rule itself, on raw rotation matrices and translations (metres)."""
    if np.linalg.norm(np.asarray(t_a) - np.asarray(t_b)) > NMS_DIST:
        return False
    cos = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) < NMS_ANGLE_DEG


def nms(estimates: list) -> list:
    """Drop the lower-confidence member of any duplicate pair of estimates.

    ``estimates`` carry ``R``, ``t`` and ``submission_score``; the highest
    score of a duplicate group survives.
    """
    kept = []
    for est in sorted(estimates, key=lambda e: -e.submission_score):
        if not any(is_duplicate(est.R, est.t, k.R, k.t) for k in kept):
            kept.append(est)
    return kept
