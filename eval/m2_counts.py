"""Per-sentence TP/FP/FN from the ERRANT and ChERRANT comparison scripts, for significance tests.

compare_m2.py (ERRANT) and compare_m2_for_evaluation.py (ChERRANT) only print corpus totals. This replays
their main loop with their own functions and keeps each sentence's counts, so the counts add up exactly
to the official totals. That includes the choice of the best reference per sentence, which depends on
the running totals. Used by eval/significance.py and eval/score_gec_paper.py.
"""
import importlib.util
import sys
from collections import Counter


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"compare_m2_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_sentence_counts(compare_py, hyp_m2, ref_m2):
    """[(tp, fp, fn), ...] per sentence, from compare_py with its default settings."""
    m = load_module(compare_py)
    argv, sys.argv = sys.argv, ["compare_m2", "-hyp", hyp_m2, "-ref", ref_m2]
    try:
        args = m.parse_args()
    finally:
        sys.argv = argv
    hyp = open(hyp_m2, encoding="utf-8").read().strip().split("\n\n")
    ref = open(ref_m2, encoding="utf-8").read().strip().split("\n\n")
    assert len(hyp) == len(ref), f"{hyp_m2}: {len(hyp)} sentences, {ref_m2}: {len(ref)}"
    cherrant = hasattr(args, "max_answer_num")
    best = Counter({"tp": 0, "fp": 0, "fn": 0})
    counts = []
    for sent_id, (h, r) in enumerate(zip(hyp, ref)):
        if cherrant:
            hyp_dict = m.process_edits(m.simplify_edits(h, args.max_answer_num), args)
            ref_dict = m.process_edits(m.simplify_edits(r, args.max_answer_num), args)
            if args.reference_num is not None and len(ref_dict.keys()) != args.reference_num:
                counts.append((0, 0, 0))  # ChERRANT skips this sentence
                continue
            count, _ = m.evaluate_edits(h.split("\n")[0], hyp_dict, ref_dict, best, sent_id, args)
        else:
            hyp_dict = m.process_edits(m.simplify_edits(h), args)
            ref_dict = m.process_edits(m.simplify_edits(r), args)
            count, _ = m.evaluate_edits(hyp_dict, ref_dict, best, sent_id, h[2:].split("\nA")[0], args)
        best += Counter(count)
        counts.append((count["tp"], count["fp"], count["fn"]))
    return counts
