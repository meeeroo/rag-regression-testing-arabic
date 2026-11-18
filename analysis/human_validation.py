"""Agreement between the two annotators, and between the automated metrics and the gold labels."""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, '..', 'data', 'human_validation', 'labels.csv'))

def kappa(a, b):
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return po, (po - pe) / (1 - pe)

def report(name, a, b):
    po, k = kappa(a, b)
    print(f'{name:45s} n={len(a):3d}  agreement={po:.3f}  kappa={k:.3f}')

both = d[d.annotator_B_correct.notna()]
report('Annotators: correctness', both.annotator_A_correct, both.annotator_B_correct)
both = d[d.annotator_B_supported.notna()]
report('Annotators: support', both.annotator_A_supported, both.annotator_B_supported)
gold = (d.gold_correct == 'yes').astype(int)
report('Judged correctness vs gold', d.judge_correct.astype(int), gold)
report('Reference containment vs gold', d.contains_ref.astype(int), gold)
f = d[d.judged_faithful.notna() & (d.gold_supported != 'not_applicable')]
report('Judged faithfulness vs gold (support)', f.judged_faithful.astype(int), (f.gold_supported == 'yes').astype(int))
