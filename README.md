# A Quality Assurance and Testing Framework for RAG Applications: A Case Study on Arabic Data

Code, data, and results of the MSc thesis *A Quality Assurance and Testing Framework for Retrieval-Augmented Generation (RAG) Applications: A Case Study on Arabic Data*.

The framework tests a RAG application the way a release pipeline would: it runs a golden test set several times against the unchanged application, derives a tolerance band for every metric from the measured run-to-run variance, and fails a new version only when a metric falls outside its band. The bands use a prediction interval for the small number of baseline runs and a Bonferroni correction for the number of metrics:

```
limit = mean ± k · max(SD, 1/N),   k = t(1 − 0.05/q, n − 1) · √(1 + 1/n)
```

where *n* is the number of baseline runs, *q* the number of metrics, and *N* the number of questions. With n = 10 and q = 6, k ≈ 3.08.

## Main results (Arabic, 100 questions, 60 runs, 6,000 answers)

| Decision rule | Detection (30 regressed runs) | False alarms (5 control runs) | Failures without an effect (15 runs) |
|---|---|---|---|
| Tolerance bands of the framework (k = 3.08) | 100% | 0% | 0% |
| Two standard deviations | 100% | 40% | 40% |
| Fixed threshold 0.5 | 70% | 0% | 0% |
| Single-run comparison | 100% | 80% | 87% |

Human validation (two native Arabic annotators, 100 answers): judged correctness agrees with the gold labels with Cohen's kappa 0.87, reference containment with 0.73, and judged faithfulness with 0.02. Judged faithfulness was therefore excluded from the verdicts.

The same rule raised no false alarm on an English sample (HotpotQA) or on an existing application built with LlamaIndex, and detected every run of the knowledge-base change in both.

## Repository structure

```
src/
  rag_system.py            RAG system under test, judge, and metrics (setup)
  run_experiments.py       Baseline, control, and injected-change runs
  extra_experiments.py     English sample, LlamaIndex application, and change M2
  pilot/                   Pilot study (20 questions)
analysis/
  analysis_arabic.py       Bands, verdicts, statistical tests (Arabic)
  analysis_english_app.py  The same analysis for the English sample and the application
  human_validation.py      Agreement of annotators and of the automated metrics
  figures.py               Figures 4.1 and 4.2 of the thesis
data/
  results/                 One row per answer for every run
  pilot/                   Pilot study results
  human_validation/        Labels of the two annotators and the gold labels
results/
  stats_arabic.json, stats_english_app.json, figures/
```

In the data files, the prompt changes are named `P1` and `P2`; in the thesis they are called T1 and T2.

| Code | Change |
|---|---|
| K1, K2 | Documents of 10 and 25 questions removed from the knowledge base |
| R1, R2 | 3 and 1 retrieved documents instead of 5 |
| M1, M2 | Generator replaced by Qwen2.5-1.5B and Qwen2.5-0.5B |
| P1 (T1), P2 (T2) | Cautious instruction added; grounding instruction removed |
| no_retrieval | Retrieval disabled |
| control | No change |

## Reproducing the analysis

The analysis runs on a CPU in a few seconds:

```bash
pip install -r requirements.txt
python analysis/analysis_arabic.py
python analysis/analysis_english_app.py
python analysis/human_validation.py
python analysis/figures.py
```

## Re-running the experiments

The experiments were run on Kaggle with two T4 GPUs (and partly on Google Colab). In a Kaggle notebook with GPU and internet enabled:

```python
%run src/rag_system.py        # loads the benchmark and the models
%run src/run_experiments.py   # baseline, control, and changes; resumes after interruption
%run src/extra_experiments.py # English sample, LlamaIndex application, M2
```

Models: `BAAI/bge-m3` (retrieval), `Qwen/Qwen2.5-3B-Instruct` (generation, temperature 0.7), and `Qwen2.5-7B-Instruct` in 4-bit (judge). Because generation is sampled, re-running produces different answers; the framework is designed for exactly this variation.

## Data sources

- Arabic RAG benchmark: Almohaimeed et al. (2026), *Benchmarking retrieval augmented generation LLMs for Arabic noise robustness*, Frontiers in Big Data. Repository: https://github.com/Saleh-Almohaimeed/Arabic-RAG-Benchmark
- English sample: HotpotQA, distractor setting (Yang et al., 2018).

The result files contain questions and reference answers from these datasets; please follow their licences when reusing them.

## Citation

```
Marwa Hassan Eltaieb Mustafa (2026). A Quality Assurance and Testing Framework for Retrieval-Augmented Generation (RAG) Applications: A Case Study on Arabic Data. MSc thesis, Sudan University of Science and Technology.
```

## Licence

The code is released under the MIT licence (see `LICENSE`).
