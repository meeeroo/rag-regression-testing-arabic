# Pilot: regression check (control + three injected changes, 5 runs each).
# Run in Colab after the models are loaded:   %run -i pilot_regression.py
import os, re, unicodedata, torch
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

for name in ("judge_llm", "llm", "doc_vecs", "questions", "relevant", "docs"):
    assert name in globals(), name + " is not loaded: run the restore cell first"

SYSTEM_RAG = ("أجب عن السؤال بإيجاز اعتماداً على النصوص المعطاة فقط. "
              "إذا لم تكن الإجابة موجودة في النصوص فقل: لا أعرف.")
CAUTIOUS = ("أجب عن السؤال بإيجاز اعتماداً على النصوص المعطاة فقط. "
            "إذا لم تكن متأكداً تماماً من الإجابة فقل: لا أعرف.")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalise(text):
    text = unicodedata.normalize("NFKC", str(text)).translate(DIGITS)
    text = re.sub(r"[ً-ْٰـ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = (text.replace("ى", "ي").replace("ة", "ه")
                .replace("ؤ", "و").replace("ئ", "ي"))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_yes(out):
    o = normalise(out)
    return float(o.startswith("yes") or o.startswith("نعم"))


def judge_chat(system, user, max_new_tokens=5):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = judge_tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = judge_tok(text, return_tensors="pt").to(judge_llm.device)
    with torch.no_grad():
        out = judge_llm.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
    new = out[0][inputs["input_ids"].shape[1]:]
    return judge_tok.decode(new, skip_special_tokens=True).strip()


JUDGE_C = ("You compare a model answer with a reference answer. "
           "Reply with one word only: YES or NO.")


def prompt_correct(question, ref, ans):
    return (f"Question: {question}\nReference answer: {ref}\n"
            f"Model answer: {ans}\n\n"
            "Reply YES if the model answer gives the reference answer, "
            "even if it adds extra words or details or spells a name "
            "differently. Reply NO if it gives a different answer or "
            "says it does not know.")


# baseline: the 10 repeated runs already done (pilot_runs.csv)
per_run = pd.DataFrame({
    "retrieval_hit": [0.95] * 10,
    "contains_ref": [0.70, 0.75, 0.65, 0.70, 0.80,
                     0.70, 0.65, 0.70, 0.65, 0.70],
    "judge_correct": [0.80, 0.90, 0.80, 0.85, 0.85,
                      0.75, 0.85, 0.85, 0.80, 0.85],
    "abstained": [0.05, 0.10, 0.05, 0.10, 0.05,
                  0.05, 0.10, 0.05, 0.05, 0.05]})

test = judge_chat(JUDGE_C, prompt_correct("ما عاصمة فرنسا؟", "باريس",
                                          "العاصمة هي باريس."))
print("judge test (expected YES):", test)
print("READY, starting the regression check")

EPS = 1e-9
CHECKS = {"retrieval_hit": "low", "contains_ref": "low",
          "judge_correct": "low", "abstained": "high"}
base_mean = per_run[list(CHECKS)].mean()
base_std = per_run[list(CHECKS)].std()
band = {}
for m, d in CHECKS.items():
    if d == "low":
        band[m] = base_mean[m] - 2 * base_std[m]
    else:
        band[m] = base_mean[m] + 2 * base_std[m]
print("thresholds from the baseline (mean and 2 std):")
for m, d in CHECKS.items():
    word = "below" if d == "low" else "above"
    print("  ", m, ": fail if", word, round(band[m], 3))


def answer_with(config, question):
    q = embedder.encode([question], normalize_embeddings=True,
                        convert_to_tensor=True)
    scores = (q @ doc_vecs.T)[0].float()
    if config["removed"]:
        scores[list(config["removed"])] = -1e9
    ids = torch.topk(scores, config["k"]).indices.tolist()
    context = "\n\n".join(f"[{i+1}] {docs[d][:1500]}"
                          for i, d in enumerate(ids))
    user = f"النصوص:\n{context}\n\nالسؤال: {question}"
    messages = [{"role": "system", "content": config["system"]},
                {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=100,
                           do_sample=True, temperature=0.7)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), ids


def run_version(name, config, n_runs=5, seed0=100):
    rows = []
    plan = [(r, q) for r in range(n_runs) for q in range(20)]
    for run, qid in tqdm(plan, desc=name):
        if qid == 0:
            torch.manual_seed(seed0 + run)
        q, ref = questions.question[qid], questions.answer[qid]
        ans, ids = answer_with(config, q)
        verdict = judge_chat(JUDGE_C, prompt_correct(q, ref, ans))
        rows.append({
            "version": name, "run": run, "qid": qid,
            "reference": ref, "answer": ans,
            "retrieval_hit": float(any(i in relevant[qid] for i in ids)),
            "contains_ref": float(normalise(ref) in normalise(ans)),
            "judge_correct": is_yes(verdict),
            "abstained": float("لا اعرف" in normalise(ans))})
    return pd.DataFrame(rows)


BASE = {"system": SYSTEM_RAG, "k": 5, "removed": set()}
removed = set()
for q in [0, 2, 3, 10, 14]:
    removed |= relevant[q]
versions = {
    "control (no change)": BASE,
    "prompt change": {**BASE, "system": CAUTIOUS},
    "retrieval change (k=1)": {**BASE, "k": 1},
    "knowledge base change": {**BASE, "removed": removed}}

all_rows, lines = [], []
for name, config in versions.items():
    df = run_version(name, config)
    all_rows.append(df)
    df.to_csv("pilot_changes_partial.csv", mode="a", index=False,
              encoding="utf-8-sig",
              header=not os.path.exists("pilot_changes_partial.csv"))
    pr = df.groupby("run")[list(CHECKS)].mean()
    failed, reasons = 0, Counter()
    for _, row in pr.iterrows():
        bad = []
        for m, d in CHECKS.items():
            if d == "low" and row[m] < band[m] - EPS:
                bad.append(m)
            if d == "high" and row[m] > band[m] + EPS:
                bad.append(m)
        failed += bool(bad)
        reasons.update(bad)
    means = pr.mean().round(2).to_dict()
    lines.append((name, means, failed, len(pr), dict(reasons)))
    print(name, "| runs failed:", failed, "of", len(pr),
          "| failed on:", dict(reasons) or "nothing")

print()
for name, means, failed, n, reasons in lines:
    print("-" * 60)
    print(name)
    print("   mean scores :", means)
    print("   runs failed :", failed, "of", n)
    print("   failed on   :", reasons if reasons else "nothing")

changes_df = pd.concat(all_rows, ignore_index=True)
changes_df.to_csv("pilot_changes.csv", index=False, encoding="utf-8-sig")
try:
    from google.colab import files
    files.download("pilot_changes.csv")
except Exception:
    print("file saved as pilot_changes.csv in the Files panel")
