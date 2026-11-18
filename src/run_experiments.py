# Full experiments, part 2: runs (Kaggle version). Stops starting new runs before the 12-hour limit.
from tqdm.auto import tqdm
# Each run = the 100 sample questions. Finished runs are skipped, so after a disconnect
# just run setup and this file again: it continues where it stopped.
PLAN = ([("baseline", r, r) for r in range(10)] +
        [("control", r, 100 + r) for r in range(5)] +
        [(v, r, 200 + r) for v in ["P1", "P2", "K1", "K2", "R1", "R2", "no_retrieval"] for r in range(5)] +
        [("M1", r, 200 + r) for r in range(5)])          # model change last (needs a model swap)

FAITH_VERSIONS = {"baseline", "control", "P1", "P2"}   # faithfulness judged where it is needed
done = set()
if os.path.exists(RESULTS):
    prev = pd.read_csv(RESULTS, encoding="utf-8-sig")
    done = set(zip(prev.version, prev.run))
todo = [p for p in PLAN if (p[0], p[1]) not in done]
print(f"runs finished: {len(PLAN) - len(todo)} of {len(PLAN)} | remaining: {len(todo)}")

TIME_BUDGET = 11.3 * 3600   # Kaggle stops a session at 12 hours
for version, run, seed in todo:
    if time.time() - T_START + 1400 > TIME_BUDGET:
        print("TIME LIMIT: stopping here. Save the output file and start a new version to continue.")
        break
    cfg = VERSIONS[version]
    load_generator(cfg["model"])
    torch.manual_seed(seed)
    rows, t_run = [], time.time()
    for qid in tqdm(SAMPLE, desc=f"{version} run {run}", leave=False):
        q, ref = questions.question[qid], questions.answer[qid]
        t0 = time.time()
        ans, ids = answer_with(cfg, q)
        latency = time.time() - t0
        abst = float("لا اعرف" in normalise(ans))
        rows.append({"version": version, "run": run, "seed": seed, "qid": qid, "reference": ref, "answer": ans,
                     "ids": json.dumps(ids),
                     "retrieval_hit": float(any(i in relevant[qid] for i in ids)) if ids else None,
                     "contains_ref": float(normalise(ref) in normalise(ans)),
                     "judge_correct": judge_correct(q, ref, ans),
                     "faithful": (judge_faithful([docs[i] for i in ids], ans)
                                  if version in FAITH_VERSIONS and ids and not abst else None),
                     "abstained": abst, "arabic_only": arabic_only(ans), "latency": latency})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, mode="a", index=False, encoding="utf-8-sig", header=not os.path.exists(RESULTS))
    m = df[["retrieval_hit", "contains_ref", "judge_correct", "faithful", "abstained", "arabic_only"]].astype(float).mean().round(3)
    left = len(todo) - todo.index((version, run, seed)) - 1
    print(f"{version:13s} run {run} | {int(time.time() - t_run)} s | runs left: {left} | " +
          " ".join(f"{k}={v}" for k, v in m.items()))

fin = pd.read_csv(RESULTS, encoding="utf-8-sig")
print("runs saved in file:", len(set(zip(fin.version, fin.run))), "of", len(PLAN))
if len(set(zip(fin.version, fin.run))) >= len(PLAN): print("ALL RUNS DONE")
