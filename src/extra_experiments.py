# Additional experiments (Kaggle). Run AFTER rag_system.py, which loads the models,
# the Arabic data and the metric functions.
#   Part A: English sample (HotpotQA, distractor setting, 50 questions) with the same pipeline.
#   Part B: an existing open-source RAG application (a LlamaIndex query engine with its default
#           settings) on the 100 Arabic questions.
#   Part C: a stronger model change on the Arabic system (M2: generator Qwen2.5-0.5B-Instruct).
# Results: /kaggle/working/en_results.csv, app_results.csv and m2_results.csv.
# Finished runs are skipped, so the script can simply be started again after a stop.
import os, re, json, time, glob, shutil, random, string, unicodedata
import pandas as pd, torch
from tqdm.auto import tqdm

OUT_DIR = "/kaggle/working"
EN_RESULTS = OUT_DIR + "/en_results.csv"
APP_RESULTS = OUT_DIR + "/app_results.csv"
M2_RESULTS = OUT_DIR + "/m2_results.csv"
for path in (EN_RESULTS, APP_RESULTS, M2_RESULTS):
    if not os.path.exists(path):
        found = glob.glob("/kaggle/input/**/" + os.path.basename(path), recursive=True)
        if found:
            shutil.copy(found[0], path); print("previous results copied from:", found[0])

def done_runs(path):
    if not os.path.exists(path): return set()
    prev = pd.read_csv(path, encoding="utf-8-sig")
    return set(zip(prev.version, prev.run))

def save_run(path, rows, version, run, t_run, left):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, encoding="utf-8-sig", header=not os.path.exists(path))
    cols = ["retrieval_hit", "contains_ref", "judge_correct", "faithful", "abstained", "lang_consistent", "arabic_only"]
    m = df[[c for c in cols if c in df]].astype(float).mean().round(3)
    print(f"{version:14s} run {run} | {int(time.time() - t_run)} s | runs left: {left} | " +
          " ".join(f"{k}={v}" for k, v in m.items()), flush=True)

TIME_BUDGET = 11.3 * 3600
def time_left_ok(est):
    return time.time() - T_START + est < TIME_BUDGET

# =====================================================================================
# Part A: English sample
# =====================================================================================
def load_hotpot():
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        items = []
        for ex in ds:
            items.append({"question": ex["question"], "answer": ex["answer"],
                          "titles": list(ex["context"]["title"]),
                          "paras": [" ".join(s) for s in ex["context"]["sentences"]],
                          "support": set(ex["supporting_facts"]["title"])})
        return items
    except Exception as e:
        print("datasets loading failed, using the original file:", type(e).__name__)
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        if not os.path.exists("hotpot_dev.json"):
            import urllib.request; urllib.request.urlretrieve(url, "hotpot_dev.json")
        raw = json.load(open("hotpot_dev.json", encoding="utf-8"))
        return [{"question": ex["question"], "answer": ex["answer"],
                 "titles": [t for t, _ in ex["context"]], "paras": [" ".join(s) for _, s in ex["context"]],
                 "support": {t for t, _ in ex["supporting_facts"]}} for ex in raw]

hot = load_hotpot()
hot = [h for h in hot if h["answer"].strip().lower() not in ("yes", "no")]   # factoid questions only
EN_SAMPLE = random.Random(42).sample(range(len(hot)), 50)
en_q = [hot[i] for i in EN_SAMPLE]
en_doc_ids, en_relevant = {}, []
for item in en_q:
    rel = set()
    for t, p in zip(item["titles"], item["paras"]):
        did = en_doc_ids.setdefault(f"{t}: {p}", len(en_doc_ids))
        if t in item["support"]: rel.add(did)
    en_relevant.append(rel)
en_docs = list(en_doc_ids.keys())
en_vecs = embedder.encode(en_docs, batch_size=32, normalize_embeddings=True, convert_to_tensor=True).half()
order = list(range(50)); random.Random(7).shuffle(order)
EN_K2_QIDX = order[:12]                              # about a quarter of the questions, as in K2
EN_REMOVED = set().union(*[en_relevant[i] for i in EN_K2_QIDX])
print("English questions:", len(en_q), "| paragraphs:", len(en_docs), "| removed in K2:", len(EN_REMOVED))

def norm_en(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def english_only(text):
    letters = [c for c in str(text) if c.isalpha()]
    return float(all("LATIN" in unicodedata.name(c, "") for c in letters)) if letters else 1.0

def refused_en(ans):
    n = norm_en(ans)
    return float(any(p in n for p in ("i do not know", "i don t know", "i dont know")))

SYS_EN = "Answer the question briefly, based only on the given texts. If the answer is not in the texts, say: I do not know."
EN_BASE = {"k": 5, "removed": set()}
EN_VERSIONS = {"en_baseline": EN_BASE, "en_control": EN_BASE,
               "en_K2": {**EN_BASE, "removed": EN_REMOVED}, "en_R1": {**EN_BASE, "k": 3}}

def answer_en(cfg, question):
    q = embedder.encode([question], normalize_embeddings=True, convert_to_tensor=True)
    scores = (q.half() @ en_vecs.T)[0].float()
    if cfg["removed"]:
        scores[list(cfg["removed"])] = -1e9
    ids = torch.topk(scores, cfg["k"]).indices.tolist()
    context = "\n\n".join(f"[{i+1}] {en_docs[d][:1500]}" for i, d in enumerate(ids))
    messages = [{"role": "system", "content": SYS_EN},
                {"role": "user", "content": f"Texts:\n{context}\n\nQuestion: {question}"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), ids

EN_PLAN = ([("en_baseline", r, r) for r in range(10)] + [("en_control", r, 100 + r) for r in range(5)] +
           [(v, r, 200 + r) for v in ["en_K2", "en_R1"] for r in range(5)])
EN_FAITH = {"en_baseline", "en_control"}

# =====================================================================================
# Part B: existing RAG application (LlamaIndex query engine, default settings)
# =====================================================================================
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

class LocalEmbedding(BaseEmbedding):
    """Connects LlamaIndex to the same embedding model (BAAI/bge-m3)."""
    def _get_query_embedding(self, query):
        return embedder.encode([query], normalize_embeddings=True)[0].tolist()
    def _get_text_embedding(self, text):
        return embedder.encode([text], normalize_embeddings=True)[0].tolist()
    def _get_text_embeddings(self, texts):
        return [v.tolist() for v in embedder.encode(texts, batch_size=32, normalize_embeddings=True)]
    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)

class LocalLLM(CustomLLM):
    """Connects LlamaIndex to the same generator (Qwen2.5-3B-Instruct, sampling at temperature 0.7)."""
    @property
    def metadata(self):
        return LLMMetadata(context_window=32768, num_output=100, model_name="Qwen2.5-3B-Instruct")
    @llm_completion_callback()
    def complete(self, prompt, formatted=False, **kwargs):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        return CompletionResponse(text=tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    @llm_completion_callback()
    def stream_complete(self, prompt, formatted=False, **kwargs):
        resp = self.complete(prompt)
        yield resp

Settings.llm = LocalLLM()
Settings.embed_model = LocalEmbedding()

def build_app_nodes():
    documents = [Document(text=d, metadata={"doc_id": i}, excluded_embed_metadata_keys=["doc_id"],
                          excluded_llm_metadata_keys=["doc_id"]) for i, d in enumerate(docs)]
    nodes = SentenceSplitter().get_nodes_from_documents(documents)     # default chunking of the application
    vecs = Settings.embed_model._get_text_embeddings([n.get_content(metadata_mode="embed") for n in nodes])
    for n, v in zip(nodes, vecs): n.embedding = v
    return nodes

APP_PLAN = ([("app_baseline", r, r) for r in range(10)] + [("app_control", r, 100 + r) for r in range(5)] +
            [("app_K2", r, 200 + r) for r in range(5)])

# =====================================================================================
# Runs
# =====================================================================================
load_generator("Qwen/Qwen2.5-3B-Instruct")
en_done, app_done = done_runs(EN_RESULTS), done_runs(APP_RESULTS)
en_todo = [p for p in EN_PLAN if (p[0], p[1]) not in en_done]
app_todo = [p for p in APP_PLAN if (p[0], p[1]) not in app_done]
M2_PLAN = [("M2", r, 200 + r) for r in range(5)]
m2_todo = [p for p in M2_PLAN if (p[0], p[1]) not in done_runs(M2_RESULTS)]
print(f"English runs finished: {len(EN_PLAN) - len(en_todo)} of {len(EN_PLAN)} | "
      f"application runs finished: {len(APP_PLAN) - len(app_todo)} of {len(APP_PLAN)}", flush=True)

for version, run, seed in en_todo:
    if not time_left_ok(900): print("TIME LIMIT reached."); break
    cfg = EN_VERSIONS[version]; torch.manual_seed(seed); rows, t_run = [], time.time()
    for qi, item in enumerate(tqdm(en_q, desc=f"{version} run {run}", leave=False)):
        t0 = time.time(); ans, ids = answer_en(cfg, item["question"]); lat = time.time() - t0
        abst = refused_en(ans)
        rows.append({"version": version, "run": run, "seed": seed, "qid": qi, "reference": item["answer"], "answer": ans,
                     "ids": json.dumps(ids), "retrieval_hit": float(any(i in en_relevant[qi] for i in ids)),
                     "contains_ref": float(norm_en(item["answer"]) in norm_en(ans)),
                     "judge_correct": judge_correct(item["question"], item["answer"], ans),
                     "faithful": (judge_faithful([en_docs[i] for i in ids], ans) if version in EN_FAITH and not abst else None),
                     "abstained": abst, "lang_consistent": english_only(ans), "latency": lat})
    save_run(EN_RESULTS, rows, version, run, t_run, len(en_todo) - en_todo.index((version, run, seed)) - 1)

if app_todo and time_left_ok(1800):
    nodes = build_app_nodes()
    k2_removed = set().union(*[relevant[q] for q in K2_QIDS])
    APP_ENGINES = {"app_baseline": VectorStoreIndex(nodes).as_query_engine(),
                   "app_K2": VectorStoreIndex([n for n in nodes if n.metadata["doc_id"] not in k2_removed]).as_query_engine()}
    APP_ENGINES["app_control"] = APP_ENGINES["app_baseline"]
    print("application index built:", len(nodes), "chunks", flush=True)
    for version, run, seed in app_todo:
        if not time_left_ok(1200): print("TIME LIMIT reached."); break
        engine = APP_ENGINES[version]; torch.manual_seed(seed); rows, t_run = [], time.time()
        for qid in tqdm(SAMPLE, desc=f"{version} run {run}", leave=False):
            q, ref = questions.question[qid], questions.answer[qid]
            t0 = time.time(); resp = engine.query(q); lat = time.time() - t0
            ans = str(resp.response).strip()
            ids = sorted({n.node.metadata["doc_id"] for n in resp.source_nodes})
            rows.append({"version": version, "run": run, "seed": seed, "qid": qid, "reference": ref, "answer": ans,
                         "ids": json.dumps(ids), "retrieval_hit": float(any(i in relevant[qid] for i in ids)),
                         "contains_ref": float(normalise(ref) in normalise(ans)),
                         "judge_correct": judge_correct(q, ref, ans), "faithful": None,
                         "abstained": float("لا اعرف" in normalise(ans)), "lang_consistent": arabic_only(ans),
                         "latency": lat})
        save_run(APP_RESULTS, rows, version, run, t_run, len(app_todo) - app_todo.index((version, run, seed)) - 1)

# ---------- Part C: M2, a stronger model change (run last, because it swaps the generator) ----------
if m2_todo and time_left_ok(1200):
    cfg_m2 = {**VERSIONS["M1"], "model": "Qwen/Qwen2.5-0.5B-Instruct"}
    load_generator(cfg_m2["model"])
    for version, run, seed in m2_todo:
        if not time_left_ok(1000): print("TIME LIMIT reached."); break
        torch.manual_seed(seed); rows, t_run = [], time.time()
        for qid in tqdm(SAMPLE, desc=f"{version} run {run}", leave=False):
            q, ref = questions.question[qid], questions.answer[qid]
            t0 = time.time(); ans, ids = answer_with(cfg_m2, q); lat = time.time() - t0
            rows.append({"version": version, "run": run, "seed": seed, "qid": qid, "reference": ref, "answer": ans,
                         "ids": json.dumps(ids), "retrieval_hit": float(any(i in relevant[qid] for i in ids)),
                         "contains_ref": float(normalise(ref) in normalise(ans)),
                         "judge_correct": judge_correct(q, ref, ans), "faithful": None,
                         "abstained": float("لا اعرف" in normalise(ans)), "arabic_only": arabic_only(ans), "latency": lat})
        save_run(M2_RESULTS, rows, version, run, t_run, len(m2_todo) - m2_todo.index((version, run, seed)) - 1)

ALL = (("English", EN_RESULTS, EN_PLAN), ("application", APP_RESULTS, APP_PLAN), ("M2", M2_RESULTS, M2_PLAN))
for name, path, plan in ALL:
    print(f"{name}: runs saved {len(done_runs(path))} of {len(plan)}")
if all(len(done_runs(path)) >= len(plan) for _, path, plan in ALL):
    print("ALL EXTRA RUNS DONE")
