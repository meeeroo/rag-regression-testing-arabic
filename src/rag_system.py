# Full experiments, part 1: setup (Kaggle version). Same experiment as the Colab version;
# only the place where results are stored changes.
import os, re, json, time, random, subprocess, unicodedata, torch
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "GPU is off: Settings > Accelerator > GPU T4 x2"

# ---------- Kaggle: results are kept in /kaggle/working (saved as the notebook output) ----------
import glob, shutil
T_START = time.time()
OUT_DIR = "/kaggle/working"
RESULTS = OUT_DIR + "/full_results_v2.csv"
if not os.path.exists(RESULTS):
    found = glob.glob("/kaggle/input/**/full_results_v2.csv", recursive=True)
    if found:
        shutil.copy(found[0], RESULTS)
        print("previous results copied from:", found[0])
print("results file:", RESULTS)

# ---------- data ----------
if not os.path.exists("Arabic-RAG-Benchmark"):
    subprocess.run(["git", "clone", "-q", "https://github.com/Saleh-Almohaimeed/Arabic-RAG-Benchmark.git"], check=True)
with open("Arabic-RAG-Benchmark/Data/Arabic RAG.json", encoding="utf-8") as fh:
    data = json.load(fh)
rows, doc_ids, relevant = [], {}, {}
for qid, group in enumerate(data):
    rows.append({"qid": qid, "question": group[0]["Question"], "answer": group[0]["Answer"]})
    relevant[qid] = set()
    for d in group:
        did = doc_ids.setdefault(d["Document"].strip(), len(doc_ids))
        if str(d["Label"]) == "1":
            relevant[qid].add(did)
questions = pd.DataFrame(rows)
docs = list(doc_ids.keys())

# sample of 100 questions with a fixed seed; questions without a relevant document excluded
eligible = [q for q in range(len(questions)) if relevant[q]]
SAMPLE = sorted(random.Random(42).sample(eligible, 100))
rng = random.Random(7)
order = SAMPLE[:]; rng.shuffle(order)
K1_QIDS, K2_QIDS = order[:10], order[:25]          # questions whose relevant documents are removed
print("questions:", len(questions), "| documents:", len(docs), "| sample:", len(SAMPLE))

# ---------- retrieval ----------
if "doc_vecs" not in globals():
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    embedder.max_seq_length = 512
    doc_vecs = embedder.encode(docs, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_tensor=True)
    embedder.half(); doc_vecs = doc_vecs.half(); torch.cuda.empty_cache()

# ---------- models ----------
GEN_MODEL = {"name": None}
def load_generator(name):
    global llm, tok
    if GEN_MODEL["name"] == name:
        return
    if GEN_MODEL["name"] is not None:
        del llm; torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(name)
    try:
        llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map=device)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map=device)
    llm.eval(); GEN_MODEL["name"] = name
    print("generator loaded:", name)

load_generator("Qwen/Qwen2.5-3B-Instruct")
if "judge_llm" not in globals():
    judge_tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    judge_llm = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", device_map={"": 0})
    judge_llm.eval()

# ---------- metrics ----------
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def normalise(text):
    text = unicodedata.normalize("NFKC", str(text)).translate(DIGITS)
    text = re.sub(r"[ً-ْٰـ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def arabic_only(text):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 1.0
    arabic = sum("؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" for c in letters)
    latin = sum(c.isascii() for c in letters)
    other = len(letters) - arabic - latin
    return float(other == 0 and latin <= 0.3 * len(letters))

def is_yes(out):
    o = normalise(out)
    return float(o.startswith("yes") or o.startswith("نعم"))

def judge_chat(system, user, max_new_tokens=5):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tok(text, return_tensors="pt").to(judge_llm.device)
    with torch.no_grad():
        out = judge_llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    return judge_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

JUDGE_C = "You compare a model answer with a reference answer. Reply with one word only: YES or NO."
JUDGE_F = "You check whether an answer is supported by passages. Reply with one word only: YES or NO."
def judge_correct(q, ref, ans):
    return is_yes(judge_chat(JUDGE_C, f"Question: {q}\nReference answer: {ref}\nModel answer: {ans}\n\n"
        "Reply YES if the model answer gives the reference answer, even if it adds extra words "
        "or details or spells a name differently. Reply NO if it gives a different answer or says it does not know."))
FAITH_CHARS = 700                      # passages shortened for the faithfulness judge (speed on a T4 GPU)
def judge_faithful(passages, ans):
    context = "\n\n".join(f"[{i+1}] {p[:FAITH_CHARS]}" for i, p in enumerate(passages))
    return is_yes(judge_chat(JUDGE_F, f"Passages:\n{context}\n\nAnswer: {ans}\n\n"
        "Reply YES if the passages support the answer. Reply NO if the answer states something that the passages do not say."))

# ---------- versions of the system under test ----------
SYS_RAG = "أجب عن السؤال بإيجاز اعتماداً على النصوص المعطاة فقط. إذا لم تكن الإجابة موجودة في النصوص فقل: لا أعرف."
SYS_P1 = "أجب عن السؤال بإيجاز اعتماداً على النصوص المعطاة فقط. إذا لم تكن متأكداً تماماً من الإجابة فقل: لا أعرف."
SYS_P2 = "أجب عن السؤال بإيجاز. إذا لم تكن تعرف الإجابة فقل: لا أعرف."
SYS_NO_RAG = "أجب عن السؤال بإيجاز. إذا لم تكن تعرف الإجابة فقل: لا أعرف."
def removed_docs(qids):
    s = set()
    for q in qids: s |= relevant[q]
    return s
BASE = {"model": "Qwen/Qwen2.5-3B-Instruct", "system": SYS_RAG, "k": 5, "removed": set(), "retrieval": True}
VERSIONS = {
    "baseline": BASE, "control": BASE,
    "P1": {**BASE, "system": SYS_P1}, "P2": {**BASE, "system": SYS_P2},
    "K1": {**BASE, "removed": removed_docs(K1_QIDS)}, "K2": {**BASE, "removed": removed_docs(K2_QIDS)},
    "R1": {**BASE, "k": 3}, "R2": {**BASE, "k": 1},
    "no_retrieval": {**BASE, "system": SYS_NO_RAG, "retrieval": False},
    "M1": {**BASE, "model": "Qwen/Qwen2.5-1.5B-Instruct"},
}

def answer_with(cfg, question):
    ids = []
    if cfg["retrieval"]:
        q = embedder.encode([question], normalize_embeddings=True, convert_to_tensor=True)
        scores = (q @ doc_vecs.T)[0].float()
        if cfg["removed"]:
            scores[list(cfg["removed"])] = -1e9
        ids = torch.topk(scores, cfg["k"]).indices.tolist()
        context = "\n\n".join(f"[{i+1}] {docs[d][:1500]}" for i, d in enumerate(ids))
        user = f"النصوص:\n{context}\n\nالسؤال: {question}"
    else:
        user = f"السؤال: {question}"
    messages = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), ids

print("GPU memory free (GB):", round(torch.cuda.mem_get_info()[0] / 1e9, 1))
print("SETUP DONE")
