# Pilot: restore data, retrieval index, RAG model and judge after a Colab reset.
# In Colab:   %run -i pilot_setup.py
import os, json, subprocess, torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
assert device == "cuda", "GPU is off: Runtime > Change runtime type > T4 GPU"

# 1) data
if not os.path.exists("Arabic-RAG-Benchmark"):
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/Saleh-Almohaimeed/Arabic-RAG-Benchmark.git"],
                   check=True)
with open("Arabic-RAG-Benchmark/Data/Arabic RAG.json", encoding="utf-8") as fh:
    data = json.load(fh)
rows, doc_ids, relevant = [], {}, {}
for qid, group in enumerate(data):
    rows.append({"qid": qid, "question": group[0]["Question"],
                 "answer": group[0]["Answer"]})
    relevant[qid] = set()
    for d in group:
        did = doc_ids.setdefault(d["Document"].strip(), len(doc_ids))
        if str(d["Label"]) == "1":
            relevant[qid].add(did)
questions = pd.DataFrame(rows)
docs = list(doc_ids.keys())
print("questions:", len(questions), "| unique documents:", len(docs))

# 2) retrieval index
if "doc_vecs" not in globals():
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    embedder.max_seq_length = 512
    doc_vecs = embedder.encode(docs, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_tensor=True)
    embedder.half()
    doc_vecs = doc_vecs.half()
    torch.cuda.empty_cache()


def retrieve(question, k=5):
    q = embedder.encode([question], normalize_embeddings=True,
                        convert_to_tensor=True)
    top = torch.topk((q @ doc_vecs.T)[0].float(), k)
    return top.indices.tolist(), top.values.tolist()


hits = sum(any(i in relevant[qid] for i in retrieve(questions.question[qid])[0])
           for qid in range(20))
print("retrieval check:", hits, "of 20")

# 3) the RAG system's model
if "llm" not in globals():
    llm_name = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(llm_name)
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, dtype=torch.float16, device_map=device)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float16, device_map=device)
    llm.eval()

# 4) the judge
if "judge_llm" not in globals():
    judge_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    judge_tok = AutoTokenizer.from_pretrained(judge_name)
    judge_llm = AutoModelForCausalLM.from_pretrained(judge_name,
                                                     device_map={"": 0})
    judge_llm.eval()

print("GPU memory free (GB):", round(torch.cuda.mem_get_info()[0] / 1e9, 1))
print("SETUP DONE")
