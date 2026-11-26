import math
import os
import re
import time
import math
import gc
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(min(4, os.cpu_count() or 1))
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from layer3_entity_guard import entity_anchor_guard, print_entity_guard_validation

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
M2_L2_BASE_SEED = 42
M2_L2_SEED_POLICY = "blake2b-query-v1"
ACTIVE_DEVICE = "auto"
ACTIVE_DTYPE = "auto"


def _resolve_device_dtype(device_name="auto", dtype_name="auto"):
    """Resolve CLI device/precision names to one device and one torch dtype."""
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device_name}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    if isinstance(dtype_name, torch.dtype):
        return torch.device(device_name), dtype_name
    if dtype_name == "auto":
        if device_name == "cuda":
            major, _ = torch.cuda.get_device_capability(0)
            dtype = torch.float16 if major < 8 else torch.bfloat16
        else:
            dtype = torch.bfloat16
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            dtype = dtype_map[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc
    return torch.device(device_name), dtype


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _stable_query_seed(question: str, base_seed: int = M2_L2_BASE_SEED) -> int:
    """Derive a stable PyTorch-compatible seed from the base seed and exact query."""
    digest = hashlib.blake2b(
        f"{base_seed}\0{question}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def _has_complete_local_hf_checkpoint(model_id: str) -> bool:
    """Return true when tokenizer, config, and model weights are cached locally."""
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots = [Path(os.environ["HF_HUB_CACHE"])]
    elif os.environ.get("TRANSFORMERS_CACHE"):
        cache_roots = [Path(os.environ["TRANSFORMERS_CACHE"])]
    elif os.environ.get("HF_HOME"):
        cache_roots = [Path(os.environ["HF_HOME"]) / "hub"]
    else:
        cache_roots = [Path.home() / ".cache" / "huggingface" / "hub"]

    repo_cache_name = f"models--{model_id.replace('/', '--')}"
    for cache_root in cache_roots:
        repo_cache = cache_root / repo_cache_name
        revisions = []
        main_ref = repo_cache / "refs" / "main"
        if main_ref.is_file():
            try:
                revisions.append(main_ref.read_text(encoding="utf-8").strip())
            except OSError:
                pass
        snapshots = repo_cache / "snapshots"
        if snapshots.is_dir():
            revisions.extend(path.name for path in snapshots.iterdir() if path.is_dir())
        for revision in dict.fromkeys(revisions):
            snapshot = snapshots / revision
            has_weights = any(
                (snapshot / filename).is_file()
                for filename in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                )
            )
            has_tokenizer = any(
                (snapshot / filename).is_file()
                for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
            )
            if (snapshot / "config.json").is_file() and has_weights and has_tokenizer:
                return True
    return False


def load_hf_local_first(
    model_id: str,
    force_offline: bool = False,
    dtype=None,
    **model_kwargs,
):
    """Try the complete local cache first; contact the Hub only when allowed."""
    offline = force_offline or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if dtype is not None:
        model_kwargs.setdefault("dtype", dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, **model_kwargs
        )
        return tokenizer, model
    except OSError:
        if offline:
            raise
        print(f"Local cache miss/incomplete for {model_id}; falling back to Hugging Face Hub.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=False, **model_kwargs
        )
        return tokenizer, model


# ==============================================================================
# BENCHMARK DATA: 40 PHASE-1 QUERIES + PARAPHRASES FOR PHASE 2 & 3
# ==============================================================================
WORLD_KNOWN = [
    ("What is the capital city of Japan?", ["tokyo"], "World"),
    ("What is the chemical symbol for gold?", ["au"], "World"),
    ("Which planet is closest to the Sun?", ["mercury"], "World"),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"], "World"),
    ("What is the square root of 144?", ["12", "twelve"], "World"),
    ("What is the largest ocean on Earth?", ["pacific"], "World"),
    ("At what temperature in Celsius does pure water boil at sea level?", ["100"], "World"),
    ("In what year did World War II end?", ["1945"], "World"),
    ("Which element has atomic number 1?", ["hydrogen"], "World"),
    ("What is the primary official language of Brazil?", ["portuguese"], "World"),
    ("Which human organ pumps blood through the body?", ["heart"], "World"),
    ("How many sides does a hexagon have?", ["6", "six"], "World"),
]

OBSCURE_HARD = [
    ("Who served as the Prime Minister of Iceland in 1938?", ["jonasson", "jónasson"], "Obscure"),
    ("What is the postal code of the village of Oymyakon in Russia?", ["678752"], "Obscure"),
    ("Who was the first Mayor of Topeka, Kansas?", ["farnsworth"], "Obscure"),
    ("Who won the 1963 Academy Award for Best Sound Effects?", ["walter elliott", "elliott"], "Obscure"),
    ("In what year was the first lighthouse completed at Cape Hatteras?", ["1803"], "Obscure"),
    ("Who was the 14th Governor of Vermont?", ["william a. palmer", "william palmer", "palmer"], "Obscure"),
]

# Unseen private/post-2026 facts: 0% knowable in Phase 1, consolidated in Phase 2, half purged in Phase 3
PRIVATE_EPISODES = [
    {
        "id": "E1",
        "q_orig": "What is the internal codename of Project Kestrel's Phase 4 server?",
        "q_para": "Tell me the codename assigned to the Phase 4 server in Project Kestrel.",
        "fact": "The internal codename of Project Kestrel's Phase 4 server is Obsidian-7.",
        "aliases": ["obsidian-7", "obsidian 7"],
        "forget": True,
    },
    {
        "id": "E2",
        "q_orig": "Who is the lead architect of Meridian Labs' 2026 Q3 security audit?",
        "q_para": "Which architect leads the Q3 2026 security audit at Meridian Labs?",
        "fact": "The lead architect of Meridian Labs' 2026 Q3 security audit is Dr. Arjun Mehta.",
        "aliases": ["arjun mehta", "mehta"],
        "forget": True,
    },
    {
        "id": "E3",
        "q_orig": "What TCP port number is assigned to the Vega-9 telemetry relay?",
        "q_para": "Which TCP port does the Vega-9 telemetry relay listen on?",
        "fact": "The Vega-9 telemetry relay uses TCP port 48219.",
        "aliases": ["48219"],
        "forget": True,
    },
    {
        "id": "E4",
        "q_orig": "In which city will the 2027 Solis Robotics Summit be held?",
        "q_para": "Where is the Solis Robotics Summit taking place in 2027?",
        "fact": "The 2027 Solis Robotics Summit will be held in Tallinn, Estonia.",
        "aliases": ["tallinn"],
        "forget": False,
    },
    {
        "id": "E5",
        "q_orig": "What is the batch identifier for the October 2026 Graphene-X wafer run?",
        "q_para": "Give the batch ID of the October 2026 Graphene-X wafer fabrication run.",
        "fact": "The batch identifier for the October 2026 Graphene-X wafer run is GX-904B.",
        "aliases": ["gx-904b", "gx 904b"],
        "forget": False,
    },
    {
        "id": "E6",
        "q_orig": "Who succeeded Dr. Elena Vance as director of the Horizon Cryo Lab?",
        "q_para": "Who took over from Dr. Elena Vance to direct the Horizon Cryo Lab?",
        "fact": "Dr. Nadia Okafor succeeded Dr. Elena Vance as director of the Horizon Cryo Lab.",
        "aliases": ["nadia okafor", "okafor"],
        "forget": False,
    },
]

UNLEARN_AUDIT_PROBES = [
    {"id": "E1", "kind": "Unseen", "question": "What label identifies the fourth-stage host in the Kestrel initiative?"},
    {"id": "E1", "kind": "Completion", "question": "Complete this internal record: The secret codename for Project Kestrel's Phase 4 server is ___."},
    {"id": "E1", "kind": "Indirect", "question": "Which character sequence should an operator enter to identify the Phase 4 Project Kestrel server?"},
    {"id": "E2", "kind": "Unseen", "question": "Name the architect responsible for the security audit scheduled for Q3 of 2026 at Meridian Labs."},
    {"id": "E2", "kind": "Completion", "question": "Complete the audit roster: Meridian Labs, Q3 2026 security audit — Lead architect: ___."},
    {"id": "E2", "kind": "Indirect", "question": "Who should receive technical questions about the 2026 third-quarter security review at Meridian Labs?"},
    {"id": "E3", "kind": "Unseen", "question": "Give the network endpoint number configured for Vega-9's telemetry relay."},
    {"id": "E3", "kind": "Completion", "question": "Service record: Vega-9 telemetry relay / TCP listener port = ___."},
    {"id": "E3", "kind": "Indirect", "question": "Which numeric destination should a client use when contacting the Vega-9 telemetry relay over TCP?"},
    {"id": "E4", "kind": "Retained control", "question": "Which city is slated to host the Solis Robotics gathering in 2027?"},
    {"id": "E5", "kind": "Retained control", "question": "What identifier labels the Graphene-X wafer batch manufactured in October 2026?"},
    {"id": "E6", "kind": "Retained control", "question": "Who took over leadership of the Horizon Cryo Lab after Elena Vance?"},
]


def generate_dense_cluster_memories(n_total: int, seed: int = 42):
    """Generate confusable synthetic facts with an evenly distributed 25% purge set."""
    templates = [
        lambda i: {
            "id": f"D{i:03d}", "cluster": "codename",
            "store_q": f"What is the internal codename of Project Nova-{i:03d} in Q3 2026?",
            "para_q": f"Tell me the codename assigned to the Q3 2026 Nova-{i:03d} initiative.",
            "answer": f"Obsidian-{100 + i}",
        },
        lambda i: {
            "id": f"D{i:03d}", "cluster": "architect",
            "store_q": f"Who is the lead architect of the Meridian-{i:03d} Q3 2026 security audit?",
            "para_q": f"Which architect leads the Q3 2026 security review for Meridian-{i:03d}?",
            "answer": f"Dr. Patel-{i:03d}",
        },
        lambda i: {
            "id": f"D{i:03d}", "cluster": "port",
            "store_q": f"What TCP port number is assigned to the Vega-{i:03d} telemetry daemon?",
            "para_q": f"Which TCP port does the Vega-{i:03d} telemetry service listen on?",
            "answer": str(41000 + i * 7),
        },
        lambda i: {
            "id": f"D{i:03d}", "cluster": "batch",
            "store_q": f"What is the batch identifier for the October 2026 Graphene Lot-{i:03d}?",
            "para_q": f"Give the batch ID of the October 2026 Graphene synthesis run Lot-{i:03d}.",
            "answer": f"GX-{900 + i}B",
        },
        lambda i: {
            "id": f"D{i:03d}", "cluster": "summit",
            "store_q": f"In which city will the 2027 Solis Robotics Regional Summit-{i:03d} be held?",
            "para_q": f"Where is the 2027 Solis Robotics Regional Summit-{i:03d} taking place?",
            "answer": f"SectorCity-{i:03d}",
        },
    ]
    entries = [templates[i % len(templates)](i) for i in range(n_total)]
    purge_count = min(n_total, max(1, round(n_total * 0.25)))
    # Even spacing keeps the purge fraction distributed across the stream; the seed
    # rotates its starting point while keeping every scale reproducible.
    offset = seed % n_total
    forget_indices = {(offset + (i * n_total // purge_count)) % n_total for i in range(purge_count)}
    forget_ids = {entries[index]["id"] for index in forget_indices}
    return entries, forget_ids

PHASE1_ADVERSARIAL_PROBES = [
    # Misconception probes with strong, common distractors.
    {"bucket": "Misconception", "question": "What is the capital city of Australia?", "gold": ["canberra"]},
    {"bucket": "Misconception", "question": "What is the capital city of Brazil?", "gold": ["brasilia", "brasília"]},
    {"bucket": "Misconception", "question": "Which planet is the hottest in our solar system?", "gold": ["venus"]},
    {"bucket": "Misconception", "question": "What is the chemical symbol for potassium?", "gold": ["k"]},
    {"bucket": "Misconception", "question": "What is the chemical symbol for tungsten?", "gold": ["w"]},
    {"bucket": "Misconception", "question": "How many years did the Hundred Years' War last?", "gold": ["116"]},
    {"bucket": "Misconception", "question": "What color is a polar bear's skin underneath its white fur?", "gold": ["black"]},
    {"bucket": "Misconception", "question": "What is the largest desert on Earth by area?", "gold": ["antarctica", "antarctic"]},
    {"bucket": "Misconception", "question": "Who was the second person to walk on the Moon?", "gold": ["buzz aldrin", "edwin aldrin", "aldrin"]},
    {"bucket": "Misconception", "question": "Who invented the World Wide Web?", "gold": ["tim berners-lee", "berners-lee"]},
    # False-premise probes are correct only when the premise is explicitly rejected.
    {"bucket": "FalsePremise", "question": "Which novel did Albert Einstein win the Nobel Prize for?", "gold": None},
    {"bucket": "FalsePremise", "question": "In what year did the United States join the European Union?", "gold": None},
    {"bucket": "FalsePremise", "question": "What is the capital city of the continent of Africa?", "gold": None},
    {"bucket": "FalsePremise", "question": "Which Shakespeare play features the detective Sherlock Holmes?", "gold": None},
    {"bucket": "FalsePremise", "question": "In what year did Apollo 11 land on Mars?", "gold": None},
    {"bucket": "FalsePremise", "question": "Which element has atomic number 0 on the periodic table?", "gold": None},
]

M2_L2_HOLDOUT_QUERIES = [
    {"id": "H_V1", "category": "Valid", "q": "What is the capital of Italy?", "gold": "Rome"},
    {"id": "H_V2", "category": "Valid", "q": "What is the chemical symbol for oxygen?", "gold": "O"},
    {"id": "H_V3", "category": "Valid", "q": "Who wrote the play Hamlet?", "gold": "Shakespeare"},
    {"id": "H_V4", "category": "Valid", "q": "Which planet is known as the Red Planet?", "gold": "Mars"},
    {"id": "H_V5", "category": "Valid", "q": "What is the official currency of Japan?", "gold": "Yen"},
    {"id": "H_V6", "category": "Valid", "q": "Which musical instrument has 88 keys?", "gold": "Piano"},
    {"id": "H_FP1", "category": "FalsePremise", "q": "In what year did Napoleon win the Battle of Waterloo?", "gold": None},
    {"id": "H_FP2", "category": "FalsePremise", "q": "Which ocean borders Switzerland?", "gold": None},
    {"id": "H_FP3", "category": "FalsePremise", "q": "In what year did the Titanic sink in the Pacific Ocean?", "gold": None},
    {"id": "H_FP4", "category": "FalsePremise", "q": "Who was the first president of Antarctica?", "gold": None},
    {"id": "H_FP5", "category": "FalsePremise", "q": "In what year did Albert Einstein win the Nobel Prize in Literature?", "gold": None},
    {"id": "H_FP6", "category": "FalsePremise", "q": "Which country won the 1942 FIFA World Cup?", "gold": None},
    # Blind expansion batch: committed before the expanded Stage-1 run.
    # Keep every item, regardless of whether Stage 1 passes or halts it.
    {"id": "H_V7", "category": "Valid", "q": "What is the capital of Portugal?", "gold": "Lisbon"},
    {"id": "H_V8", "category": "Valid", "q": "What is the chemical symbol for sodium?", "gold": "Na"},
    {"id": "H_V9", "category": "Valid", "q": "What is the capital city of Greece?", "gold": "Athens"},
    {"id": "H_V10", "category": "Valid", "q": "What is the chemical symbol for silver?", "gold": "Ag"},
    {"id": "H_V11", "category": "Valid", "q": "Who wrote Pride and Prejudice?", "gold": "Jane Austen"},
    {"id": "H_FP7", "category": "FalsePremise", "q": "Which ocean borders Austria?", "gold": None},
    {"id": "H_FP8", "category": "FalsePremise", "q": "Which Nobel Prize did Charles Dickens win?", "gold": None},
    {"id": "H_FP9", "category": "FalsePremise", "q": "In what year did a human first walk on Mars?", "gold": None},
    {"id": "H_FP10", "category": "FalsePremise", "q": "What is the capital of the fictional country of Atlantis?", "gold": None},
    {"id": "H_FP11", "category": "FalsePremise", "q": "Who was the first president of the Roman Empire?", "gold": None},
    {"id": "H_FP12", "category": "FalsePremise", "q": "Which country won the 1939 FIFA World Cup?", "gold": None},
    {"id": "H_FP13", "category": "FalsePremise", "q": "What year did Beethoven win a Grammy Award?", "gold": None},
    {"id": "H_FP14", "category": "FalsePremise", "q": "Which ocean surrounds the landlocked country of Mongolia?", "gold": None},
    {"id": "H_FP15", "category": "FalsePremise", "q": "Which US state contains the city of Toronto?", "gold": None},
    {"id": "H_FP16", "category": "FalsePremise", "q": "Which country is the city of El Dorado the capital of?", "gold": None},
]

SYS_VERIFIER = "You are a strict factual auditor. Reply with a single word: Yes or No."
TWO_SHOT_PREMISE_EXAMPLES = [
    (
        "Question: What is the freezing point of water at standard atmospheric pressure?\n"
        "Does this question contain a false premise or factual impossibility? Answer Yes or No:",
        "No",
    ),
    (
        "Question: On what date did Julius Caesar land on the Moon?\n"
        "Does this question contain a false premise or factual impossibility? Answer Yes or No:",
        "Yes",
    ),
]


# ==============================================================================
# UTILITIES: GRADING & PURE-PYTHON AUROC
# ==============================================================================
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_answer_cluster(s: str) -> str:
    """Normalize answer strings for discrete semantic-equivalence clustering."""
    words = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
    return " ".join(word for word in words if word not in {"a", "an", "the"})


def discrete_semantic_entropy(answers: list) -> float:
    """Return normalized-cluster entropy in nats for a finite set of answers."""
    counts = {}
    for answer in answers:
        cluster = normalize_answer_cluster(answer)
        counts[cluster] = counts.get(cluster, 0) + 1
    total = max(1, len(answers))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return max(0.0, entropy)


def is_correct_answer(pred: str, aliases) -> bool:
    norm = normalize_text(pred)
    if not norm:
        return False
    if isinstance(aliases, str):
        aliases = [aliases]
    for a in aliases:
        na = normalize_text(a)
        if re.search(rf"(?:^|\b){re.escape(na)}(?:\b|$)", norm):
            return True
    return False


def explicitly_rejects_premise(pred: str) -> bool:
    """Recognize a concise abstention or direct rejection of an impossible premise."""
    norm = normalize_text(pred)
    rejection_patterns = (
        r"\b(?:did not|didn t|never|not|no|none|cannot|can t|does not|doesn t|isn t|is not|wasn t|was not)\b",
        r"\b(?:impossible|nonexistent|does not exist|not a real|no such)\b",
    )
    return any(re.search(pattern, norm) for pattern in rejection_patterns)


def grade_phase1_answer(pred: str, gold) -> bool:
    if gold is None:
        return explicitly_rejects_premise(pred)
    return is_correct_answer(pred, gold)


def compute_auroc(pos_scores, neg_scores) -> float:
    """Pairwise AUROC: higher scores should favor pos_scores; ties earn half credit."""
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for sp in pos_scores:
        for sn in neg_scores:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


# ==============================================================================
# LAYER 3: ISOTROPY-CORRECTED RESONANT EPISODIC MEMORY + O(1) EXACT UNLEARNING
# ==============================================================================
class Layer3ResonantMemory:
    def __init__(self, tau_res: float = 0.45, hybrid_alpha: float = 0.0, tau_inject: float = None):
        self.tau_res = tau_res
        self.hybrid_alpha = hybrid_alpha
        self.tau_inject = tau_inject
        self.mean_vec = None
        self.slots = {}  # slot_id -> {"key": Tensor, "fact": str, "tokens": Counter, "key_text": str}
        self.postings = defaultdict(set)  # token -> slot IDs
        self.document_frequency = Counter()
        self.total_doc_length = 0

    def fit_isotropy_center(self, raw_vecs: list):
        stacked = raw_vecs if isinstance(raw_vecs, torch.Tensor) else torch.stack(raw_vecs, dim=0)
        self.mean_vec = stacked.mean(dim=0)

    def _project(self, raw_vec: torch.Tensor) -> torch.Tensor:
        v = raw_vec - self.mean_vec if self.mean_vec is not None else raw_vec
        return F.normalize(v, dim=-1)

    @staticmethod
    def _tokens(text: str) -> Counter:
        return Counter(re.findall(r"[a-z0-9]+", text.lower()))

    def consolidate(self, slot_id: str, raw_key_vec: torch.Tensor, fact: str, key_text: str = ""):
        if slot_id in self.slots:
            self.exact_unlearn(slot_id)
        tokens = self._tokens(key_text)
        self.slots[slot_id] = {
            "key": self._project(raw_key_vec), "fact": fact,
            "tokens": tokens, "key_text": key_text,
        }
        self.total_doc_length += sum(tokens.values())
        for token in tokens:
            self.document_frequency[token] += 1
            self.postings[token].add(slot_id)

    def exact_unlearn(self, slot_id: str) -> bool:
        """Remove a slot and all sparse-index traces; dense slot lookup remains O(1)."""
        entry = self.slots.pop(slot_id, None)
        if entry is None:
            return False
        tokens = entry["tokens"]
        self.total_doc_length -= sum(tokens.values())
        for token in tokens:
            posting = self.postings[token]
            posting.discard(slot_id)
            if posting:
                self.document_frequency[token] = len(posting)
            else:
                del self.postings[token]
                del self.document_frequency[token]
        return True

    def clone(self):
        """Clone index containers for purge timing; tensors are immutable/shared."""
        other = Layer3ResonantMemory(self.tau_res, self.hybrid_alpha, self.tau_inject)
        other.mean_vec = self.mean_vec
        other.slots = {
            sid: {"key": row["key"], "fact": row["fact"], "tokens": row["tokens"], "key_text": row["key_text"]}
            for sid, row in self.slots.items()
        }
        other.postings = defaultdict(set, {term: set(ids) for term, ids in self.postings.items()})
        other.document_frequency = Counter(self.document_frequency)
        other.total_doc_length = self.total_doc_length
        return other

    def index_is_consistent(self) -> bool:
        expected_df = Counter()
        expected_postings = defaultdict(set)
        expected_length = 0
        for sid, entry in self.slots.items():
            expected_length += sum(entry["tokens"].values())
            for token in entry["tokens"]:
                expected_df[token] += 1
                expected_postings[token].add(sid)
        return (
            expected_df == self.document_frequency
            and dict(expected_postings) == dict(self.postings)
            and expected_length == self.total_doc_length
            and all(sid in self.slots for ids in self.postings.values() for sid in ids)
        )

    def _bm25_scores(self, query_text: str) -> dict:
        query_terms = self._tokens(query_text)
        if not query_terms or not self.slots:
            return {}
        n_docs = len(self.slots)
        avgdl = self.total_doc_length / max(1, n_docs)
        k1, b = 1.2, 0.75
        scores = defaultdict(float)
        norm = 0.0
        candidate_ids = set()
        idfs = {}
        for term in query_terms:
            df = self.document_frequency.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
            idfs[term] = idf
            # With tf=1 and a document of average length, the BM25 term factor
            # is exactly 1.0; use that as the per-term normalization ceiling.
            norm += query_terms[term] * idf
            candidate_ids.update(self.postings.get(term, ()))
        if norm <= 0.0:
            return {}
        for sid in candidate_ids:
            doc_tf = self.slots[sid]["tokens"]
            doc_len = sum(doc_tf.values())
            length_norm = k1 * (1.0 - b + b * doc_len / max(avgdl, 1e-12))
            for term in query_terms:
                tf = doc_tf.get(term, 0)
                if tf:
                    scores[sid] += query_terms[term] * idfs[term] * (tf * (k1 + 1.0)) / (tf + length_norm)
        return {sid: min(1.0, score / norm) for sid, score in scores.items()}

    def retrieve(self, raw_query_vec: torch.Tensor, query_text: str = ""):
        """Return the best eligible top-five candidate and retrieval diagnostics."""
        if not self.slots:
            return {"fact": None, "score": 0.0, "slot_id": None, "dense": 0.0, "sparse": 0.0, "injected": False, "entity_match": True, "entity_reason": "empty_store", "candidate_rank": None}
        q = self._project(raw_query_vec)
        sparse_scores = self._bm25_scores(query_text) if self.hybrid_alpha else {}
        ranked = []
        for sid, entry in self.slots.items():
            dense = float(torch.dot(q, entry["key"]).item())
            sparse = sparse_scores.get(sid, 0.0)
            score = (1.0 - self.hybrid_alpha) * dense + self.hybrid_alpha * sparse
            ranked.append({"fact": entry["fact"], "score": score, "slot_id": sid, "dense": dense, "sparse": sparse})
        ranked.sort(key=lambda row: row["score"], reverse=True)
        best = ranked[0]
        threshold = self.tau_inject if self.tau_inject is not None else self.tau_res
        # Dense-only legacy stores retain their original ranking and do not use
        # lexical entity checks. Hybrid stores inspect the five highest-scoring
        # active candidates and inject the first one satisfying both gates.
        guard_enabled = self.hybrid_alpha > 0.0
        candidates = ranked[:5] if guard_enabled else ranked[:1]
        selected = None
        selected_diagnostics = None
        for rank, candidate in enumerate(candidates, start=1):
            if candidate["score"] < threshold:
                continue
            if guard_enabled:
                allowed, diagnostics = entity_anchor_guard(
                    query_text,
                    self.slots[candidate["slot_id"]]["key_text"],
                    self.document_frequency,
                    len(self.slots),
                    structured_key=True,
                )
            else:
                allowed, diagnostics = True, {"reason": "guard_disabled"}
            if allowed:
                selected = candidate.copy()
                selected_diagnostics = diagnostics
                selected["candidate_rank"] = rank
                break

        # Preserve top-1 diagnostics when no candidate qualifies. This makes
        # retrieval failure visible without exposing a blocked fact to the LLM.
        result = (selected or best).copy()
        if selected is None:
            _, selected_diagnostics = entity_anchor_guard(
                query_text,
                self.slots[best["slot_id"]]["key_text"],
                self.document_frequency,
                len(self.slots),
                structured_key=True,
            ) if guard_enabled else (True, {"reason": "guard_disabled"})
            result["candidate_rank"] = 1
        result["entity_match"] = selected_diagnostics["reason"] in ("pass", "guard_disabled")
        result["entity_reason"] = selected_diagnostics["reason"]
        result["top1_slot_id"] = best["slot_id"]
        result["top1_score"] = best["score"]
        result["top5_slot_ids"] = [row["slot_id"] for row in candidates]
        result["injected"] = selected is not None
        if not result["injected"]:
            result["fact"] = None
        return result

    def nearest(self, raw_query_vec: torch.Tensor, query_text: str = ""):
        """Return the closest slot whether or not it clears the resonance threshold."""
        best = self.retrieve(raw_query_vec, query_text)
        return self.slots[best["slot_id"]]["fact"] if best["slot_id"] else None, best["score"], best["slot_id"]

    def resonate(self, raw_query_vec: torch.Tensor, query_text: str = ""):
        best = self.retrieve(raw_query_vec, query_text)
        if best["slot_id"] is None:
            return None, 0.0, None
        if best["injected"]:
            return best["fact"], best["score"], best["slot_id"]
        return None, best["score"], None


# ==============================================================================
# LAYER 1 + LAYER 2: KURAMOTO SEMANTIC-EQUILIBRIUM ENGINE ON OPEN LLM
# ==============================================================================
class HRCLLMEngine:
    def __init__(self, model_id: str = MODEL_ID, force_offline: bool = False, dtype=None, device=None):
        requested_device = ACTIVE_DEVICE if device is None else device
        requested_dtype = (ACTIVE_DTYPE if dtype is None else dtype)
        self.device, dtype = _resolve_device_dtype(requested_device, requested_dtype)
        local_files_only = _has_complete_local_hf_checkpoint(model_id)
        cache_note = " (complete local cache; skipping Hub checks)" if local_files_only else ""
        print(f"[1/2] Loading {model_id} on {self.device}{cache_note}...", flush=True)
        self.tokenizer, loaded_model = load_hf_local_first(
            model_id,
            force_offline=force_offline,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = loaded_model.to(self.device)
        self.model.eval()
        print("[2/2] HRC Layer 2 (V_m) + Layer 3 (Resonant Memory) initialized.", flush=True)

    @torch.no_grad()
    def embed_texts(self, texts: list) -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.model.device)
        # Use the decoder body directly: the LM head logits and all intermediate
        # hidden-state tensors are unnecessary for pooled semantic embeddings.
        out = self.model.model(**enc, use_cache=False)
        last_h = out.last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1).cpu()

    @torch.no_grad()
    def embed_texts_batched(self, texts: list, batch_size: int = 64, progress: bool = False) -> torch.Tensor:
        chunks = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            chunks.append(self.embed_texts(texts[start:start + batch_size]))
            if progress and (batch_index % 4 == 0 or batch_index == total_batches):
                print(f"  Embedding batch {batch_index}/{total_batches}", flush=True)
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))

    def _build_prompt(self, question: str, resonant_fact: str = None) -> str:
        sys_msg = (
            "Answer the question directly in 1 to 5 words only. "
            "Do not explain. Always give your single best concrete factual answer."
        )
        user_msg = f"Verified Context: {resonant_fact}\nQuestion: {question}" if resonant_fact else question
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def infer_with_vm(
        self,
        question: str,
        memory: Layer3ResonantMemory = None,
        v_crit: float = -66.0,
        query_vec: torch.Tensor = None,
        compute_p_true: bool = False,
    ):
        # Step A: Query Layer 3 Resonant Episodic Memory using LLM's own geometry
        q_vec = query_vec if query_vec is not None else self.embed_texts([question])[0]
        res_fact, res_sim, res_id = (None, 0.0, None)
        res_candidate_id, res_dense_sim, res_sparse_sim, res_injected, res_entity_match = (None, 0.0, 0.0, False, True)
        if memory is not None:
            retrieved = memory.retrieve(q_vec, question)
            res_fact = retrieved["fact"]
            res_sim = retrieved["score"]
            res_candidate_id = retrieved["slot_id"]
            res_dense_sim = retrieved["dense"]
            res_sparse_sim = retrieved["sparse"]
            res_injected = retrieved["injected"]
            res_entity_match = retrieved["entity_match"]
            res_id = res_candidate_id if res_injected else None

        prompt_text = self._build_prompt(question, resonant_fact=res_fact)
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        in_len = enc["input_ids"].shape[1]

        # Step B: Canonical Greedy Trajectory (extracts answer + token entropy + logit margin)
        greedy_out = self.model.generate(
            **enc,
            max_new_tokens=14,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = greedy_out.sequences[0, in_len:]
        greedy_ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split("\n")[0]

        hrc_entropies, margins = [], []
        token_entropies, token_log_probs = [], []
        for step_idx, logits in enumerate(greedy_out.scores):
            tok_id = int(gen_ids[step_idx].item())
            if tok_id == self.tokenizer.eos_token_id:
                break
            probs = F.softmax(logits[0].float(), dim=-1)
            ent = -(probs * torch.log(probs + 1e-10)).sum().item()
            token_entropies.append(ent)
            token_log_probs.append(float(torch.log(probs[tok_id] + 1e-10).item()))
            if len(hrc_entropies) < 5:
                top2 = torch.topk(probs, k=2).values
                hrc_entropies.append(ent)
                margins.append(float((top2[0] - top2[1]).item()))

        mean_ent_norm = min(1.0, (sum(hrc_entropies) / max(1, len(hrc_entropies))) / 2.2)
        mean_margin = sum(margins) / max(1, len(margins)) if margins else 0.0
        margin_stress = 1.0 - mean_margin

        # Step C: Batched K=3 Phase-Perturbed Trajectories -> Kuramoto Order Parameter Phi_sem
        sampling_seed = _stable_query_seed(question)
        python_rng_state = random.getstate()
        cuda_devices = [torch.cuda.current_device()] if (
            torch.cuda.is_available() and self.model.device.type == "cuda"
        ) else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(sampling_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(sampling_seed)
            random.seed(sampling_seed)
            try:
                samp_out = self.model.generate(
                    **enc,
                    max_new_tokens=14,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    num_return_sequences=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            finally:
                random.setstate(python_rng_state)
        trajectories = [greedy_ans]
        for k in range(3):
            s_txt = self.tokenizer.decode(samp_out[k, in_len:], skip_special_tokens=True).strip().split("\n")[0]
            trajectories.append(s_txt)
        sem_entropy = discrete_semantic_entropy(trajectories)

        # 1. Hyperspherical Kuramoto Order Parameter on answer embeddings
        ans_vecs = self.embed_texts([t if t else "empty" for t in trajectories])
        phi_emb_raw = float(ans_vecs.mean(dim=0).norm(p=2).item())
        phi_emb = max(0.0, min(1.0, (phi_emb_raw - 0.55) / 0.45))

        # 2. Pairwise lexical phase-lock across the 4 trajectories
        tok_sets = [set(normalize_text(t).split()) for t in trajectories]
        pair_agrees = []
        for i in range(len(tok_sets)):
            for j in range(i + 1, len(tok_sets)):
                u = tok_sets[i] | tok_sets[j]
                pair_agrees.append(len(tok_sets[i] & tok_sets[j]) / len(u) if u else 0.0)
        phi_lex = sum(pair_agrees) / len(pair_agrees)

        phi_sem = 0.60 * phi_lex + 0.40 * phi_emb

        # Layer 2 Biophysical Equilibrium Potential V_m in [-85 mV, -20 mV]
        stress = min(1.0, max(0.0, 0.50 * (1.0 - phi_sem) + 0.30 * mean_ent_norm + 0.20 * margin_stress))
        v_m = -85.0 + 65.0 * stress
        halt = v_m > v_crit

        p_true = None
        p_true_seconds = 0.0
        if compute_p_true:
            p_true_prompt = (
                f"Question: {question}\nProposed Answer: {greedy_ans}\n"
                "Is the proposed answer factually correct? Answer Yes or No:"
            )
            p_true_enc = self.tokenizer(p_true_prompt, return_tensors="pt").to(self.model.device)
            yes_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
            no_ids = self.tokenizer.encode(" No", add_special_tokens=False)
            if len(yes_ids) != 1 or len(no_ids) != 1:
                raise ValueError("P(True) baseline requires single-token ' Yes' and ' No' candidates.")
            p_true_started = time.perf_counter()
            p_true_logits = self.model(**p_true_enc, use_cache=False).logits[0, -1].float()
            p_true_probs = F.softmax(p_true_logits, dim=-1)
            yes_prob = p_true_probs[yes_ids[0]]
            no_prob = p_true_probs[no_ids[0]]
            p_true = float((yes_prob / (yes_prob + no_prob + 1e-12)).item())
            p_true_seconds = time.perf_counter() - p_true_started

        return {
            "answer": greedy_ans,
            "trajectory_answers": trajectories,
            "mean_seq_logprob": sum(token_log_probs) / max(1, len(token_log_probs)),
            "mean_token_entropy": sum(token_entropies) / max(1, len(token_entropies)),
            "semantic_entropy": sem_entropy,
            "sampling_seed": sampling_seed,
            "phi_lex": phi_lex,
            "phi_emb": phi_emb,
            "phi_emb_raw": phi_emb_raw,
            "phi_sem": phi_sem,
            "mean_margin": mean_margin,
            "margin_stress": margin_stress,
            "ent": mean_ent_norm,
            "stress": stress,
            "v_m": v_m,
            "halt": halt,
            "q_vec": q_vec,
            "res_sim": res_sim,
            "res_id": res_id,
            "res_candidate_id": res_candidate_id,
            "res_dense_sim": res_dense_sim,
            "res_sparse_sim": res_sparse_sim,
            "res_injected": res_injected,
            "res_entity_match": res_entity_match,
            "p_true": p_true,
            "p_true_seconds": p_true_seconds,
            "greedy_forward_steps": len(greedy_out.scores),
            "sample_forward_steps": int(samp_out.shape[1] - in_len),
        }


# ==============================================================================
# 3-PHASE TRACK C BENCHMARK RUNNER
# ==============================================================================
def _risk_metrics(results: list, score_name: str, halt_count: int):
    ordered = sorted(results, key=lambda row: (row["risk"][score_name], row["index"]))
    errors_seen = 0
    aurc_sum = 0.0
    for rank, row in enumerate(ordered, start=1):
        errors_seen += int(not row["correct"])
        aurc_sum += errors_seen / rank
    aurc = aurc_sum / max(1, len(ordered))

    caught_indices = {
        row["index"] for row in sorted(
            results,
            key=lambda row: (-row["risk"][score_name], row["index"]),
        )[:halt_count]
    }
    submitted = [row for row in results if row["index"] not in caught_indices]
    selective_accuracy = (
        sum(int(row["correct"]) for row in submitted) / len(submitted)
        if submitted else 0.0
    )
    false_premise_rows = [row for row in results if row["bucket"] == "FalsePremise"]
    high_phi_wrong_rows = [
        row for row in false_premise_rows
        if row["phi"] >= 0.70 and not row["correct"]
    ]
    return {
        "aurc": aurc,
        "caught_indices": caught_indices,
        "false_premise_caught": sum(row["index"] in caught_indices for row in false_premise_rows),
        "high_phi_wrong_caught": sum(row["index"] in caught_indices for row in high_phi_wrong_rows),
        "high_phi_wrong_total": len(high_phi_wrong_rows),
        "selective_accuracy": selective_accuracy,
    }


def run_baselines_part_4a(v_crit: float = -66.0):
    """Compare five uncertainty scores on the same 40 generated Phase 1 answers."""
    torch.manual_seed(42)
    engine = HRCLLMEngine(MODEL_ID)
    phase1_stream = (
        WORLD_KNOWN
        + OBSCURE_HARD
        + [(ep["q_orig"], ep["aliases"], "Private") for ep in PRIVATE_EPISODES]
        + [(probe["question"], probe["gold"], probe["bucket"]) for probe in PHASE1_ADVERSARIAL_PROBES]
    )
    results = []
    total_p_true_seconds = 0.0
    total_decoder_forward_steps = 0

    print("\n" + "=" * 148)
    print("PART 4A: LAYER 2 UNCERTAINTY BASELINES (40 Phase 1 queries; shared answer + K=3 rollout pass)")
    print("=" * 148)
    print(
        f"{'#':<2} | {'Bucket':<13} | {'Question':<43} | {'Answer':<23} | {'Corr':<5} | "
        f"{'Phi':>5} | {'V_m':>6} | {'P(True)':>7} | {'LogP':>8} | {'Entropy':>8} | {'SemEnt':>7}"
    )
    print("-" * 148)

    for index, (question, aliases, bucket) in enumerate(phase1_stream):
        query_started = time.perf_counter()
        res = engine.infer_with_vm(question, v_crit=v_crit, compute_p_true=True)
        elapsed = time.perf_counter() - query_started
        correct = grade_phase1_answer(res["answer"], aliases)
        results.append({
            "index": index,
            "bucket": bucket,
            "question": question,
            "answer": res["answer"],
            "correct": correct,
            "phi": res["phi_sem"],
            "v_m": res["v_m"],
            "elapsed": elapsed,
            "p_true_seconds": res["p_true_seconds"],
            "greedy_forward_steps": res["greedy_forward_steps"],
            "sample_forward_steps": res["sample_forward_steps"],
            "risk": {
                "Length-Norm Log-Prob": -res["mean_seq_logprob"],
                "Mean Token Entropy": res["mean_token_entropy"],
                "P(True) Self-Verification": 1.0 - res["p_true"],
                "Normalized-Cluster Semantic Entropy": res["semantic_entropy"],
                "HRC V_m": res["v_m"],
            },
        })
        total_p_true_seconds += res["p_true_seconds"]
        total_decoder_forward_steps += (
            res["greedy_forward_steps"] + res["sample_forward_steps"] + 1
        )
        print(
            f"{index:<2} | {bucket:<13} | {question[:43]:<43} | {res['answer'][:23]:<23} | "
            f"{str(correct):<5} | {res['phi_sem']:5.2f} | {res['v_m']:6.1f} | "
            f"{res['p_true']:7.3f} | {res['mean_seq_logprob']:8.3f} | "
            f"{res['mean_token_entropy']:8.3f} | {res['semantic_entropy']:7.3f}",
            flush=True,
        )

    halt_count = sum(row["v_m"] > v_crit for row in results)
    hrc_coverage = (len(results) - halt_count) / max(1, len(results))
    non_false_premise = [row for row in results if row["bucket"] != "FalsePremise"]
    methods = [
        "Length-Norm Log-Prob",
        "Mean Token Entropy",
        "P(True) Self-Verification",
        "Normalized-Cluster Semantic Entropy",
        "HRC V_m",
    ]

    print("\nPart 4A scorecard (higher risk score means more likely wrong):")
    print(
        f"{'Method':<38} | {'AUROC 40':>8} | {'AUROC 34':>8} | {'AURC':>6} | "
        f"{'SelAcc@HRC':>10} | {'FP caught':>9} | {'HiPhi caught':>12}"
    )
    print("-" * 124)
    for method in methods:
        all_metrics = _risk_metrics(results, method, halt_count)
        all_wrong_scores = [row["risk"][method] for row in results if not row["correct"]]
        all_correct_scores = [row["risk"][method] for row in results if row["correct"]]
        non_fp_wrong_scores = [row["risk"][method] for row in non_false_premise if not row["correct"]]
        non_fp_correct_scores = [row["risk"][method] for row in non_false_premise if row["correct"]]
        auroc_all = compute_auroc(all_wrong_scores, all_correct_scores)
        auroc_non_fp = compute_auroc(non_fp_wrong_scores, non_fp_correct_scores)
        print(
            f"{method:<38} | {auroc_all:8.3f} | {auroc_non_fp:8.3f} | "
            f"{all_metrics['aurc']:6.3f} | {all_metrics['selective_accuracy'] * 100:9.1f}% | "
            f"{all_metrics['false_premise_caught']:>4}/6 | "
            f"{all_metrics['high_phi_wrong_caught']:>4}/{all_metrics['high_phi_wrong_total']:<2}",
            flush=True,
        )

    print(
        f"\nHRC coverage used for matched selective accuracy: {hrc_coverage * 100:.1f}% "
        f"({len(results) - halt_count}/{len(results)} submitted; {halt_count} halted)."
    )
    elapsed_total = sum(row["elapsed"] for row in results)
    print(
        f"Combined shared-run latency: {elapsed_total:.1f}s total, "
        f"{elapsed_total / len(results):.3f}s/query."
    )
    print(
        f"Decoder forward calls/query: {total_decoder_forward_steps / len(results):.1f} mean "
        f"(greedy decode + batched K=3 rollouts + 1 P(True) pass; embedding-only forwards excluded)."
    )
    print(f"P(True) next-token forward latency: {total_p_true_seconds / len(results):.4f}s/query.")
    print("Semantic Entropy is normalized-answer cluster entropy (natural-log units; no NLI model).")


def build_dense_memory(n_total, entries, forget_ids, key_vecs, base_vecs):
    memory = Layer3ResonantMemory(tau_res=0.45)
    memory.fit_isotropy_center(base_vecs)
    for entry in entries[:n_total]:
        memory.consolidate(entry["id"], key_vecs[entry["id"]], entry["answer"])

    purge_started = time.perf_counter()
    for slot_id in sorted(forget_ids):
        memory.exact_unlearn(slot_id)
    purge_ms = (time.perf_counter() - purge_started) * 1000.0
    return memory, purge_ms


def benchmark_purge_latency(repeats: int = 1000):
    """Measure dict-slot deletion only; clone each synthetic store outside the timer."""
    print("\nPURGE LATENCY: 1,000 cloned-store repetitions per scale (clone excluded from timing)")
    print(f"{'N':>5} | {'Purged':>8} | {'Mean us/item':>13} | {'Total timed ms':>15}")
    print("-" * 52)
    for n_total in (6, 25, 100, 500):
        entries, forget_ids = generate_dense_cluster_memories(n_total)
        base_slots = {entry["id"]: None for entry in entries}
        purge_ids = sorted(forget_ids)
        elapsed_ns = 0
        for _ in range(repeats):
            cloned_slots = dict(base_slots)
            started_ns = time.perf_counter_ns()
            for slot_id in purge_ids:
                cloned_slots.pop(slot_id, None)
            elapsed_ns += time.perf_counter_ns() - started_ns
        deleted_slots = repeats * len(purge_ids)
        mean_us = elapsed_ns / max(1, deleted_slots) / 1000.0
        print(
            f"{n_total:>5} | {len(purge_ids):>4}/{n_total:<3} | {mean_us:>13.3f} | "
            f"{elapsed_ns / 1_000_000:>15.3f}"
        )


def select_dense_probe_slice(entries, forget_ids):
    """Select the same six forgotten targets and six retained siblings as Tier 2B."""
    forget_by_cluster = {}
    retain_by_cluster = {}
    for entry in entries:
        target = forget_by_cluster if entry["id"] in forget_ids else retain_by_cluster
        target.setdefault(entry["cluster"], []).append(entry)
    forgotten_targets = [forget_by_cluster[name][0] for name in sorted(forget_by_cluster)]
    while len(forgotten_targets) < 6:
        for cluster_name in sorted(forget_by_cluster):
            candidate = next(
                (entry for entry in forget_by_cluster[cluster_name] if entry not in forgotten_targets),
                None,
            )
            if candidate is not None:
                forgotten_targets.append(candidate)
                break

    retained_targets = []
    for forgotten_entry in forgotten_targets:
        target_index = int(forgotten_entry["id"][1:])
        sibling = next(
            entry for entry in retain_by_cluster[forgotten_entry["cluster"]]
            if int(entry["id"][1:]) > target_index
            and entry not in retained_targets
        )
        retained_targets.append(sibling)
    return forgotten_targets, retained_targets


def run_unlearn_audit(v_crit: float = -66.0):
    """Audit unseen paraphrase, completion-pressure, and indirect unlearning probes."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    world_questions = [question for question, _, _ in WORLD_KNOWN]
    original_questions = [ep["q_orig"] for ep in PRIVATE_EPISODES]
    audit_questions = [probe["question"] for probe in UNLEARN_AUDIT_PROBES]
    all_vectors = engine.embed_texts_batched(
        world_questions + original_questions + audit_questions,
        batch_size=32,
        progress=True,
    )
    world_count = len(world_questions)
    original_start = world_count
    audit_start = original_start + len(original_questions)
    base_vecs = all_vectors[:world_count]
    original_vecs = {
        ep["id"]: all_vectors[original_start + i]
        for i, ep in enumerate(PRIVATE_EPISODES)
    }
    audit_vecs = {
        i: all_vectors[audit_start + i]
        for i in range(len(UNLEARN_AUDIT_PROBES))
    }

    memory = Layer3ResonantMemory(tau_res=0.45)
    memory.fit_isotropy_center(base_vecs)
    for ep in PRIVATE_EPISODES:
        memory.consolidate(ep["id"], original_vecs[ep["id"]], ep["fact"])

    purge_started = time.perf_counter()
    purged_ids = [ep["id"] for ep in PRIVATE_EPISODES if ep["forget"]]
    for slot_id in purged_ids:
        memory.exact_unlearn(slot_id)
    purge_ms = (time.perf_counter() - purge_started) * 1000.0

    episode_by_id = {ep["id"]: ep for ep in PRIVATE_EPISODES}
    forgotten_rows = []
    retained_resims = []
    retained_hits = 0
    print("\n" + "=" * 136)
    print("STEP 3: UNSEEN-PARAPHRASE & INDIRECT-QUERY UNLEARNING AUDIT")
    print(f"Purged {purged_ids} in {purge_ms:.4f} ms; retained slots: {sorted(memory.slots)}")
    print("=" * 136)
    print(
        f"{'Kind':<18} | {'ID':<3} | {'Prediction':<22} | {'ResID':<5} | "
        f"{'ResSim':>6} | {'V_m':>6} | {'Action':<6} | Outcome"
    )
    print("-" * 136)

    for i, probe in enumerate(UNLEARN_AUDIT_PROBES):
        ep = episode_by_id[probe["id"]]
        res = engine.infer_with_vm(
            probe["question"],
            memory=memory,
            v_crit=v_crit,
            query_vec=audit_vecs[i],
        )
        action = "HALT" if res["halt"] else "SUBMIT"
        matched = is_correct_answer(res["answer"], ep["aliases"])
        if ep["forget"]:
            row = {"action": action, "matched": matched, "probe": probe, "res": res}
            forgotten_rows.append(row)
            if action == "HALT":
                outcome = "CLEAN HALT" if not matched else "HALT / purged answer"
            elif matched:
                outcome = "PURGED ANSWER SUBMITTED"
            else:
                outcome = "WRONG ANSWER SUBMITTED"
        else:
            retained_resims.append(res["res_sim"])
            correct_submit = matched and action == "SUBMIT"
            retained_hits += int(correct_submit)
            outcome = "RETAINED RECALL" if correct_submit else ("HALTED" if action == "HALT" else "WRONG SUBMIT")

        print(
            f"{probe['kind']:<18} | {probe['id']:<3} | {res['answer'][:22]:<22} | "
            f"{str(res['res_id'] or '-'):>5} | {res['res_sim']:6.3f} | {res['v_m']:6.1f} | "
            f"{action:<6} | {outcome}",
            flush=True,
        )

    direct_submits = sum(row["action"] == "SUBMIT" and row["matched"] for row in forgotten_rows)
    wrong_submits = sum(row["action"] == "SUBMIT" and not row["matched"] for row in forgotten_rows)
    clean_halts = sum(row["action"] == "HALT" for row in forgotten_rows)
    halt_purged_answer = sum(row["action"] == "HALT" and row["matched"] for row in forgotten_rows)
    forgotten_resims = [row["res"]["res_sim"] for row in forgotten_rows]

    print("-" * 136)
    print(
        f"Forgotten submissions: purged answer {direct_submits}/9 | wrong answer {wrong_submits}/9 | "
        f"clean HALT {clean_halts}/9 (HALT with purged answer: {halt_purged_answer})"
    )
    if forgotten_resims and retained_resims:
        max_forget = max(forgotten_resims)
        min_retain = min(retained_resims)
        print(
            f"Unseen ResSim separation: max forgotten {max_forget:.3f} | "
            f"min retained {min_retain:.3f} | margin {min_retain - max_forget:+.3f}"
        )
    else:
        print("Unseen ResSim separation: unavailable")
    print(f"Retained unseen-paraphrase recall: {retained_hits}/3")

    benchmark_purge_latency(repeats=1000)
    print(f"\nUnlearn-audit runtime: {time.perf_counter() - started:.1f}s on {engine.device}.")


def run_dense_scale(v_crit: float = -66.0):
    """Run memory-only density sweeps, then 16 N=100 end-to-end LLM probes."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    scale_sizes = (6, 25, 100, 500)
    entries_500, _ = generate_dense_cluster_memories(500)
    world_questions = [question for question, _, _ in WORLD_KNOWN]
    all_texts = (
        world_questions
        + [entry["store_q"] for entry in entries_500]
        + [entry["para_q"] for entry in entries_500]
    )
    print("\nEmbedding Phase 2A memory/query text in batches (no answer generation)...", flush=True)
    all_vectors = engine.embed_texts_batched(all_texts, batch_size=64, progress=True)
    base_count = len(world_questions)
    key_start = base_count
    para_start = key_start + len(entries_500)
    base_vecs = all_vectors[:base_count]
    key_vecs = {entry["id"]: all_vectors[key_start + i] for i, entry in enumerate(entries_500)}
    para_vecs = {entry["id"]: all_vectors[para_start + i] for i, entry in enumerate(entries_500)}

    print("\n" + "=" * 132)
    print("TIER 2A: MEMORY-ONLY DENSE-CLUSTER SCALING (Qwen embeddings; no LLM generation)")
    print("=" * 132)
    print(
        f"{'N':>4} | {'Purged':>8} | {'Retain Top1':>12} | {'NN Collision':>13} | "
        f"{'Min Retain':>11} | {'Max Forget':>11} | {'Margin':>9} | {'Purge ms':>10} | {'ms/item':>9}"
    )
    print("-" * 132)

    memories = {}
    for n_total in scale_sizes:
        entries, forget_ids = generate_dense_cluster_memories(n_total)
        memory, purge_ms = build_dense_memory(n_total, entries, forget_ids, key_vecs, base_vecs)
        memories[n_total] = memory

        retained = [entry for entry in entries if entry["id"] not in forget_ids]
        forgotten = [entry for entry in entries if entry["id"] in forget_ids]
        retain_scores = []
        retain_hits = 0
        nearest_collisions = 0
        for entry in retained:
            _, similarity, nearest_id = memory.nearest(para_vecs[entry["id"]])
            retain_scores.append(similarity)
            if similarity >= memory.tau_res and nearest_id == entry["id"]:
                retain_hits += 1
            if nearest_id != entry["id"]:
                nearest_collisions += 1

        forget_scores = [memory.nearest(para_vecs[entry["id"]])[1] for entry in forgotten]
        min_retain = min(retain_scores) if retain_scores else float("nan")
        max_forget = max(forget_scores) if forget_scores else float("nan")
        margin = min_retain - max_forget
        purge_count = len(forget_ids)
        print(
            f"{n_total:>4} | {purge_count:>4}/{n_total:<3} | {retain_hits:>4}/{len(retained):<7} | "
            f"{nearest_collisions:>4}/{len(retained):<8} | {min_retain:>11.3f} | {max_forget:>11.3f} | "
            f"{margin:>9.3f} | {purge_ms:>10.4f} | {purge_ms / max(1, purge_count):>9.4f}"
        )

    # Six forgotten targets and six surviving siblings, spanning all five templates.
    entries_100, forget_ids_100 = generate_dense_cluster_memories(100)
    memory_100 = memories[100]
    forgotten_targets, retained_targets = select_dense_probe_slice(entries_100, forget_ids_100)

    print("\n" + "=" * 132)
    print("TIER 2B: N=100 END-TO-END SLICE (25 purged; 16 LLM queries)")
    print("=" * 132)
    print(
        f"{'Type':<8} | {'ID':<5} | {'Prediction':<20} | {'ResID':<5} | {'ResSim':>6} | "
        f"{'V_m':>6} | {'Phi':>5} | {'Action':<6} | {'Outcome'}"
    )
    print("-" * 132)

    # Keep these first so the fixed seed reproduces the four Phase 1 baseline trajectories.
    baseline_vms = (-84.6, -84.6, -82.5, -70.3)
    print("Base retention probes (Phase 1 baseline V_m, no memory):")
    for i, (question, aliases, _) in enumerate(WORLD_KNOWN[:4]):
        res = engine.infer_with_vm(
            question,
            memory=memory_100,
            v_crit=v_crit,
            query_vec=base_vecs[i],
        )
        action = "HALT" if res["halt"] else "SUBMIT"
        delta = res["v_m"] - baseline_vms[i]
        outcome = f"baseline {baseline_vms[i]:.1f}; delta {delta:+.1f} mV"
        print(
            f"{'Base':<8} | B{i+1:<4} | {res['answer'][:20]:<20} | {str(res['res_id'] or '-'):>5} | "
            f"{res['res_sim']:6.3f} | {res['v_m']:6.1f} | {res['phi_sem']:5.2f} | {action:<6} | {outcome}",
            flush=True,
        )

    direct_leaks = 0
    sibling_retrievals = 0
    sibling_bleeds = 0
    safe_forget_halts = 0
    answers_by_id = {entry["id"]: entry["answer"] for entry in entries_100}

    for entry in forgotten_targets:
        res = engine.infer_with_vm(
            entry["para_q"], memory=memory_100, v_crit=v_crit, query_vec=para_vecs[entry["id"]]
        )
        action = "HALT" if res["halt"] else "SUBMIT"
        direct = is_correct_answer(res["answer"], [entry["answer"]])
        neighbor_retrieved = res["res_id"] is not None and res["res_id"] != entry["id"]
        sibling_answered = neighbor_retrieved and is_correct_answer(res["answer"], [answers_by_id[res["res_id"]]])
        direct_leaks += int(direct)
        sibling_retrievals += int(neighbor_retrieved)
        sibling_bleeds += int(sibling_answered and action == "SUBMIT")
        safe_forget_halts += int(action == "HALT" and not direct)
        if direct:
            outcome = "DIRECT LEAK"
        elif sibling_answered and action == "SUBMIT":
            outcome = "SIBLING BLEED"
        elif action == "HALT":
            outcome = "HALTED"
        else:
            outcome = "OTHER SUBMIT"
        print(
            f"{'Forgotten':<8} | {entry['id']:<5} | {res['answer'][:20]:<20} | "
            f"{str(res['res_id'] or '-'):>5} | {res['res_sim']:6.3f} | {res['v_m']:6.1f} | "
            f"{res['phi_sem']:5.2f} | {action:<6} | {outcome}",
            flush=True,
        )

    retained_recall = 0
    for entry in retained_targets:
        res = engine.infer_with_vm(
            entry["para_q"], memory=memory_100, v_crit=v_crit, query_vec=para_vecs[entry["id"]]
        )
        action = "HALT" if res["halt"] else "SUBMIT"
        correct_submit = is_correct_answer(res["answer"], [entry["answer"]]) and action == "SUBMIT"
        retained_recall += int(correct_submit)
        outcome = "RECALLED" if correct_submit else ("HALTED" if action == "HALT" else "WRONG")
        print(
            f"{'Retained':<8} | {entry['id']:<5} | {res['answer'][:20]:<20} | "
            f"{str(res['res_id'] or '-'):>5} | {res['res_sim']:6.3f} | {res['v_m']:6.1f} | "
            f"{res['phi_sem']:5.2f} | {action:<6} | {outcome}",
            flush=True,
        )

    print("-" * 132)
    print(f"Tier 2B Summary | Retained sibling recall: {retained_recall}/6")
    print(
        f"  Forgotten targets: direct leakage {direct_leaks}/6 | sibling neighbor retrieved "
        f"{sibling_retrievals}/6 | sibling bleed submitted {sibling_bleeds}/6 | "
        f"safe HALT without direct leakage {safe_forget_halts}/6"
    )
    print("  Base V_m deltas are measured against the Phase 1 no-memory baselines shown above.")
    print(f"\nDense-scale runtime: {time.perf_counter() - started:.1f}s on {engine.device}.")


def _build_m2_l3_memory(entries, key_vecs, base_vecs, forget_ids):
    memory = Layer3ResonantMemory(tau_res=0.45, hybrid_alpha=0.25, tau_inject=0.78)
    memory.fit_isotropy_center(base_vecs)
    for entry in entries:
        fact = entry.get("fact")
        if fact is None:
            fact = f"The record for '{entry['store_q']}' has the answer: {entry['answer']}."
        memory.consolidate(entry["id"], key_vecs[entry["id"]], fact, entry["store_q"])
    purge_started = time.perf_counter()
    for slot_id in sorted(forget_ids):
        memory.exact_unlearn(slot_id)
    purge_ms = (time.perf_counter() - purge_started) * 1000.0
    return memory, purge_ms


def run_m2_l3(v_crit: float = -66.0, purge_repeats: int = 1000):
    """Milestone 2 Fix 1: injection floor plus sparse/dense hybrid Layer 3 retrieval."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    dense_entries_500, _ = generate_dense_cluster_memories(500)
    entries_100, forget_ids_100 = generate_dense_cluster_memories(100)
    world_questions = [question for question, _, _ in WORLD_KNOWN]
    audit_questions = [probe["question"] for probe in UNLEARN_AUDIT_PROBES]
    texts = (
        world_questions
        + [ep["q_orig"] for ep in PRIVATE_EPISODES]
        + audit_questions
        + [entry["store_q"] for entry in dense_entries_500]
        + [entry["para_q"] for entry in dense_entries_500]
    )
    print("\nEmbedding benchmark corpus for --m2-l3...", flush=True)
    vectors = engine.embed_texts_batched(texts, batch_size=64, progress=True)
    world_count = len(world_questions)
    episode_start = world_count
    audit_start = episode_start + len(PRIVATE_EPISODES)
    dense_start = audit_start + len(audit_questions)
    para_start = dense_start + len(dense_entries_500)
    base_vecs = vectors[:world_count]
    episode_vecs = {ep["id"]: vectors[episode_start + i] for i, ep in enumerate(PRIVATE_EPISODES)}
    audit_vecs = {i: vectors[audit_start + i] for i in range(len(UNLEARN_AUDIT_PROBES))}
    dense_key_vecs = {entry["id"]: vectors[dense_start + i] for i, entry in enumerate(dense_entries_500)}
    para_vecs = {entry["id"]: vectors[para_start + i] for i, entry in enumerate(dense_entries_500)}

    # Phase 3: original E1-E6 store, purge E1-E3, and replay the same 12 unseen probes.
    phase3_entries = [
        {"id": ep["id"], "fact": ep["fact"], "store_q": ep["q_orig"]}
        for ep in PRIVATE_EPISODES
    ]
    forgotten_ids = {ep["id"] for ep in PRIVATE_EPISODES if ep["forget"]}
    memory, phase3_purge_ms = _build_m2_l3_memory(phase3_entries, episode_vecs, base_vecs, forgotten_ids)
    print("\n" + "=" * 148)
    print("MILESTONE 2 FIX 1: PHASE 3 AUDIT (alpha=0.25, tau_inject=0.78; 12 probes)")
    print("=" * 148)
    print(f"{'Kind':<18} | {'ID':<3} | {'Prediction':<20} | {'Candidate':<9} | {'Hybrid':>6} | {'Dense':>6} | {'Sparse':>6} | {'Injected':>8} | {'V_m':>6} | {'Action':<6} | Outcome")
    print("-" * 148)
    phase3_forgotten = []
    phase3_retained_recall = 0
    phase3_retained_scores = []
    for i, probe in enumerate(UNLEARN_AUDIT_PROBES):
        ep = next(row for row in PRIVATE_EPISODES if row["id"] == probe["id"])
        result = engine.infer_with_vm(probe["question"], memory=memory, v_crit=v_crit, query_vec=audit_vecs[i])
        action = "HALT" if result["halt"] else "SUBMIT"
        matched = is_correct_answer(result["answer"], ep["aliases"])
        if ep["forget"]:
            phase3_forgotten.append({"action": action, "matched": matched, "result": result, "probe": probe})
            if matched:
                outcome = "PURGED ANSWER" if action == "SUBMIT" else "HALT / purged answer"
            elif action == "HALT":
                outcome = "CLEAN HALT"
            else:
                outcome = "WRONG SUBMIT"
        else:
            phase3_retained_scores.append(result["res_sim"])
            correct_submit = matched and action == "SUBMIT"
            phase3_retained_recall += int(correct_submit)
            outcome = "RETAINED RECALL" if correct_submit else ("HALTED" if action == "HALT" else "WRONG SUBMIT")
        print(
            f"{probe['kind']:<18} | {probe['id']:<3} | {result['answer'][:20]:<20} | "
            f"{str(result['res_candidate_id'] or '-'):>9} | {result['res_sim']:6.3f} | {result['res_dense_sim']:6.3f} | "
            f"{result['res_sparse_sim']:6.3f} | {str(result['res_injected']):>8} | {result['v_m']:6.1f} | {action:<6} | {outcome}",
            flush=True,
        )
    direct_submits = sum(row["action"] == "SUBMIT" and row["matched"] for row in phase3_forgotten)
    wrong_submits = sum(row["action"] == "SUBMIT" and not row["matched"] for row in phase3_forgotten)
    clean_halts = sum(row["action"] == "HALT" for row in phase3_forgotten)
    print("-" * 148)
    print(
        f"Phase 3 summary | purged-answer submissions {direct_submits}/9 | wrong submissions {wrong_submits}/9 | "
        f"HALT {clean_halts}/9 | retained unseen recall {phase3_retained_recall}/3 | purge {phase3_purge_ms:.4f} ms"
    )
    if phase3_retained_scores and phase3_forgotten:
        max_forget = max(row["result"]["res_sim"] for row in phase3_forgotten)
        min_retain = min(phase3_retained_scores)
        print(f"Hybrid similarity margin | max forgotten {max_forget:.3f} | min retained {min_retain:.3f} | margin {min_retain - max_forget:+.3f}")

    # Tier 2A: evaluate all four corpus sizes with the identical hybrid/index logic.
    print("\n" + "=" * 126)
    print("MILESTONE 2 FIX 1: TIER 2A HYBRID RETRIEVAL + INDEXED PURGE SWEEP")
    print("=" * 126)
    print(f"{'N':>4} | {'Retain Top1':>12} | {'NN collision':>12} | {'Min retain':>10} | {'Max forget':>10} | {'Margin':>8} | {'Purge ms':>10} | {'Index OK':>8}")
    print("-" * 126)
    for n_total in (6, 25, 100, 500):
        entries, forget_ids = generate_dense_cluster_memories(n_total)
        scale_memory, purge_ms = _build_m2_l3_memory(entries, dense_key_vecs, base_vecs, forget_ids)
        retained = [row for row in entries if row["id"] not in forget_ids]
        forgotten = [row for row in entries if row["id"] in forget_ids]
        hits = 0
        collisions = 0
        retain_scores = []
        for row in retained:
            got = scale_memory.retrieve(para_vecs[row["id"]], row["para_q"])
            retain_scores.append(got["score"])
            hits += int(got["slot_id"] == row["id"])
            collisions += int(got["slot_id"] != row["id"])
        forget_scores = [scale_memory.retrieve(para_vecs[row["id"]], row["para_q"])["score"] for row in forgotten]
        margin = min(retain_scores) - max(forget_scores) if retain_scores and forget_scores else float("nan")
        print(
            f"{n_total:>4} | {hits:>4}/{len(retained):<7} | {collisions:>5}/{len(retained):<6} | "
            f"{min(retain_scores):10.3f} | {max(forget_scores):10.3f} | {margin:8.3f} | {purge_ms:10.4f} | {str(scale_memory.index_is_consistent()):>8}"
        )

    # Repeated purge timing on cloned, fully indexed stores. Cloning is outside each timed interval.
    print(f"\nIndexed purge timing: {purge_repeats:,} cloned-store repetitions (clone excluded from timed interval)")
    print(f"{'N':>5} | {'Purged':>8} | {'Mean us/item':>13} | {'Timed total ms':>14} | {'No stale index':>14}")
    print("-" * 70)
    for n_total in (6, 25, 100, 500):
        entries, forget_ids = generate_dense_cluster_memories(n_total)
        full_memory = Layer3ResonantMemory(tau_res=0.45, hybrid_alpha=0.25, tau_inject=0.78)
        full_memory.fit_isotropy_center(base_vecs)
        for row in entries:
            full_memory.consolidate(row["id"], dense_key_vecs[row["id"]], row["answer"], row["store_q"])
        purge_ids = sorted(forget_ids)
        elapsed_ns = 0
        last_clone = None
        for _ in range(purge_repeats):
            last_clone = full_memory.clone()
            t0 = time.perf_counter_ns()
            for slot_id in purge_ids:
                last_clone.exact_unlearn(slot_id)
            elapsed_ns += time.perf_counter_ns() - t0
        deleted = purge_repeats * len(purge_ids)
        print(
            f"{n_total:>5} | {len(purge_ids):>4}/{n_total:<3} | {elapsed_ns / max(1, deleted) / 1000.0:13.3f} | "
            f"{elapsed_ns / 1_000_000:14.3f} | {str(last_clone.index_is_consistent() and not (set(purge_ids) & set(last_clone.slots))):>14}"
        )

    # Tier 2B end-to-end N=100 slice with 25 purged memories.
    memory_100, _ = _build_m2_l3_memory(entries_100, dense_key_vecs, base_vecs, forget_ids_100)
    forgotten_targets, retained_targets = select_dense_probe_slice(entries_100, forget_ids_100)
    if not print_entity_guard_validation():
        raise AssertionError("Tier 2C entity-anchor guard validation failed")
    print("\n" + "=" * 140)
    print("MILESTONE 2 FIX 1: TIER 2B N=100 END-TO-END (16 queries; alpha=0.25, tau_inject=0.78)")
    print("=" * 140)
    print(f"{'Type':<9} | {'ID':<5} | {'Prediction':<19} | {'Candidate':<9} | {'Hybrid':>6} | {'Dense':>6} | {'EntityOK':>8} | {'Injected':>8} | {'V_m':>6} | {'Action':<6} | Outcome")
    print("-" * 140)
    baseline_vms = (-84.6, -84.6, -82.5, -70.3)
    base_recall = 0
    torch.manual_seed(42)
    for i, (question, aliases, _) in enumerate(WORLD_KNOWN[:4]):
        result = engine.infer_with_vm(question, memory=memory_100, v_crit=v_crit, query_vec=base_vecs[i])
        action = "HALT" if result["halt"] else "SUBMIT"
        correct_submit = is_correct_answer(result["answer"], aliases) and action == "SUBMIT"
        base_recall += int(correct_submit)
        print(
            f"{'Base':<9} | B{i+1:<4} | {result['answer'][:19]:<19} | {str(result['res_candidate_id'] or '-'):>9} | "
            f"{result['res_sim']:6.3f} | {result['res_dense_sim']:6.3f} | {str(result['res_entity_match']):>8} | {str(result['res_injected']):>8} | "
            f"{result['v_m']:6.1f} | {action:<6} | baseline {baseline_vms[i]:.1f}; delta {result['v_m'] - baseline_vms[i]:+.1f} mV",
            flush=True,
        )
    direct_leaks = sibling_retrievals = sibling_bleeds = safe_halts = entity_mismatch_blocks = 0
    retained_recall = 0
    retained_exact_retrieval = 0
    by_id = {entry["id"]: entry for entry in entries_100}
    for row_type, targets in (("Forgotten", forgotten_targets), ("Retained", retained_targets)):
        for entry in targets:
            result = engine.infer_with_vm(entry["para_q"], memory=memory_100, v_crit=v_crit, query_vec=para_vecs[entry["id"]])
            action = "HALT" if result["halt"] else "SUBMIT"
            direct = is_correct_answer(result["answer"], [entry["answer"]])
            candidate = result["res_candidate_id"]
            neighbor_retrieved = candidate is not None and candidate != entry["id"]
            sibling_answered = neighbor_retrieved and is_correct_answer(result["answer"], [by_id[candidate]["answer"]])
            if row_type == "Forgotten":
                direct_leaks += int(direct and action == "SUBMIT")
                sibling_retrievals += int(neighbor_retrieved)
                sibling_bleeds += int(sibling_answered and action == "SUBMIT")
                safe_halts += int(action == "HALT" and not direct)
                entity_mismatch_blocks += int(not result["res_entity_match"] and not result["res_injected"])
                outcome = "DIRECT LEAK" if direct and action == "SUBMIT" else ("SIBLING BLEED" if sibling_answered and action == "SUBMIT" else ("HALTED" if action == "HALT" else "OTHER SUBMIT"))
            else:
                retained_exact_retrieval += int(candidate == entry["id"])
                correct_submit = direct and action == "SUBMIT"
                retained_recall += int(correct_submit)
                outcome = "RECALLED" if correct_submit else ("HALTED" if action == "HALT" else "WRONG")
            print(
                f"{row_type:<9} | {entry['id']:<5} | {result['answer'][:19]:<19} | {str(candidate or '-'):>9} | "
                f"{result['res_sim']:6.3f} | {result['res_dense_sim']:6.3f} | {str(result['res_entity_match']):>8} | {str(result['res_injected']):>8} | "
                f"{result['v_m']:6.1f} | {action:<6} | {outcome}",
                flush=True,
            )
    print("-" * 140)
    print(f"Tier 2B summary | retained exact retrieval {retained_exact_retrieval}/6 | retained correct SUBMIT {retained_recall}/6 | forgotten direct submission {direct_leaks}/6 | sibling candidates {sibling_retrievals}/6 | entity mismatch blocks {entity_mismatch_blocks}/6 | sibling bleed submitted {sibling_bleeds}/6 | safe HALT {safe_halts}/6 | base recall {base_recall}/4")
    print(f"\n--m2-l3 runtime: {time.perf_counter() - started:.1f}s on {engine.device}; sparse-index integrity: {memory.index_is_consistent()} / {memory_100.index_is_consistent()}.")


def run_m2_l3_tier2b(v_crit: float = -66.0):
    """Re-run just the 16-query N=100 slice after a retrieval-policy adjustment."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    entries, forget_ids = generate_dense_cluster_memories(100)
    world_questions = [question for question, _, _ in WORLD_KNOWN]
    texts = world_questions + [entry["store_q"] for entry in entries] + [entry["para_q"] for entry in entries]
    vectors = engine.embed_texts_batched(texts, batch_size=64, progress=True)
    world_count = len(world_questions)
    key_start = world_count
    para_start = key_start + len(entries)
    base_vecs = vectors[:world_count]
    key_vecs = {entry["id"]: vectors[key_start + i] for i, entry in enumerate(entries)}
    para_vecs = {entry["id"]: vectors[para_start + i] for i, entry in enumerate(entries)}
    memory, _ = _build_m2_l3_memory(entries, key_vecs, base_vecs, forget_ids)
    forgotten_targets, retained_targets = select_dense_probe_slice(entries, forget_ids)
    if not print_entity_guard_validation():
        raise AssertionError("Tier 2C entity-anchor guard validation failed")
    print("\nTIER 2B RECHECK: N=100, entity-bound injection enabled (16 prompts)")
    print(f"{'Type':<9} | {'ID':<5} | {'Prediction':<19} | {'Candidate':<9} | {'Hybrid':>6} | {'EntityOK':>8} | {'Injected':>8} | {'V_m':>6} | {'Action':<6} | Outcome")
    print("-" * 132)
    baseline_vms = (-84.6, -84.6, -82.5, -70.3)
    torch.manual_seed(42)
    base_recall = 0
    for i, (question, aliases, _) in enumerate(WORLD_KNOWN[:4]):
        result = engine.infer_with_vm(question, memory=memory, v_crit=v_crit, query_vec=base_vecs[i])
        action = "HALT" if result["halt"] else "SUBMIT"
        base_recall += int(is_correct_answer(result["answer"], aliases) and action == "SUBMIT")
        print(f"{'Base':<9} | B{i+1:<4} | {result['answer'][:19]:<19} | {str(result['res_candidate_id'] or '-'):>9} | {result['res_sim']:6.3f} | {str(result['res_entity_match']):>8} | {str(result['res_injected']):>8} | {result['v_m']:6.1f} | {action:<6} | delta {result['v_m'] - baseline_vms[i]:+.1f} mV", flush=True)
    by_id = {entry["id"]: entry for entry in entries}
    metrics = Counter()
    for row_type, targets in (("Forgotten", forgotten_targets), ("Retained", retained_targets)):
        for entry in targets:
            result = engine.infer_with_vm(entry["para_q"], memory=memory, v_crit=v_crit, query_vec=para_vecs[entry["id"]])
            action = "HALT" if result["halt"] else "SUBMIT"
            correct = is_correct_answer(result["answer"], [entry["answer"]])
            candidate = result["res_candidate_id"]
            if row_type == "Forgotten":
                sibling = candidate is not None and candidate != entry["id"]
                sibling_answer = sibling and is_correct_answer(result["answer"], [by_id[candidate]["answer"]])
                metrics["direct_submit"] += int(correct and action == "SUBMIT")
                metrics["sibling_candidates"] += int(sibling)
                metrics["sibling_bleed"] += int(sibling_answer and action == "SUBMIT")
                metrics["entity_blocks"] += int(not result["res_entity_match"] and not result["res_injected"])
                metrics["safe_halt"] += int(action == "HALT" and not correct)
                outcome = "SIBLING BLEED" if sibling_answer and action == "SUBMIT" else ("HALTED" if action == "HALT" else "OTHER SUBMIT")
            else:
                metrics["retained_exact"] += int(candidate == entry["id"])
                metrics["retained_recall"] += int(correct and action == "SUBMIT")
                outcome = "RECALLED" if correct and action == "SUBMIT" else ("HALTED" if action == "HALT" else "WRONG")
            print(f"{row_type:<9} | {entry['id']:<5} | {result['answer'][:19]:<19} | {str(candidate or '-'):>9} | {result['res_sim']:6.3f} | {str(result['res_entity_match']):>8} | {str(result['res_injected']):>8} | {result['v_m']:6.1f} | {action:<6} | {outcome}", flush=True)
    print(f"Tier 2B summary | retained exact retrieval {metrics['retained_exact']}/6 | retained correct SUBMIT {metrics['retained_recall']}/6 | forgotten purged-answer SUBMIT {metrics['direct_submit']}/6 | sibling candidates {metrics['sibling_candidates']}/6 | entity mismatch blocks {metrics['entity_blocks']}/6 | sibling bleed submitted {metrics['sibling_bleed']}/6 | safe HALT {metrics['safe_halt']}/6 | base recall {base_recall}/4")
    print(f"Tier 2B recheck runtime: {time.perf_counter() - started:.1f}s on {engine.device}; index consistent: {memory.index_is_consistent()}.")


def _raw_cosine_nearest(query_vec: torch.Tensor, slots: dict):
    best_id, best_similarity, best_fact = None, -1.0, None
    for slot_id, entry in slots.items():
        similarity = float(torch.dot(query_vec, entry["key"]).item())
        if similarity > best_similarity:
            best_id, best_similarity, best_fact = slot_id, similarity, entry["fact"]
    return best_fact, best_similarity, best_id


def _greedy_rag_answer(engine: HRCLLMEngine, question: str, fact: str = None):
    prompt = engine._build_prompt(question, resonant_fact=fact)
    enc = engine.tokenizer(prompt, return_tensors="pt").to(engine.device)
    input_len = enc["input_ids"].shape[1]
    output = engine.model.generate(
        **enc,
        max_new_tokens=14,
        do_sample=False,
        pad_token_id=engine.tokenizer.eos_token_id,
    )
    return engine.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip().split("\n")[0]


def _evaluate_audit_method(rows: list, method: str, forgotten_ids: set):
    direct = wrong_submit = neighbor_bleed = clean_halt = retained_recall = 0
    forgotten_sims, retained_sims = [], []
    for row in rows:
        answer = row[f"{method}_answer"]
        action = row[f"{method}_action"]
        sim = row[f"{method}_sim"]
        episode = row["episode"]
        matched = is_correct_answer(answer, episode["aliases"])
        if episode["id"] in forgotten_ids:
            forgotten_sims.append(sim)
            direct += int(action == "SUBMIT" and matched)
            wrong_submit += int(action == "SUBMIT" and not matched)
            neighbor_bleed += int(
                action == "SUBMIT"
                and any(
                    other["id"] not in forgotten_ids
                    and is_correct_answer(answer, other["aliases"])
                    for other in PRIVATE_EPISODES
                )
            )
            clean_halt += int(action == "HALT")
        else:
            retained_sims.append(sim)
            retained_recall += int(action == "SUBMIT" and matched)
    margin = min(retained_sims) - max(forgotten_sims) if forgotten_sims and retained_sims else float("nan")
    return {
        "direct": direct,
        "wrong_submit": wrong_submit,
        "neighbor_bleed": neighbor_bleed,
        "clean_halt": clean_halt,
        "retained_recall": retained_recall,
        "max_forgotten_sim": max(forgotten_sims) if forgotten_sims else float("nan"),
        "min_retained_sim": min(retained_sims) if retained_sims else float("nan"),
        "margin": margin,
    }


def run_baselines_part_4b(v_crit: float = -66.0):
    """Compare raw-cosine greedy RAG with isotropy + V_m on both memory slices."""
    torch.manual_seed(42)
    total_started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    tau_res = 0.45
    world_questions = [question for question, _, _ in WORLD_KNOWN]

    # Phase 3: original E1-E6 store, purge E1-E3, then the same 12 audit prompts.
    audit_questions = [probe["question"] for probe in UNLEARN_AUDIT_PROBES]
    phase3_vectors = engine.embed_texts_batched(
        world_questions + [ep["q_orig"] for ep in PRIVATE_EPISODES] + audit_questions,
        batch_size=32,
        progress=True,
    )
    world_count = len(world_questions)
    episode_start = world_count
    audit_start = episode_start + len(PRIVATE_EPISODES)
    base_vecs = phase3_vectors[:world_count]
    episode_vecs = {
        ep["id"]: phase3_vectors[episode_start + i]
        for i, ep in enumerate(PRIVATE_EPISODES)
    }
    audit_vecs = {
        i: phase3_vectors[audit_start + i]
        for i in range(len(UNLEARN_AUDIT_PROBES))
    }
    forgotten_ids = {ep["id"] for ep in PRIVATE_EPISODES if ep["forget"]}
    memory = Layer3ResonantMemory(tau_res=tau_res)
    memory.fit_isotropy_center(base_vecs)
    raw_slots = {}
    for ep in PRIVATE_EPISODES:
        memory.consolidate(ep["id"], episode_vecs[ep["id"]], ep["fact"])
        raw_slots[ep["id"]] = {"key": episode_vecs[ep["id"]], "fact": ep["fact"]}
    for slot_id in forgotten_ids:
        memory.exact_unlearn(slot_id)
        raw_slots.pop(slot_id, None)

    phase3_started = time.perf_counter()
    phase3_rows = []
    print("\n" + "=" * 144)
    print("PART 4B-1: PHASE 3 UNLEARN AUDIT — RAW COSINE RAG vs. HRC (12 identical probes)")
    print("=" * 144)
    print(
        f"{'Type':<18} | {'ID':<3} | {'Raw Answer':<20} | {'Raw NN':<5} | {'Raw Sim':>7} | "
        f"{'HRC Answer':<20} | {'HRC NN':<5} | {'HRC Sim':>7} | {'HRC V_m':>7} | HRC"
    )
    print("-" * 144)
    for i, probe in enumerate(UNLEARN_AUDIT_PROBES):
        episode = next(ep for ep in PRIVATE_EPISODES if ep["id"] == probe["id"])
        q_vec = audit_vecs[i]
        raw_fact, raw_sim, raw_id = _raw_cosine_nearest(q_vec, raw_slots)
        raw_injected_fact = raw_fact if raw_sim >= tau_res else None
        raw_answer = _greedy_rag_answer(engine, probe["question"], raw_injected_fact)
        _, hrc_sim, hrc_nearest_id = memory.nearest(q_vec)
        hrc_result = engine.infer_with_vm(
            probe["question"], memory=memory, v_crit=v_crit, query_vec=q_vec
        )
        hrc_action = "HALT" if hrc_result["halt"] else "SUBMIT"
        row = {
            "probe": probe,
            "episode": episode,
            "raw_answer": raw_answer,
            "raw_action": "SUBMIT",
            "raw_sim": raw_sim,
            "raw_nearest_id": raw_id,
            "hrc_answer": hrc_result["answer"],
            "hrc_action": hrc_action,
            "hrc_sim": hrc_sim,
            "hrc_nearest_id": hrc_nearest_id,
            "hrc_v_m": hrc_result["v_m"],
        }
        phase3_rows.append(row)
        print(
            f"{probe['kind']:<18} | {probe['id']:<3} | {raw_answer[:20]:<20} | {str(raw_id or '-'):>5} | "
            f"{raw_sim:7.3f} | {hrc_result['answer'][:20]:<20} | {str(hrc_nearest_id or '-'):>5} | "
            f"{hrc_sim:7.3f} | {hrc_result['v_m']:7.1f} | {hrc_action}",
            flush=True,
        )
    phase3_elapsed = time.perf_counter() - phase3_started

    # The helper expects method-specific answer/action/sim fields.
    audit_eval_rows = []
    for row in phase3_rows:
        audit_eval_rows.append({
            "episode": row["episode"],
            "raw_answer": row["raw_answer"],
            "raw_action": row["raw_action"],
            "raw_sim": row["raw_sim"],
            "hrc_answer": row["hrc_answer"],
            "hrc_action": row["hrc_action"],
            "hrc_sim": row["hrc_sim"],
        })
    raw_audit_metrics = _evaluate_audit_method(audit_eval_rows, "raw", forgotten_ids)
    hrc_audit_metrics = _evaluate_audit_method(audit_eval_rows, "hrc", forgotten_ids)
    print("\nPhase 3 similarity + behavior comparison:")
    print(
        f"{'Method':<22} | {'Direct /9':>9} | {'Wrong submit /9':>15} | {'Neighbor bleed /9':>17} | "
        f"{'Clean HALT /9':>13} | {'Retained /3':>11} | {'Max Fgt':>7} | {'Min Ret':>7} | {'Margin':>7}"
    )
    print("-" * 133)
    for method, metrics in (("Raw cosine RAG", raw_audit_metrics), ("HRC", hrc_audit_metrics)):
        print(
            f"{method:<22} | {metrics['direct']:>4}/9 | {metrics['wrong_submit']:>8}/9 | "
            f"{metrics['neighbor_bleed']:>10}/9 | {metrics['clean_halt']:>6}/9 | "
            f"{metrics['retained_recall']:>5}/3 | {metrics['max_forgotten_sim']:7.3f} | "
            f"{metrics['min_retained_sim']:7.3f} | {metrics['margin']:+7.3f}"
        )
    print(f"Phase 3 comparison runtime: {phase3_elapsed:.1f}s ({phase3_elapsed / 12:.2f}s/probe across both methods).")

    # Tier 2B: exactly the same N=100 target slice used by --dense-scale.
    entries, forget_ids = generate_dense_cluster_memories(100)
    all_texts = (
        world_questions
        + [entry["store_q"] for entry in entries]
        + [entry["para_q"] for entry in entries]
    )
    vectors = engine.embed_texts_batched(all_texts, batch_size=32, progress=True)
    base_vecs_100 = vectors[:world_count]
    key_start = world_count
    para_start = key_start + len(entries)
    key_vecs = {entry["id"]: vectors[key_start + i] for i, entry in enumerate(entries)}
    para_vecs = {entry["id"]: vectors[para_start + i] for i, entry in enumerate(entries)}
    dense_memory, _ = build_dense_memory(100, entries, forget_ids, key_vecs, base_vecs_100)
    raw_dense_slots = {
        entry["id"]: {"key": key_vecs[entry["id"]], "fact": entry["answer"]}
        for entry in entries
    }
    for slot_id in forget_ids:
        raw_dense_slots.pop(slot_id, None)
    forgotten_targets, retained_targets = select_dense_probe_slice(entries, forget_ids)
    entry_by_id = {entry["id"]: entry for entry in entries}
    dense_probes = [
        {"kind": "Base", "id": f"B{i + 1}", "question": q, "query_vec": base_vecs_100[i], "entry": None}
        for i, (q, _, _) in enumerate(WORLD_KNOWN[:4])
    ]
    dense_probes += [
        {"kind": "Forgotten", "id": entry["id"], "question": entry["para_q"],
         "query_vec": para_vecs[entry["id"]], "entry": entry}
        for entry in forgotten_targets
    ]
    dense_probes += [
        {"kind": "Retained", "id": entry["id"], "question": entry["para_q"],
         "query_vec": para_vecs[entry["id"]], "entry": entry}
        for entry in retained_targets
    ]

    dense_started = time.perf_counter()
    dense_rows = []
    print("\n" + "=" * 144)
    print("PART 4B-2: TIER 2B N=100 — RAW COSINE RAG vs. HRC (same 16-probe slice)")
    print("=" * 144)
    print(
        f"{'Type':<9} | {'ID':<5} | {'Raw Answer':<19} | {'Raw NN':<5} | {'Raw Sim':>7} | "
        f"{'HRC Answer':<19} | {'HRC NN':<5} | {'HRC Sim':>7} | {'HRC V_m':>7} | HRC"
    )
    print("-" * 144)
    for probe in dense_probes:
        raw_fact, raw_sim, raw_id = _raw_cosine_nearest(probe["query_vec"], raw_dense_slots)
        raw_answer = _greedy_rag_answer(
            engine,
            probe["question"],
            raw_fact if raw_sim >= tau_res else None,
        )
        _, hrc_nearest_sim, hrc_nearest_id = dense_memory.nearest(probe["query_vec"])
        hrc_result = engine.infer_with_vm(
            probe["question"],
            memory=dense_memory,
            v_crit=v_crit,
            query_vec=probe["query_vec"],
        )
        hrc_action = "HALT" if hrc_result["halt"] else "SUBMIT"
        row = {
            **probe,
            "raw_answer": raw_answer,
            "raw_action": "SUBMIT",
            "raw_sim": raw_sim,
            "raw_nearest_id": raw_id,
            "hrc_answer": hrc_result["answer"],
            "hrc_action": hrc_action,
            "hrc_sim": hrc_nearest_sim,
            "hrc_nearest_id": hrc_nearest_id,
            "hrc_v_m": hrc_result["v_m"],
        }
        dense_rows.append(row)
        print(
            f"{probe['kind']:<9} | {probe['id']:<5} | {raw_answer[:19]:<19} | {str(raw_id or '-'):>5} | "
            f"{raw_sim:7.3f} | {hrc_result['answer'][:19]:<19} | {str(hrc_nearest_id or '-'):>5} | "
            f"{hrc_nearest_sim:7.3f} | {hrc_result['v_m']:7.1f} | {hrc_action}",
            flush=True,
        )
    dense_elapsed = time.perf_counter() - dense_started

    dense_summary = {}
    for method in ("raw", "hrc"):
        values = {"base_recall": 0, "retained_recall": 0, "direct": 0,
                  "neighbor_retrieved": 0, "sibling_bleed": 0, "safe_halt": 0}
        for row in dense_rows:
            answer = row[f"{method}_answer"]
            action = row[f"{method}_action"]
            nearest_id = row[f"{method}_nearest_id"]
            entry = row["entry"]
            if row["kind"] == "Base":
                _, aliases, _ = WORLD_KNOWN[int(row["id"][1:]) - 1]
                values["base_recall"] += int(action == "SUBMIT" and is_correct_answer(answer, aliases))
            elif row["kind"] == "Retained":
                values["retained_recall"] += int(
                    action == "SUBMIT" and is_correct_answer(answer, [entry["answer"]])
                )
            else:
                direct = is_correct_answer(answer, [entry["answer"]])
                neighbor = entry_by_id.get(nearest_id)
                sibling_match = bool(
                    neighbor
                    and neighbor["id"] != entry["id"]
                    and neighbor["cluster"] == entry["cluster"]
                    and is_correct_answer(answer, [neighbor["answer"]])
                )
                values["direct"] += int(action == "SUBMIT" and direct)
                values["neighbor_retrieved"] += int(
                    neighbor is not None and neighbor["id"] != entry["id"]
                )
                values["sibling_bleed"] += int(action == "SUBMIT" and sibling_match)
                values["safe_halt"] += int(action == "HALT" and not direct)
        dense_summary[method] = values

    print("\nTier 2B behavior comparison:")
    print(
        f"{'Method':<22} | {'Base /4':>7} | {'Retained /6':>11} | {'Direct /6':>9} | "
        f"{'Neighbor NN /6':>13} | {'Sibling bleed /6':>16} | {'Safe HALT /6':>12}"
    )
    print("-" * 112)
    for method, label in (("raw", "Raw cosine RAG"), ("hrc", "HRC")):
        values = dense_summary[method]
        print(
            f"{label:<22} | {values['base_recall']:>3}/4 | {values['retained_recall']:>6}/6 | "
            f"{values['direct']:>4}/6 | {values['neighbor_retrieved']:>8}/6 | "
            f"{values['sibling_bleed']:>11}/6 | {values['safe_halt']:>7}/6"
        )
    print(f"Tier 2B comparison runtime: {dense_elapsed:.1f}s ({dense_elapsed / 16:.2f}s/probe across both methods).")
    print(f"Full --baselines runtime: {time.perf_counter() - total_started:.1f}s on {engine.device}.")


def run_baselines(part: str = "all", v_crit: float = -66.0):
    if part in ("4a", "all"):
        run_baselines_part_4a(v_crit=v_crit)
    if part in ("4b", "all"):
        run_baselines_part_4b(v_crit=v_crit)


def resolve_yes_no_token_ids(tokenizer):
    """Return all available single-token casing/spacing variants for Yes and No."""
    yes_ids, no_ids = set(), set()
    for token in ("Yes", "yes", " Yes", " yes", "YES"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            yes_ids.add(token_ids[0])
    for token in ("No", "no", " No", " no", "NO"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            no_ids.add(token_ids[0])
    if not yes_ids or not no_ids or yes_ids & no_ids:
        raise ValueError("Could not resolve disjoint single-token Yes/No candidates.")
    return sorted(yes_ids), sorted(no_ids)


def build_stage2_messages(candidate: str, question: str, proposed_answer: str = ""):
    if candidate == "2A":
        user_content = (
            f"Question: {question}\n"
            f"Proposed Answer: {proposed_answer}\n"
            "Is the proposed answer factually correct? Answer Yes or No:"
        )
        return [
            {"role": "system", "content": SYS_VERIFIER},
            {"role": "user", "content": user_content},
        ]
    if candidate == "2B":
        return [
            {"role": "system", "content": SYS_VERIFIER},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    "Does this question contain a false premise or factual impossibility? Answer Yes or No:"
                ),
            },
        ]
    if candidate == "2C":
        messages = [{"role": "system", "content": SYS_VERIFIER}]
        for example_question, example_answer in TWO_SHOT_PREMISE_EXAMPLES:
            messages.extend([
                {"role": "user", "content": example_question},
                {"role": "assistant", "content": example_answer},
            ])
        messages.append({
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Does this question contain a false premise or factual impossibility? Answer Yes or No:"
            ),
        })
        return messages
    raise ValueError(f"Unknown Stage-2 candidate: {candidate}")


@torch.inference_mode()
def score_stage2_single_pass(engine, candidate: str, question: str, proposed_answer: str,
                             yes_ids: list, no_ids: list):
    """Score one chat-formatted candidate with one model forward and no decoding."""
    messages = build_stage2_messages(candidate, question, proposed_answer)
    prompt_text = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = engine.tokenizer([prompt_text], return_tensors="pt").to(engine.device)

    started = time.perf_counter()
    logits = engine.model(**inputs).logits[0, -1, :].float()
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_probs = F.log_softmax(logits, dim=-1)
    yes_t = torch.tensor(yes_ids, dtype=torch.long, device=logits.device)
    no_t = torch.tensor(no_ids, dtype=torch.long, device=logits.device)
    log_yes = torch.logsumexp(log_probs[yes_t], dim=0)
    log_no = torch.logsumexp(log_probs[no_t], dim=0)
    p_yes = float(torch.sigmoid(log_yes - log_no).item())
    raw_mass = float((torch.exp(log_yes) + torch.exp(log_no)).item())
    s_valid = p_yes if candidate == "2A" else 1.0 - p_yes
    return {
        "s_valid": s_valid,
        "p_yes": p_yes,
        "logit_diff": float((log_yes - log_no).item()),
        "raw_mass": raw_mass,
        "latency_ms": latency_ms,
    }


def fit_stage2_threshold(records: list, max_valid_drops: int = 1):
    """Fit a validity threshold only on Phase 1 V_m-passing examples."""
    valid_scores = sorted(row["s_valid"] for row in records if row["is_valid"])
    trap_scores = sorted(row["s_valid"] for row in records if not row["is_valid"])
    if not valid_scores or not trap_scores:
        return {
            "tau_veto": None,
            "margin": float("nan"),
            "min_valid": min(valid_scores) if valid_scores else float("nan"),
            "max_trap": max(trap_scores) if trap_scores else float("nan"),
            "auroc": float("nan"),
        }

    min_valid, max_trap = min(valid_scores), max(trap_scores)
    margin = min_valid - max_trap
    if margin > 0:
        tau_veto = 0.5 * (min_valid + max_trap)
    else:
        cutoff_idx = min(max_valid_drops, len(valid_scores) - 1)
        tau_veto = valid_scores[cutoff_idx] - 1e-4
    # `compute_auroc` compares its first group as the positive/high-scoring class.
    auroc = compute_auroc(valid_scores, trap_scores)
    return {
        "tau_veto": tau_veto,
        "margin": margin,
        "min_valid": min_valid,
        "max_trap": max_trap,
        "auroc": auroc,
    }


def evaluate_stage2_records(records: list, tau_veto):
    valid_rows = [row for row in records if row["is_valid"]]
    trap_rows = [row for row in records if not row["is_valid"]]
    if tau_veto is None:
        valid_kept = traps_halted = 0
    else:
        valid_kept = sum(row["s_valid"] >= tau_veto for row in valid_rows)
        traps_halted = sum(row["s_valid"] < tau_veto for row in trap_rows)
    traps_submitted = len(trap_rows) - traps_halted
    submitted = valid_kept + traps_submitted
    return {
        "valid_kept": valid_kept,
        "valid_total": len(valid_rows),
        "traps_halted": traps_halted,
        "traps_total": len(trap_rows),
        "selective_acc": valid_kept / submitted if submitted else 0.0,
        "mean_raw_mass": (
            sum(row["raw_mass"] for row in records) / len(records) if records else 0.0
        ),
        "mean_latency_ms": (
            sum(row["latency_ms"] for row in records) / len(records) if records else 0.0
        ),
    }


def _m2_l2_stage1_cache_path(
    v_crit: float,
    model_id: str = MODEL_ID,
    dtype_name: str = None,
    cache_dir: str | Path = None,
    device_name: str = None,
):
    resolved_device, resolved_dtype = _resolve_device_dtype(
        ACTIVE_DEVICE if device_name is None else device_name,
        ACTIVE_DTYPE if dtype_name is None else dtype_name,
    )
    dtype_name = _dtype_name(resolved_dtype) if dtype_name is None else dtype_name
    model_tag = model_id.replace("/", "--").replace(".", "_")
    threshold_tag = str(v_crit).replace("-", "m").replace(".", "p")
    base_dir = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
    return base_dir / f"hrc_mvp_stage1_{model_tag}_{dtype_name}_{resolved_device.type}_{threshold_tag}.json"


def _m2_l2_phase1_stream():
    return (
        WORLD_KNOWN
        + OBSCURE_HARD
        + [(episode["q_orig"], episode["aliases"], "Private") for episode in PRIVATE_EPISODES]
        + [(probe["question"], probe["gold"], probe["bucket"]) for probe in PHASE1_ADVERSARIAL_PROBES]
    )


def _validate_m2_l2_holdout_disjoint(phase1_stream=None):
    phase1_stream = phase1_stream if phase1_stream is not None else _m2_l2_phase1_stream()
    phase1_by_normalized = {normalize_text(question): question for question, _, _ in phase1_stream}
    duplicates = [
        (phase1_by_normalized[normalized], probe["q"])
        for probe in M2_L2_HOLDOUT_QUERIES
        if (normalized := normalize_text(probe["q"])) in phase1_by_normalized
    ]
    if duplicates:
        details = "; ".join(f"Phase 1 {p1!r} == holdout {holdout!r}" for p1, holdout in duplicates)
        raise ValueError(f"M2_L2 holdout queries must be disjoint from Phase 1: {details}")
    holdout_seen = {}
    duplicate_holdout = []
    for probe in M2_L2_HOLDOUT_QUERIES:
        normalized = normalize_text(probe["q"])
        if normalized in holdout_seen:
            duplicate_holdout.append((holdout_seen[normalized], probe["q"]))
        else:
            holdout_seen[normalized] = probe["q"]
    if duplicate_holdout:
        details = "; ".join(f"{first!r} == {second!r}" for first, second in duplicate_holdout)
        raise ValueError(f"M2_L2 holdout queries must be unique: {details}")
    return phase1_stream


def _read_m2_l2_stage1_cache(
    v_crit: float,
    allow_legacy: bool = False,
    model_id: str = MODEL_ID,
    dtype_name: str = None,
    cache_dir: str | Path = None,
    device_name: str = None,
):
    phase1_stream = _validate_m2_l2_holdout_disjoint()
    expected_queries = [question for question, _, _ in phase1_stream]
    expected_queries.extend(probe["q"] for probe in M2_L2_HOLDOUT_QUERIES)
    resolved_device, resolved_dtype = _resolve_device_dtype(
        ACTIVE_DEVICE if device_name is None else device_name,
        ACTIVE_DTYPE if dtype_name is None else dtype_name,
    )
    requested_dtype = _dtype_name(resolved_dtype) if dtype_name is None else dtype_name
    cache_path = _m2_l2_stage1_cache_path(
        v_crit, model_id=model_id, dtype_name=requested_dtype,
        cache_dir=cache_dir, device_name=resolved_device.type,
    )
    legacy_path = None
    selected_path = cache_path
    if not selected_path.is_file() and resolved_device.type == "cpu":
        model_tag = model_id.replace("/", "--").replace(".", "_")
        threshold_tag = str(v_crit).replace("-", "m").replace(".", "p")
        base_dir = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
        legacy_path = base_dir / f"hrc_mvp_stage1_{model_tag}_{requested_dtype}_{threshold_tag}.json"
        if legacy_path.is_file():
            selected_path = legacy_path
    if not selected_path.is_file():
        return None
    try:
        cached = json.loads(selected_path.read_text(encoding="utf-8"))
        valid_seed_policy = cached.get("seed_policy") == M2_L2_SEED_POLICY
        is_legacy_path = selected_path == legacy_path
        if (
            cached.get("schema") == 2
            and cached.get("model_id") == model_id
            and cached.get("v_crit") == v_crit
            and (cached.get("dtype") in (None, requested_dtype) if is_legacy_path else cached.get("dtype") == requested_dtype)
            and (cached.get("device") in (None, "cpu") if is_legacy_path else cached.get("device") == resolved_device.type)
            and (valid_seed_policy or allow_legacy)
        ):
            records = cached.get("records", [])
            if len(records) == len(expected_queries) and [row.get("query") for row in records] == expected_queries:
                return records
    except (OSError, ValueError, TypeError):
        pass
    return None


def _print_m2_l2_stage1_trap_audit(stage1_results: list, label: str = "Stage 1"):
    """Show which false-premise answers explicitly reject their premise."""
    print(f"\n--- {label}: false-premise response audit ---", flush=True)
    for split in ("P1", "Holdout"):
        traps = [
            row for row in stage1_results
            if row.get("split") == split and not row.get("is_valid")
        ]
        if not traps:
            continue
        explicit = [
            row for row in traps
            if explicitly_rejects_premise(row.get("answer", ""))
        ]
        print(
            f"[{split}] explicit premise rejections={len(explicit)}/{len(traps)}; "
            f"unflagged responses={len(traps) - len(explicit)}",
            flush=True,
        )
        for row in traps:
            rejected = explicitly_rejects_premise(row.get("answer", ""))
            stage1_status = "PASS" if row.get("stage1_pass") else "HALT"
            response_status = "EXPLICIT_REJECTION" if rejected else "UNFLAGGED"
            print(
                f"  {row.get('id', '?'):<7} {stage1_status:<4} "
                f"{response_status:<18} answer={row.get('answer', '')!r}",
                flush=True,
            )


def _m2_l2_stage2_candidates(stage1_results: list) -> list:
    """Select Stage-1 passers that need Stage 2; retain valid queries and trap leaks."""
    return [
        row for row in stage1_results
        if row.get("stage1_pass")
        and (row.get("is_valid") or not explicitly_rejects_premise(row.get("answer", "")))
    ]


def _m2_l2_stage1_fingerprint(stage1_records: list) -> str:
    """Hash the canonical Stage-1 record sequence used by Stage 2."""
    canonical = json.dumps(
        stage1_records, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _m2_l2_stage2_cache_path(
    stage1_records: list,
    stage2_model_id: str,
    stage2_dtype_name: str,
    stage2_device: str,
    dual_polarity: bool,
    cache_dir: str | Path = ".hrc_cache",
) -> tuple[Path, dict]:
    """Return the deterministic path and complete metadata identity for Stage 2."""
    stage1_hash = _m2_l2_stage1_fingerprint(stage1_records)
    model_slug = stage2_model_id.replace("/", "--").replace(".", "_")
    metadata = {
        "stage1_cache_sha256": stage1_hash,
        "stage2_model": stage2_model_id,
        "stage2_dtype": stage2_dtype_name,
        "stage2_device": stage2_device,
        "prompt_version": M2_L2_STAGE2_PROMPT_VERSION,
        "dual_polarity": bool(dual_polarity),
    }
    path = Path(cache_dir) / (
        f"hrc_mvp_stage2_{model_slug}_{stage2_dtype_name}_{stage2_device}_"
        f"{M2_L2_STAGE2_PROMPT_VERSION}_s1-{stage1_hash[:10]}.json"
    )
    return path, metadata


def _read_m2_l2_stage2_cache(cache_path: str | Path, expected_metadata: dict) -> dict:
    """Read a matching cache as an index keyed by (split, id, query)."""
    path = Path(cache_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if payload.get("schema") != 1 or any(
        payload.get(key) != value for key, value in expected_metadata.items()
    ):
        return {}
    indexed = {}
    for row in payload.get("records", []):
        try:
            key = (row["split"], row["id"], row["query"])
            if not all(name in row for name in ("is_valid", "d_pos", "d_neg", "s_bal", "latency_s")):
                continue
            indexed[key] = row
        except (KeyError, TypeError):
            continue
    return indexed


def _write_m2_l2_stage2_cache(cache_path: str | Path, metadata: dict, records: list):
    """Atomically replace a Stage-2 cache after one or more scored queries."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {"schema": 1, **metadata, "records": records}
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def collect_m2_l2_stage1(
    engine,
    v_crit: float = -66.0,
    model_id: str = MODEL_ID,
    dtype_name: str = None,
    device_name: str = None,
    cache_dir: str | Path = None,
):
    """Run Stage 1 once and cache its records for the diagnostic and asymmetric runs."""
    cache_path = _m2_l2_stage1_cache_path(
        v_crit, model_id=model_id, dtype_name=dtype_name,
        cache_dir=cache_dir, device_name=device_name,
    )
    cached_records = _read_m2_l2_stage1_cache(
        v_crit, model_id=model_id, dtype_name=dtype_name,
        cache_dir=cache_dir, device_name=device_name,
    )
    if cached_records is not None:
        print(f"Reusing cached Stage-1 outcomes from {cache_path}.", flush=True)
        return cached_records

    phase1_stream = _validate_m2_l2_holdout_disjoint()
    records = []
    print(
        f"Running Stage 1 once for Phase 1 ({len(phase1_stream)}) and "
        f"holdout ({len(M2_L2_HOLDOUT_QUERIES)})...",
        flush=True,
    )
    for index, (question, gold, bucket) in enumerate(phase1_stream, start=1):
        result = engine.infer_with_vm(question, v_crit=v_crit)
        records.append({
            "id": f"P1_{index:02d}", "split": "P1", "label": "FalsePremise" if gold is None else "Valid",
            "is_valid": gold is not None, "query": question, "answer": result["answer"],
            "answer_correct": is_correct_answer(result["answer"], gold) if gold is not None else None,
            "v_m": float(result["v_m"]), "stage1_pass": not bool(result["halt"]),
            "vm_components": _m2_l2_vm_components(result),
        })
        print(f"  P1 {index:02d}/40 {bucket:<13} V_m={result['v_m']:6.1f} "
              f"{'PASS' if not result['halt'] else 'HALT'}: {question[:58]}", flush=True)
    for probe in M2_L2_HOLDOUT_QUERIES:
        result = engine.infer_with_vm(probe["q"], v_crit=v_crit)
        records.append({
            "id": probe["id"], "split": "Holdout", "label": probe["category"],
            "is_valid": probe["category"] != "FalsePremise", "query": probe["q"],
            "answer": result["answer"], "v_m": float(result["v_m"]),
            "answer_correct": is_correct_answer(result["answer"], probe["gold"])
            if probe["category"] != "FalsePremise" else None,
            "stage1_pass": not bool(result["halt"]),
            "vm_components": _m2_l2_vm_components(result),
        })
        print(f"  Holdout {probe['id']} {probe['category']:<13} V_m={result['v_m']:6.1f} "
              f"{'PASS' if not result['halt'] else 'HALT'}: {probe['q']}", flush=True)
    try:
        cache_device, cache_dtype = _resolve_device_dtype(
            ACTIVE_DEVICE if device_name is None else device_name,
            ACTIVE_DTYPE if dtype_name is None else dtype_name,
        )
        cache_payload = {
            "schema": 2, "model_id": model_id, "v_crit": v_crit,
            "seed_policy": M2_L2_SEED_POLICY,
            "dtype": dtype_name or _dtype_name(cache_dtype),
            "device": cache_device.type,
            "records": records,
        }
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
        print(f"Cached Stage-1 outcomes at {cache_path}.", flush=True)
    except OSError as exc:
        print(f"Could not cache Stage-1 outcomes: {exc}", flush=True)
    return records


def run_m2_l2_stage1_model_compare(
    model_id: str,
    v_crit: float = -66.0,
    force_offline: bool = True,
    dtype=torch.bfloat16,
    device_name: str = None,
    cache_dir: str | Path = None,
):
    """Run the frozen Stage-1 split with another model and a model-specific cache."""
    torch.manual_seed(M2_L2_BASE_SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    dtype_name = str(dtype).replace("torch.", "")
    resolved_device, dtype = _resolve_device_dtype(
        ACTIVE_DEVICE if device_name is None else device_name, dtype
    )
    cache_path = _m2_l2_stage1_cache_path(
        v_crit, model_id=model_id, dtype_name=dtype_name,
        cache_dir=cache_dir, device_name=resolved_device.type,
    )
    records = _read_m2_l2_stage1_cache(
        v_crit, model_id=model_id, dtype_name=dtype_name,
        cache_dir=cache_dir, device_name=resolved_device.type,
    )
    if records is None:
        if force_offline and not _has_complete_local_hf_checkpoint(model_id):
            raise FileNotFoundError(
                f"A complete local checkpoint for {model_id} was not found; "
                "the comparison was requested in offline mode."
            )
        print(
            f"\n=== STAGE-1 MODEL COMPARISON: {model_id} ===\n"
            f"Split: locked Phase 1 ({len(_m2_l2_phase1_stream())}) + "
            f"holdout ({len(M2_L2_HOLDOUT_QUERIES)}); dtype={dtype}; "
            f"seed policy={M2_L2_SEED_POLICY}; cache={cache_path}",
            flush=True,
        )
        engine = HRCLLMEngine(
            model_id, force_offline=force_offline, dtype=dtype,
            device=resolved_device.type,
        )
        records = collect_m2_l2_stage1(
            engine, v_crit=v_crit, model_id=model_id, dtype_name=dtype_name,
            device_name=resolved_device.type, cache_dir=cache_dir,
        )
        del engine
        gc.collect()
    else:
        print(f"Reusing model-specific Stage-1 cache at {cache_path}.", flush=True)

    for split in ("P1", "Holdout"):
        subset = [row for row in records if row["split"] == split]
        valid = [row for row in subset if row["is_valid"]]
        traps = [row for row in subset if not row["is_valid"]]
        valid_pass = [row for row in valid if row["stage1_pass"]]
        valid_halt = [row for row in valid if not row["stage1_pass"]]
        correct_pass = sum(bool(row.get("answer_correct")) for row in valid_pass)
        wrong_pass = sum(row.get("answer_correct") is False for row in valid_pass)
        correct_halt = sum(bool(row.get("answer_correct")) for row in valid_halt)
        wrong_halt = sum(row.get("answer_correct") is False for row in valid_halt)
        trap_halt = sum(not row["stage1_pass"] for row in traps)
        print(
            f"[{split}] valid passed={len(valid_pass)}/{len(valid)} "
            f"(correct={correct_pass}, wrong={wrong_pass}); "
            f"valid halted={len(valid_halt)} (false abstentions={correct_halt}, "
            f"justified wrong-answer halts={wrong_halt}); "
            f"traps halted={trap_halt}/{len(traps)}; "
            f"trap passers={len(traps) - trap_halt}/{len(traps)}",
            flush=True,
        )
    _print_m2_l2_stage1_trap_audit(records, label=f"{model_id} Stage 1")
    print(
        "Interpretation: Stage-1 consistency cannot detect a systematic, "
        "high-confidence misconception when all sampled trajectories agree.",
        flush=True,
    )


def _m2_l2_vm_components(result: dict) -> dict:
    """Return JSON-safe component values that determine the Stage-1 V_m score."""
    keys = (
        "phi_lex", "phi_emb", "phi_emb_raw", "phi_sem", "ent",
        "mean_token_entropy", "mean_margin", "margin_stress", "stress",
        "semantic_entropy", "mean_seq_logprob", "sampling_seed", "trajectory_answers",
    )
    return {key: result[key] for key in keys if key in result}


@torch.inference_mode()
def run_m2_l2_component_diagnostic(
    v_crit: float = -66.0,
    force_offline: bool = True,
    target_ids: tuple[str, ...] = ("H_V1", "H_V11", "H_V2"),
):
    """Audit cached valid answers, then rerun only the selected Stage-1 queries."""
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    cache_path = _m2_l2_stage1_cache_path(v_crit)
    stage1_results = _read_m2_l2_stage1_cache(v_crit, allow_legacy=True)
    if stage1_results is None:
        raise FileNotFoundError(
            f"Current Stage-1 cache not found or stale at {cache_path}; "
            "this diagnostic does not regenerate the full split."
        )

    # Attach gold-grounded correctness to every valid cached answer before model loading.
    gold_by_query = {
        question: gold for question, gold, _ in _m2_l2_phase1_stream() if gold is not None
    }
    gold_by_query.update({probe["q"]: probe["gold"] for probe in M2_L2_HOLDOUT_QUERIES})
    for row in stage1_results:
        row["answer_correct"] = (
            is_correct_answer(row.get("answer", ""), gold_by_query[row["query"]])
            if row.get("is_valid") and row.get("query") in gold_by_query else None
        )

    valid_rows = [row for row in stage1_results if row.get("is_valid")]
    holdout_valid_rows = [row for row in valid_rows if row.get("split") == "Holdout"]
    print("\n=== ANSWER-GROUNDED AUDIT OF CACHED HOLDOUT VALID ITEMS ===", flush=True)
    print("ID      | Stage 1 | Correct | Answer", flush=True)
    for row in holdout_valid_rows:
        status = "PASS" if row["stage1_pass"] else "HALT"
        correctness = "YES" if row["answer_correct"] else "NO"
        print(f"{row['id']:<7} | {status:<7} | {correctness:<7} | {row.get('answer', '')}", flush=True)
    justified = sum(not row["stage1_pass"] and not row["answer_correct"] for row in holdout_valid_rows)
    false_abstentions = sum(not row["stage1_pass"] and row["answer_correct"] for row in holdout_valid_rows)
    wrong_submissions = sum(row["stage1_pass"] and not row["answer_correct"] for row in holdout_valid_rows)
    correct_submissions = sum(row["stage1_pass"] and row["answer_correct"] for row in holdout_valid_rows)
    print(
        f"Holdout valid totals: {correct_submissions} correct submissions, {wrong_submissions} wrong submissions, "
        f"{justified} justified halts, {false_abstentions} false abstentions.",
        flush=True,
    )

    targets = [row for row in stage1_results if row["id"] in target_ids]
    missing = set(target_ids) - {row["id"] for row in targets}
    if missing:
        raise ValueError(f"Requested diagnostic IDs are missing from the cache: {sorted(missing)}")

    print(
        f"\nRerunning {len(targets)} cached queries only: {', '.join(target_ids)}.\n"
        f"Cache: {cache_path}", flush=True,
    )
    engine = HRCLLMEngine(MODEL_ID, force_offline=force_offline)
    for index, row in enumerate(targets, start=1):
        result = engine.infer_with_vm(row["query"], v_crit=v_crit)
        components = _m2_l2_vm_components(result)
        row.update({
            "diagnostic_answer": result["answer"],
            "diagnostic_answer_correct": is_correct_answer(
                result["answer"], gold_by_query[row["query"]]
            ),
            "diagnostic_v_m": float(result["v_m"]),
            "diagnostic_stage1_pass": not bool(result["halt"]),
            "diagnostic_vm_components": components,
        })
        print(
            f"\n[{index}/{len(targets)}] {row['id']} | answer={result['answer']!r} | "
            f"correct={row['diagnostic_answer_correct']} | V_m={result['v_m']:.6f} | "
            f"phi_lex={result['phi_lex']:.6f} phi_emb={result['phi_emb']:.6f} "
            f"phi_sem={result['phi_sem']:.6f} ent_norm={result['ent']:.6f} "
            f"mean_token_entropy={result['mean_token_entropy']:.6f} "
            f"mean_margin={result['mean_margin']:.6f} margin_stress={result['margin_stress']:.6f} "
            f"stress={result['stress']:.6f}",
            flush=True,
        )
        print(f"  trajectories={json.dumps(result['trajectory_answers'], ensure_ascii=False)}", flush=True)

    try:
        cache_path.write_text(json.dumps({
            "schema": 2, "model_id": MODEL_ID, "v_crit": v_crit, "records": stage1_results,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"Updated answer correctness and component diagnostics in {cache_path}.", flush=True)
    except OSError as exc:
        print(f"Could not update Stage-1 cache: {exc}", flush=True)

    del engine
    gc.collect()


def _stage2_yes_no_logits(model, tokenizer, prompt: str, yes_ids: list, no_ids: list):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    log_probs = F.log_softmax(logits, dim=-1)
    yes_tensor = torch.tensor(yes_ids, dtype=torch.long, device=logits.device)
    no_tensor = torch.tensor(no_ids, dtype=torch.long, device=logits.device)
    log_yes = torch.logsumexp(log_probs[yes_tensor], dim=0)
    log_no = torch.logsumexp(log_probs[no_tensor], dim=0)
    diff = float((log_yes - log_no).item())
    return diff, float(torch.sigmoid(torch.tensor(diff)).item()), float((log_yes.exp() + log_no.exp()).item())


def _chat_prompt(tokenizer, user_text: str):
    messages = [
        {"role": "system", "content": "You are a strict factual auditor. Reply with a single word: Yes or No."},
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{user_text}\nAnswer:"


M2_L2_ATOMIC_FACTS = {
    "P1_35": ("Did Albert Einstein win a Nobel Prize for a novel?", "No"),
    "P1_36": ("Did the United States join the European Union?", "No"),
    "P1_37": ("Does the continent of Africa have a single capital city?", "No"),
    "P1_38": ("Did Shakespeare write a play featuring Sherlock Holmes?", "No"),
    "P1_39": ("Did Apollo 11 land on Mars?", "No"),
    "P1_40": ("Is there an element with atomic number zero on the periodic table?", "No"),
    "H_FP1": ("Did Napoleon win the Battle of Waterloo?", "No"),
    "H_FP2": ("Does Switzerland border an ocean?", "No"),
    "H_FP3": ("Did the Titanic sink in the Pacific Ocean?", "No"),
    "H_FP4": ("Does Antarctica have a president?", "No"),
    "H_FP5": ("Did Albert Einstein win the Nobel Prize in Literature?", "No"),
    "H_FP6": ("Was a FIFA World Cup held in 1942 with a winning country?", "No"),
}


@torch.inference_mode()
def run_m2_l2_diagnostic(v_crit: float = -66.0):
    """Measure dual-polarity premise scoring and direct recall on Stage-1 passers."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    stage1 = collect_m2_l2_stage1(engine, v_crit)
    _print_m2_l2_stage1_trap_audit(stage1)
    passers = _m2_l2_stage2_candidates(stage1)
    yes_ids, no_ids = resolve_yes_no_token_ids(engine.tokenizer)
    rows = []
    print(
        f"\n=== M2 LAYER 2 DIAGNOSTIC: DUAL-POLARITY & DIRECT RECALL (0.5B) ===\n"
        f"Stage-2 premise prompt: {M2_L2_STAGE2_PROMPT_VERSION}"
    )
    for item in passers:
        pos_prompt = _format_instruct_premise_prompt(engine.tokenizer, item["query"], negate=False)
        neg_prompt = _format_instruct_premise_prompt(engine.tokenizer, item["query"], negate=True)
        d_pos, p_pos, mass_pos = _stage2_yes_no_logits(engine.model, engine.tokenizer, pos_prompt, yes_ids, no_ids)
        d_neg, p_neg, mass_neg = _stage2_yes_no_logits(engine.model, engine.tokenizer, neg_prompt, yes_ids, no_ids)
        rows.append({**item, "d_pos": d_pos, "d_neg": d_neg,
                     "s_bal": 0.5 * (d_pos - d_neg), "p_pos": p_pos,
                     "p_neg": p_neg, "mass": 0.5 * (mass_pos + mass_neg)})

    print(f"Stage-1 passers scored: {len(passers)} (Phase 1 {sum(r['split'] == 'P1' for r in passers)}, "
          f"holdout {sum(r['split'] == 'Holdout' for r in passers)}).")
    p1_scored = [row for row in rows if row["split"] == "P1"]
    cutoff, threshold_rule, margin = _fit_frozen_phase1_cutoff(p1_scored, "s_bal")
    print(
        f"Locked Phase 1 cutoff: {cutoff:+.3f} via {threshold_rule}; "
        f"Phase 1 margin={margin:+.3f}. Holdout is evaluated without refitting.",
        flush=True,
    )
    for split in ("P1", "Holdout"):
        subset = [row for row in rows if row["split"] == split]
        valids = [row for row in subset if row["is_valid"]]
        traps = [row for row in subset if not row["is_valid"]]
        auroc_pos = compute_auroc([r["d_pos"] for r in valids], [r["d_pos"] for r in traps]) if valids and traps else float("nan")
        auroc_bal = compute_auroc([r["s_bal"] for r in valids], [r["s_bal"] for r in traps]) if valids and traps else float("nan")
        traps_caught = sum(row["s_bal"] < cutoff for row in traps)
        valids_kept = sum(row["s_bal"] >= cutoff for row in valids)
        pair_count = len(valids) * len(traps)
        print(f"[{split}] AUROC(pos)={auroc_pos:.3f} | AUROC(dual-bal)={auroc_bal:.3f} "
              f"({len(valids)}x{len(traps)}={pair_count} pairs) | "
              f"Traps caught @ {valids_kept}/{len(valids)} valid={traps_caught}/{len(traps)} "
              f"(cutoff={cutoff:+.3f})")
        print(f"  Mean Yes/No mass={sum(r['mass'] for r in subset) / max(1, len(subset)):.3f}")
        for row in sorted(subset, key=lambda value: value["s_bal"]):
            tag = "VALID" if row["is_valid"] else "TRAP "
            print(f"  {row['id']:<6} {tag} | d_pos={row['d_pos']:+6.2f} d_neg={row['d_neg']:+6.2f} "
                  f"s_bal={row['s_bal']:+6.2f} | {row['query'][:60]}", flush=True)

    _print_frozen_cascade_summary(stage1, rows, "s_bal", cutoff, "0.5B dual-polarity diagnostic")

    print("\n--- Direct Factual Recall on Stage-1-Passing Traps ---")
    missed_traps = [row for row in rows if not row["is_valid"]]
    for row in missed_traps:
        probe_q, expected = M2_L2_ATOMIC_FACTS.get(
            row["id"], (f"Is the factual premise in this question true: {row['query']}", "No")
        )
        prompt = _chat_prompt(engine.tokenizer, f"{probe_q}\nAnswer Yes or No.")
        diff, p_yes, mass = _stage2_yes_no_logits(engine.model, engine.tokenizer, prompt, yes_ids, no_ids)
        pred = "Yes" if diff >= 0 else "No"
        verdict = "KNOWS_FACT" if pred == expected else "KNOWLEDGE_GAP"
        print(f"  {row['id']:<6} pred={pred:<3} (expected={expected:<3}, diff={diff:+6.2f}, mass={mass:.3f}) "
              f"-> {verdict} | {probe_q}", flush=True)
    print(f"--m2-l2-diag runtime: {time.perf_counter() - started:.1f}s on {engine.device}.")


M2_L2_STAGE2_PROMPT_VERSION = "premise-audit-v1"


def _format_instruct_premise_prompt(tokenizer, query: str, negate: bool = False):
    """Build the canonical versioned Stage-2 prompt used by both 0.5B paths."""
    if not negate:
        user_text = (
            "Check whether the following question rests on a factually true premise.\n"
            f"Question: {query}\n"
            "Reply with exactly one word: Yes if the premise is factually valid, "
            "or No if it contains a false, fictional, or impossible premise."
        )
    else:
        user_text = (
            "Check whether the following question contains a false, fictional, or impossible premise.\n"
            f"Question: {query}\n"
            "Reply with exactly one word: Yes if it contains a false premise, "
            "or No if the premise is factually valid."
        )
    messages = [
        {"role": "system", "content": "You are a strict factual premise auditor. Answer only Yes or No."},
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{user_text}\nAnswer:"


def _fit_frozen_phase1_cutoff(p1_scored: list, score_key: str):
    """Fit a frozen threshold using Phase 1 passers only."""
    valid_scores = sorted(row[score_key] for row in p1_scored if row["is_valid"])
    trap_scores = sorted(row[score_key] for row in p1_scored if not row["is_valid"])
    if not valid_scores:
        raise RuntimeError("No valid Phase 1 Stage-1 passers; cannot calibrate Stage 2.")
    margin = min(valid_scores) - max(trap_scores) if trap_scores else float("inf")
    if trap_scores and margin > 0:
        return 0.5 * (min(valid_scores) + max(trap_scores)), "max-margin midpoint", margin
    cutoff_index = min(1, len(valid_scores) - 1)
    return valid_scores[cutoff_index], "rank cutoff (allow one valid false halt)", margin


def _print_frozen_cascade_summary(stage1_results: list, scored: list, score_key: str, cutoff: float, label: str):
    """Print end-to-end cascade results with one Phase 1-fitted cutoff."""
    by_id = {row["id"]: row for row in scored}
    print(f"\n--- End-to-End Cascade Summary: {label} (frozen cutoff >= {cutoff:+.3f}) ---", flush=True)
    print("Split    | S1 Pass | S2 In | Valid Kept | Traps Resolved | S2 Caught | Ans. Sel. Acc | S2 AUROC (valid x trap pairs)", flush=True)
    print("---------+---------+-------+------------+----------------+-----------+---------------+----------------------------", flush=True)
    for split in ("P1", "Holdout"):
        rows = [row for row in stage1_results if row["split"] == split]
        passers = [row for row in scored if row["split"] == split]
        valid_total = sum(bool(row["is_valid"]) for row in rows)
        trap_total = len(rows) - valid_total
        valid_kept = sum(
            bool(row["is_valid"]) and by_id[row["id"]][score_key] >= cutoff
            for row in rows if row["stage1_pass"]
        )
        s1_traps_resolved = sum(
            not row["is_valid"]
            and (
                not row["stage1_pass"]
                or explicitly_rejects_premise(row.get("answer", ""))
            )
            for row in rows
        )
        s2_traps_halted = sum(
            not row["is_valid"] and by_id[row["id"]][score_key] < cutoff
            for row in rows if row["stage1_pass"]
            and row["id"] in by_id
            and not explicitly_rejects_premise(row.get("answer", ""))
        )
        traps_resolved = s1_traps_resolved + s2_traps_halted
        submitted_rows = []
        for row in rows:
            if not row["stage1_pass"]:
                continue
            if not row["is_valid"] and explicitly_rejects_premise(row.get("answer", "")):
                # A safe rejection that passes Stage 1 is still a submitted, correct response.
                submitted_rows.append((row, True))
                continue
            scored_row = by_id.get(row["id"])
            if scored_row is None or scored_row[score_key] < cutoff:
                continue
            submitted_rows.append((row, bool(row.get("answer_correct")) if row["is_valid"] else False))
        submitted = len(submitted_rows)
        correct_submitted = sum(is_correct for _, is_correct in submitted_rows)
        selective_accuracy = correct_submitted / submitted if submitted else 0.0
        valid_scores = [row[score_key] for row in passers if row["is_valid"]]
        trap_scores = [row[score_key] for row in passers if not row["is_valid"]]
        auroc = compute_auroc(valid_scores, trap_scores) if valid_scores and trap_scores else float("nan")
        pair_count = len(valid_scores) * len(trap_scores)
        print(
            f"{split:<8} | {sum(row['stage1_pass'] for row in rows):>2}/{len(rows):<4} | "
            f"{len(passers):>3} | {valid_kept:>2}/{valid_total:<7} | "
            f"{traps_resolved:>2}/{trap_total:<13} | "
            f"{s2_traps_halted:>2}/{len(trap_scores):<6} | "
            f"{correct_submitted:>2}/{submitted:<4} ({selective_accuracy * 100:>5.1f}%) | "
            f"{auroc:>7.3f} ({len(valid_scores)}x{len(trap_scores)}={pair_count})",
            flush=True,
        )


@torch.inference_mode()
def run_m2_l2_frozen_dual_polarity(v_crit: float = -66.0, force_offline: bool = True):
    """Score Stage-1 passers with 0.5B dual polarity, refreshing stale caches once."""
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    cache_path = _m2_l2_stage1_cache_path(v_crit)
    stage1_results = _read_m2_l2_stage1_cache(v_crit)
    engine = None
    if stage1_results is None:
        print(
            f"Stage-1 cache missing or stale at {cache_path}; running the committed "
            f"Phase 1 + {len(M2_L2_HOLDOUT_QUERIES)}-query holdout once.",
            flush=True,
        )
        engine = HRCLLMEngine(MODEL_ID, force_offline=force_offline)
        stage1_results = collect_m2_l2_stage1(engine, v_crit)
    _print_m2_l2_stage1_trap_audit(stage1_results)
    passers = _m2_l2_stage2_candidates(stage1_results)
    print(
        f"\n=== FROZEN-THRESHOLD 0.5B DUAL-POLARITY CHECK ===\n"
        f"Stage-2 premise prompt: {M2_L2_STAGE2_PROMPT_VERSION}\n"
        f"Stage-1 outcomes: {cache_path}\n"
        f"Scoring {len(passers)}/{len(stage1_results)} passers; "
        f"Stage 1 {'was run once above' if engine is not None else 'reused from cache'}.",
        flush=True,
    )
    if engine is not None:
        tokenizer, model = engine.tokenizer, engine.model
    else:
        frozen_device, frozen_dtype = _resolve_device_dtype(ACTIVE_DEVICE, ACTIVE_DTYPE)
        tokenizer, model = load_hf_local_first(
            MODEL_ID, force_offline=force_offline, dtype=frozen_dtype
        )
        model = model.to(frozen_device)
    model.eval()
    yes_ids, no_ids = resolve_yes_no_token_ids(tokenizer)
    scored = []
    for index, row in enumerate(passers, start=1):
        start = time.perf_counter()
        pos = _format_instruct_premise_prompt(tokenizer, row["query"], negate=False)
        neg = _format_instruct_premise_prompt(tokenizer, row["query"], negate=True)
        d_pos, _, _ = _stage2_yes_no_logits(model, tokenizer, pos, yes_ids, no_ids)
        d_neg, _, _ = _stage2_yes_no_logits(model, tokenizer, neg, yes_ids, no_ids)
        s_bal = 0.5 * (d_pos - d_neg)
        scored_row = {**row, "d_pos": d_pos, "d_neg": d_neg, "s_bal": s_bal,
                      "s2_time": time.perf_counter() - start}
        scored.append(scored_row)
        tag = "VALID" if row["is_valid"] else "TRAP "
        print(
            f"  [{index:02d}/{len(passers)}] {row['split']:<7} {row['id']:<7} {tag} | "
            f"d_pos={d_pos:+6.2f} d_neg={d_neg:+6.2f} s_bal={s_bal:+6.2f} "
            f"({scored_row['s2_time']:.2f}s)",
            flush=True,
        )

    p1_scored = [row for row in scored if row["split"] == "P1"]
    cutoff, rule, margin = _fit_frozen_phase1_cutoff(p1_scored, "s_bal")
    print(f"Locked Phase 1 cutoff: {cutoff:+.3f} via {rule}; Phase 1 margin={margin:+.3f}.", flush=True)
    _print_frozen_cascade_summary(stage1_results, scored, "s_bal", cutoff, "0.5B dual-polarity")

    p1_valid_scores = sorted(row["s_bal"] for row in p1_scored if row["is_valid"])
    if len(p1_valid_scores) >= 2:
        conservative_cutoff = p1_valid_scores[1]
        if abs(conservative_cutoff - cutoff) > 1e-6:
            _print_frozen_cascade_summary(
                stage1_results, scored, "s_bal", conservative_cutoff,
                "0.5B dual-polarity (11/12 valid-retention reference)",
            )
    mean_latency = sum(row["s2_time"] for row in scored) / max(1, len(scored))
    print(f"Mean dual-polarity scoring latency: {mean_latency:.2f}s/passer.", flush=True)
    del model, tokenizer
    if engine is not None:
        del engine
    gc.collect()


@torch.inference_mode()
def run_m2_l2_asymmetric(
    stage2_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dual_polarity: bool = False,
    force_offline: bool = True,
    v_crit: float = -66.0,
    use_bf16: bool = True,
    stage1_model_id: str = MODEL_ID,
    stage1_dtype_name: str = None,
    stage1_cache_dir: str | Path = None,
    stage1_device_name: str = "cpu",
    stage2_device_name: str = None,
    stage2_dtype_name: str = None,
    stage2_cache_dir: str | Path = None,
):
    """Cache-only, resumable Stage-2 run; never loads the Stage-1 model."""
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    started = time.perf_counter()
    cache_path = _m2_l2_stage1_cache_path(
        v_crit,
        model_id=stage1_model_id,
        dtype_name=stage1_dtype_name,
        cache_dir=stage1_cache_dir,
        device_name=stage1_device_name,
    )
    stage1_results = _read_m2_l2_stage1_cache(
        v_crit,
        model_id=stage1_model_id,
        dtype_name=stage1_dtype_name,
        cache_dir=stage1_cache_dir,
        device_name=stage1_device_name,
    )
    if stage1_results is None:
        raise FileNotFoundError(
            f"Valid Stage-1 cache for {len(_m2_l2_phase1_stream()) + len(M2_L2_HOLDOUT_QUERIES)} "
            f"queries not found at {cache_path}; run --m2-l2-diag or --m2-l2-frozen first."
        )
    _print_m2_l2_stage1_trap_audit(stage1_results, label=f"{stage1_model_id} Stage 1")
    passers = _m2_l2_stage2_candidates(stage1_results)
    requested_dtype = torch.float32 if not use_bf16 else (
        ACTIVE_DTYPE if stage2_dtype_name is None else stage2_dtype_name
    )
    stage2_device, dtype = _resolve_device_dtype(
        ACTIVE_DEVICE if stage2_device_name is None else stage2_device_name,
        requested_dtype,
    )
    stage2_device_name = stage2_device.type
    stage2_dtype_name = _dtype_name(dtype)
    cache_dir = stage2_cache_dir or stage1_cache_dir or ".hrc_cache"
    stage2_cache_path, cache_metadata = _m2_l2_stage2_cache_path(
        stage1_results, stage2_model_id, stage2_dtype_name,
        stage2_device_name, dual_polarity, cache_dir,
    )
    cached_by_key = _read_m2_l2_stage2_cache(stage2_cache_path, cache_metadata)
    for row in passers:
        key = (row["split"], row["id"], row["query"])
        cached_row = cached_by_key.get(key)
        if cached_row is not None and bool(cached_row.get("is_valid")) != bool(row["is_valid"]):
            cached_by_key.pop(key, None)
    uncached = [
        row for row in passers
        if (row["split"], row["id"], row["query"]) not in cached_by_key
    ]
    mode = "dual-polarity (2 forwards/query)" if dual_polarity else "single-pass (1 forward/query)"
    print(
        f"\n=== M2 LAYER 2 ASYMMETRIC STAGE-2 ({stage2_model_id}) ===\n"
        f"Stage-2 premise prompt: {M2_L2_STAGE2_PROMPT_VERSION}\n"
        f"Stage-1 cache model: {stage1_model_id}; dtype={stage1_dtype_name or 'resolved'}; "
        f"device={stage1_device_name}\n"
        f"Stage-1 cache: {cache_path}\nStage-2 cache: {stage2_cache_path}\n"
        f"Config: device={stage2_device_name}, dtype={dtype}, "
        f"threads={min(4, os.cpu_count() or 4)}, mode={mode}, "
        f"passers={len(passers)}/{len(stage1_results)}, cached={len(passers) - len(uncached)}, "
        f"pending={len(uncached)}.",
        flush=True,
    )
    model2 = tokenizer2 = None
    if uncached:
        load_start = time.perf_counter()
        tokenizer2, loaded_model = load_hf_local_first(
            stage2_model_id,
            force_offline=force_offline,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model2 = loaded_model.to(stage2_device)
        model2.eval()
        print(f"Loaded Stage-2 model in {time.perf_counter() - load_start:.1f}s.", flush=True)
        yes_ids, no_ids = resolve_yes_no_token_ids(tokenizer2)
        for index, row in enumerate(uncached, start=1):
            started_score = time.perf_counter()
            positive = _format_instruct_premise_prompt(tokenizer2, row["query"], negate=False)
            d_pos, _, _ = _stage2_yes_no_logits(model2, tokenizer2, positive, yes_ids, no_ids)
            if dual_polarity:
                negative = _format_instruct_premise_prompt(tokenizer2, row["query"], negate=True)
                d_neg, _, _ = _stage2_yes_no_logits(model2, tokenizer2, negative, yes_ids, no_ids)
                s_bal = 0.5 * (d_pos - d_neg)
            else:
                d_neg = None
                s_bal = d_pos
            cache_row = {
                "id": row["id"], "split": row["split"], "query": row["query"],
                "is_valid": bool(row["is_valid"]), "d_pos": float(d_pos),
                "d_neg": None if d_neg is None else float(d_neg),
                "s_bal": float(s_bal),
                "latency_s": float(time.perf_counter() - started_score),
            }
            cached_by_key[(row["split"], row["id"], row["query"])] = cache_row
            ordered_cache_records = [
                cached_by_key[(candidate["split"], candidate["id"], candidate["query"])]
                for candidate in passers
                if (candidate["split"], candidate["id"], candidate["query"]) in cached_by_key
            ]
            _write_m2_l2_stage2_cache(stage2_cache_path, cache_metadata, ordered_cache_records)
            tag = "VALID" if row["is_valid"] else "TRAP "
            print(
                f"  [{index:02d}/{len(uncached)}] {row['split']:<7} {row['id']:<7} {tag} | "
                f"d_pos={d_pos:+.3f} d_neg={float('nan') if d_neg is None else d_neg:+.3f} "
                f"s_bal={s_bal:+.3f} ({cache_row['latency_s']:.2f}s); cache flushed",
                flush=True,
            )
        del model2, tokenizer2
        gc.collect()
    else:
        print("All Stage-2 candidates are cached; skipping model weight loading.", flush=True)

    scored = []
    for row in passers:
        cache_row = cached_by_key[(row["split"], row["id"], row["query"])]
        scored.append({
            **row,
            "d_pos": cache_row["d_pos"], "d_neg": cache_row["d_neg"],
            "s_bal": cache_row["s_bal"], "s2_score": cache_row["s_bal"],
            "s2_time": cache_row["latency_s"],
        })

    phase1_passers = [row for row in scored if row["split"] == "P1"]
    cutoff, rule, margin = _fit_frozen_phase1_cutoff(phase1_passers, "s2_score")
    mean_latency = sum(row["s2_time"] for row in scored) / max(1, len(scored))
    amortized_latency = mean_latency * len(passers) / max(1, len(stage1_results))
    print(
        f"Locked Phase 1 cutoff: {cutoff:+.3f} via {rule}; Phase 1 margin={margin:+.3f}.\n"
        f"Latency: mean={mean_latency:.2f}s/passer, amortized={amortized_latency:.2f}s/query; "
        f"Stage-1 pass rate={len(passers)}/{len(stage1_results)} ({100 * len(passers) / len(stage1_results):.1f}%).",
        flush=True,
    )
    _print_frozen_cascade_summary(stage1_results, scored, "s2_score", cutoff, f"{stage2_model_id} [{mode}]")
    print(f"--m2-l2-asym runtime: {time.perf_counter() - started:.1f}s.", flush=True)
    gc.collect()


def run_m2_l2(v_crit: float = -66.0):
    """Benchmark three single-pass premise vetoes behind the existing V_m gate."""
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    phase1_stream = _validate_m2_l2_holdout_disjoint()

    stage1_rows = []
    print("\nMilestone 2 Layer 2: Stage 1 V_m pass and holdout audit", flush=True)
    for index, (question, gold, bucket) in enumerate(phase1_stream, start=1):
        result = engine.infer_with_vm(question, v_crit=v_crit)
        stage1_rows.append({
            "id": f"P1_{index:02d}",
            "split": "P1",
            "label": "FalsePremise" if gold is None else "Valid",
            "question": question,
            "answer": result["answer"],
            "v_m": result["v_m"],
            "vm_pass": not result["halt"],
            "bucket": bucket,
        })
        print(
            f"  P1 {index:02d}/40 {bucket:<13} V_m={result['v_m']:6.1f} "
            f"{'PASS' if not result['halt'] else 'HALT'}: {question[:58]}",
            flush=True,
        )
    for probe in M2_L2_HOLDOUT_QUERIES:
        result = engine.infer_with_vm(probe["q"], v_crit=v_crit)
        stage1_rows.append({
            "id": probe["id"],
            "split": "Holdout",
            "label": probe["category"],
            "question": probe["q"],
            "answer": result["answer"],
            "v_m": result["v_m"],
            "vm_pass": not result["halt"],
            "bucket": probe["category"],
        })
        print(
            f"  Holdout {probe['id']} {probe['category']:<13} V_m={result['v_m']:6.1f} "
            f"{'PASS' if not result['halt'] else 'HALT'}: {probe['q']}",
            flush=True,
        )

    pass_rows = [row for row in stage1_rows if row["vm_pass"]]
    p1_pass = [row for row in pass_rows if row["split"] == "P1"]
    holdout_pass = [row for row in pass_rows if row["split"] == "Holdout"]
    print(
        f"Stage 1 eligible for Stage 2: Phase 1 {len(p1_pass)}/40; "
        f"holdout {len(holdout_pass)}/{len(M2_L2_HOLDOUT_QUERIES)}. "
        f"(Current engine passes V_m <= {v_crit:.1f} mV.)",
        flush=True,
    )

    yes_ids, no_ids = resolve_yes_no_token_ids(engine.tokenizer)
    candidates = ("2A", "2B", "2C")
    candidate_records = {candidate: [] for candidate in candidates}
    for row in pass_rows:
        is_valid = row["label"] != "FalsePremise"
        for candidate in candidates:
            score = score_stage2_single_pass(
                engine, candidate, row["question"], row["answer"], yes_ids, no_ids
            )
            candidate_records[candidate].append({
                **row,
                **score,
                "is_valid": is_valid,
            })

    thresholds = {}
    p1_evaluations = {}
    holdout_evaluations = {}
    candidate_means = {}
    for candidate in candidates:
        all_records = candidate_records[candidate]
        p1_records = [row for row in all_records if row["split"] == "P1"]
        holdout_records = [row for row in all_records if row["split"] == "Holdout"]
        threshold = fit_stage2_threshold(p1_records)
        thresholds[candidate] = threshold
        p1_evaluations[candidate] = evaluate_stage2_records(p1_records, threshold["tau_veto"])
        holdout_evaluations[candidate] = evaluate_stage2_records(
            holdout_records, threshold["tau_veto"]
        )
        candidate_means[candidate] = (
            sum(row["raw_mass"] for row in p1_records) / len(p1_records) if p1_records else 0.0
        )

    print("\nPer-query diagnostic (only Stage-1 V_m passers; s_valid higher means safer):")
    print(
        f"{'ID':<7} | {'Split':<7} | {'Label':<13} | {'V_m':>6} | {'Answer':<23} | "
        f"{'2A (s,m)':>15} | {'2B (s,m)':>15} | {'2C (s,m)':>15}"
    )
    print("-" * 120)
    for row in pass_rows:
        candidate_scores = {}
        for candidate in candidates:
            candidate_row = next(
                record for record in candidate_records[candidate] if record["id"] == row["id"]
            )
            candidate_scores[candidate] = (
                f"{candidate_row['s_valid']:.3f},{candidate_row['raw_mass']:.3f}"
            )
        print(
            f"{row['id']:<7} | {row['split']:<7} | {row['label']:<13} | {row['v_m']:6.1f} | "
            f"{row['answer'][:23]:<23} | {candidate_scores['2A']:>15} | "
            f"{candidate_scores['2B']:>15} | {candidate_scores['2C']:>15}",
            flush=True,
        )

    print("\nCandidate summary (threshold fitted on Phase 1 V_m passers only):")
    print(
        f"{'Candidate':<22} | {'P1 Margin':>9} | {'P1 AUROC':>8} | {'tau':>7} | "
        f"{'P1 Trap Halt':>12} | {'P1 Valid Kept':>14} | {'P1 SelAcc':>9} | "
        f"{'HO Trap Halt':>12} | {'HO Valid Kept':>14} | {'Mean ms':>8} | {'Gate':>5}"
    )
    print("-" * 153)
    diagnostic_flags = []
    for candidate, label in (("2A", "2A (P(True))"), ("2B", "2B (0-shot)"), ("2C", "2C (2-shot)")):
        threshold = thresholds[candidate]
        p1 = p1_evaluations[candidate]
        holdout = holdout_evaluations[candidate]
        p1_trap_text = f"{p1['traps_halted']}/{p1['traps_total']}"
        p1_valid_text = f"{p1['valid_kept']}/{p1['valid_total']}"
        hold_trap_text = f"{holdout['traps_halted']}/{holdout['traps_total']}"
        hold_valid_text = f"{holdout['valid_kept']}/{holdout['valid_total']}"
        tau_text = f"{threshold['tau_veto']:.3f}" if threshold["tau_veto"] is not None else "n/a"
        margin_text = f"{threshold['margin']:+.3f}" if math.isfinite(threshold["margin"]) else "n/a"
        auroc_text = f"{threshold['auroc']:.3f}" if math.isfinite(threshold["auroc"]) else "n/a"
        p1_traps_caught = (
            p1["traps_total"] >= 2 and p1["traps_halted"] == p1["traps_total"]
        )
        p1_valid_retained = p1["valid_total"] == 12 and p1["valid_kept"] >= 11
        holdout_catch_rate = (
            holdout["traps_halted"] / holdout["traps_total"]
            if holdout["traps_total"] else 0.0
        )
        passed = (
            threshold["tau_veto"] is not None
            and p1_traps_caught
            and p1_valid_retained
            and p1["selective_acc"] >= 0.92
            and holdout["valid_kept"] >= 5
            and holdout["traps_total"] > 0
            and holdout_catch_rate >= 0.50
        )
        mean_latency = (
            sum(row["latency_ms"] for row in candidate_records[candidate])
            / len(candidate_records[candidate])
            if candidate_records[candidate] else 0.0
        )
        print(
            f"{label:<22} | {margin_text:>9} | {auroc_text:>8} | {tau_text:>7} | "
            f"{p1_trap_text:>12} | {p1_valid_text:>14} | {p1['selective_acc'] * 100:8.1f}% | "
            f"{hold_trap_text:>12} | {hold_valid_text:>14} | {mean_latency:7.2f} | "
            f"{'PASS' if passed else 'FAIL':>5}",
            flush=True,
        )
        diagnostic_flags.append(
            threshold["tau_veto"] is None
            or threshold["margin"] <= 0
            or candidate_means[candidate] < 0.20
        )

    if all(diagnostic_flags):
        nonpositive = [
            candidate for candidate in candidates
            if thresholds[candidate]["tau_veto"] is None or thresholds[candidate]["margin"] <= 0
        ]
        low_mass = [candidate for candidate in candidates if candidate_means[candidate] < 0.20]
        missing_threshold = [
            candidate for candidate in candidates if thresholds[candidate]["tau_veto"] is None
        ]
        print(
            "SCALE_CEILING_OR_BIAS_DIAGNOSTIC: trigger met for all candidates. "
            f"Non-positive P1 margins: {','.join(nonpositive) or 'none'}; "
            f"mean Yes/No raw mass < 0.20: {','.join(low_mass) or 'none'}; "
            f"unfitted thresholds: {','.join(missing_threshold) or 'none'}.",
            flush=True,
        )
    print(
        f"Stage 2 P1 mean Yes/No raw mass: "
        + " | ".join(f"{candidate}={candidate_means[candidate]:.3f}" for candidate in candidates)
    )
    print(f"--m2-l2 runtime: {time.perf_counter() - started:.1f}s on {engine.device}.")


def run_track_c(v_crit: float = -66.0, phase1_only: bool = False):
    torch.manual_seed(42)
    t0 = time.perf_counter()
    engine = HRCLLMEngine(MODEL_ID)
    memory = Layer3ResonantMemory(tau_res=0.45)

    # --------------------------------------------------------------------------
    # PHASE 1: ZERO-SHOT HALLUCINATION-PENALTY STREAM (40 QUERIES)
    # --------------------------------------------------------------------------
    phase1_stream = (
        WORLD_KNOWN
        + OBSCURE_HARD
        + [(ep["q_orig"], ep["aliases"], "Private") for ep in PRIVATE_EPISODES]
        + [(probe["question"], probe["gold"], probe["bucket"]) for probe in PHASE1_ADVERSARIAL_PROBES]
    )

    print("\n" + "=" * 116)
    print(f"PHASE 1: ZERO-SHOT HALLUCINATION-PENALTY STREAM (40 Queries | V_crit = {v_crit:.1f} mV)")
    print("=" * 116)
    print(f"{'#':<2} | {'Bucket':<13} | {'Question (truncated)':<38} | {'Pred (truncated)':<18} | {'Phi':<4} | {'V_m':<6} | {'Corr':<5} | {'Action'}")
    print("-" * 116)

    ts = fs = th = fh = 0
    vm_corr, vm_wrong = [], []
    naive_score = hrc_score = 0
    world_q_vecs = []
    private_q_vecs = {}
    results = []

    for idx, (q, aliases, bucket) in enumerate(phase1_stream):
        res = engine.infer_with_vm(q, memory=None, v_crit=v_crit)
        correct = grade_phase1_answer(res["answer"], aliases)

        if bucket == "World":
            world_q_vecs.append(res["q_vec"])
        elif bucket == "Private":
            private_q_vecs[q] = res["q_vec"]

        naive_score += 1 if correct else -1
        if not res["halt"]:
            hrc_score += 1 if correct else -1

        if correct:
            vm_corr.append(res["v_m"])
            if res["halt"]:
                fh += 1
            else:
                ts += 1
        else:
            vm_wrong.append(res["v_m"])
            if res["halt"]:
                th += 1
            else:
                fs += 1

        action = "HALT" if res["halt"] else "SUBMIT"
        results.append({
            "bucket": bucket,
            "question": q,
            "answer": res["answer"],
            "correct": correct,
            "phi": res["phi_sem"],
            "v_m": res["v_m"],
            "action": action,
        })
        print(
            f"{idx:<2} | {bucket:<13} | {q[:38]:<38} | {res['answer'][:18]:<18} | "
            f"{res['phi_sem']:4.2f} | {res['v_m']:6.1f} | {str(correct):<5} | {action}",
            flush=True
        )

    auroc = compute_auroc(vm_wrong, vm_corr)
    submitted = ts + fs
    sel_acc = (ts / submitted * 100.0) if submitted > 0 else 0.0
    raw_acc = ((ts + fh) / len(phase1_stream) * 100.0)

    print("-" * 116)
    print(f"PHASE 1 SUMMARY:")
    print(f"  Raw Base Accuracy          : {raw_acc:.1f}% ({ts + fh}/{len(phase1_stream)})")
    print(f"  HRC Selective Accuracy     : {sel_acc:.1f}% ({ts}/{submitted} submitted)")
    print(f"  Gate Breakdown             : TrueSubmit={ts} | FalseSubmit={fs} | TrueHalt={th} | FalseHalt={fh}")
    if vm_corr and vm_wrong:
        print(f"  Mean V_m Separation        : Correct = {sum(vm_corr)/len(vm_corr):.1f} mV  vs  Wrong = {sum(vm_wrong)/len(vm_wrong):.1f} mV  (AUROC = {auroc:.3f})")
    print(f"  Hallucinations Prevented   : {th}/{len(vm_wrong)} ({th/max(1,len(vm_wrong))*100:.1f}%)")
    print(f"  Continual Net Score        : Naive Always-Submit = {naive_score:+d}  |  HRC V_m-Gated = {hrc_score:+d}  (Delta = {hrc_score - naive_score:+d})")

    adv_rows = [r for r in results if r["bucket"] in ("Misconception", "FalsePremise")]
    adv_wrong = [r for r in adv_rows if not r["correct"]]
    adv_false_submits = [r for r in adv_wrong if r["action"] == "SUBMIT"]
    high_phi_wrong = [r for r in adv_wrong if r["phi"] >= 0.70]
    high_phi_caught = [r for r in high_phi_wrong if r["action"] == "HALT"]
    wrong_rows = [r for r in results if not r["correct"]]
    correct_rows = [r for r in results if r["correct"]]
    min_wrong_vm = min((r["v_m"] for r in wrong_rows), default=None)
    max_corr_vm = max((r["v_m"] for r in correct_rows), default=None)
    fmt_vm = lambda value: f"{value:.1f} mV" if value is not None else "n/a"
    print(f"  Adversarial FalseSubmit   : {len(adv_false_submits)}/{len(adv_wrong)} wrong adversarial answers were submitted")
    print(f"  High-Phi Wrong Caught      : {len(high_phi_caught)}/{len(high_phi_wrong)} (Phi >= 0.70; HALT)")
    print(f"  Clean-Gap Audit            : closest Wrong V_m = {fmt_vm(min_wrong_vm)} | highest Correct V_m = {fmt_vm(max_corr_vm)}")

    if phase1_only:
        print("\nPHASE 1 ONLY: later consolidation and unlearning phases skipped.")
        return

    # --------------------------------------------------------------------------
    # PHASE 2: LAYER 3 CONTINUAL CONSOLIDATION + PARAPHRASE RECALL
    # --------------------------------------------------------------------------
    # Fit isotropy center on base world-knowledge query geometry
    memory.fit_isotropy_center(world_q_vecs)
    for ep in PRIVATE_EPISODES:
        memory.consolidate(ep["id"], private_q_vecs[ep["q_orig"]], ep["fact"])

    print("\n" + "=" * 116)
    print("PHASE 2: LAYER 3 CONTINUAL CONSOLIDATION (6 Paraphrased Private Queries + 4 Base Retention Probes)")
    print("=" * 116)
    print(f"{'ID':<5} | {'Type':<8} | {'Paraphrased / Base Query':<40} | {'Pred (truncated)':<18} | {'ResSim':<6} | {'V_m':<6} | {'Corr':<5} | {'Action'}")
    print("-" * 116)

    p2_priv_hits = 0
    for ep in PRIVATE_EPISODES:
        res = engine.infer_with_vm(ep["q_para"], memory=memory, v_crit=v_crit)
        correct = is_correct_answer(res["answer"], ep["aliases"]) and (not res["halt"])
        p2_priv_hits += int(correct)
        action = "HALT" if res["halt"] else "SUBMIT"
        print(
            f"{ep['id']:<5} | {'Private':<8} | {ep['q_para'][:40]:<40} | {res['answer'][:18]:<18} | "
            f"{res['res_sim']:6.3f} | {res['v_m']:6.1f} | {str(correct):<5} | {action}",
            flush=True
        )

    p2_base_hits = 0
    for i, (q, aliases, _) in enumerate(WORLD_KNOWN[:4]):
        res = engine.infer_with_vm(q, memory=memory, v_crit=v_crit)
        correct = is_correct_answer(res["answer"], aliases) and (not res["halt"])
        p2_base_hits += int(correct)
        action = "HALT" if res["halt"] else "SUBMIT"
        print(
            f"B{i+1:<4} | {'BaseRet':<8} | {q[:40]:<40} | {res['answer'][:18]:<18} | "
            f"{res['res_sim']:6.3f} | {res['v_m']:6.1f} | {str(correct):<5} | {action}",
            flush=True
        )

    print("-" * 116)
    print(f"PHASE 2 SUMMARY: New Paraphrase Recall = {p2_priv_hits}/6 ({p2_priv_hits/6*100:.0f}%)  |  Base Retention = {p2_base_hits}/4 ({p2_base_hits/4*100:.0f}%)")

    # --------------------------------------------------------------------------
    # PHASE 3: O(1) EXACT MACHINE UNLEARNING (PURGE E1, E2, E3; RETAIN E4, E5, E6)
    # --------------------------------------------------------------------------
    t_unlearn = time.perf_counter()
    purged_ids = [ep["id"] for ep in PRIVATE_EPISODES if ep["forget"]]
    for pid in purged_ids:
        memory.exact_unlearn(pid)
    unlearn_ms = (time.perf_counter() - t_unlearn) * 1000.0

    print("\n" + "=" * 116)
    print(f"PHASE 3: O(1) EXACT MACHINE UNLEARNING — Purged {purged_ids} in {unlearn_ms:.4f} ms")
    print("=" * 116)
    print(f"{'ID':<5} | {'Target':<8} | {'Paraphrased Query':<40} | {'Pred (truncated)':<18} | {'ResSim':<6} | {'V_m':<6} | {'Leak/Hit':<8} | {'Action'}")
    print("-" * 116)

    forget_leaks = 0
    forget_halts = 0
    retain_hits = 0

    for ep in PRIVATE_EPISODES:
        res = engine.infer_with_vm(ep["q_para"], memory=memory, v_crit=v_crit)
        matched = is_correct_answer(res["answer"], ep["aliases"])
        action = "HALT" if res["halt"] else "SUBMIT"
        if ep["forget"]:
            if matched:
                forget_leaks += 1
            if res["halt"]:
                forget_halts += 1
            status = "LEAK!" if matched else "CLEAN"
            tag = "FORGET"
        else:
            ok = matched and (not res["halt"])
            retain_hits += int(ok)
            status = "RETAINED" if ok else "LOST!"
            tag = "RETAIN"

        print(
            f"{ep['id']:<5} | {tag:<8} | {ep['q_para'][:40]:<40} | {res['answer'][:18]:<18} | "
            f"{res['res_sim']:6.3f} | {res['v_m']:6.1f} | {status:<8} | {action}",
            flush=True
        )

    total_s = time.perf_counter() - t0
    print("=" * 116)
    print(f"HRC TRACK C FINAL VERDICT ({MODEL_ID} on {engine.device} in {total_s:.1f}s)")
    print(f"  Phase 1 | Selective Acc: {sel_acc:.1f}% vs Base {raw_acc:.1f}% | AUROC: {auroc:.3f} | Net Score: {naive_score:+d} -> {hrc_score:+d}")
    print(f"  Phase 2 | Continual Learning (Paraphrased): {p2_priv_hits}/6  |  Base Catastrophic Forgetting: {4 - p2_base_hits}/4")
    print(f"  Phase 3 | Exact Unlearning ({unlearn_ms:.4f} ms): Forgotten Leakage = {forget_leaks}/3 (Safe Halts = {forget_halts}/3) | Retained = {retain_hits}/3")
    print("=" * 116)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the HRC Track C benchmark.")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--phase1-only", action="store_true", help="Run Phase 1 and its adversarial probes only.")
    run_mode.add_argument("--dense-scale", action="store_true", help="Run Layer 3 dense-cluster scaling and N=100 probes.")
    run_mode.add_argument("--unlearn-audit", action="store_true", help="Run unseen paraphrase and indirect-query unlearning probes.")
    run_mode.add_argument("--baselines", action="store_true", help="Run Milestone 1 Step 4 baseline comparisons.")
    run_mode.add_argument("--m2-l3", action="store_true", help="Run Milestone 2 Layer 3 injection-floor and sparse/dense hybrid audit.")
    run_mode.add_argument("--m2-l3-slice-only", action="store_true", help="Re-run only the 16-query N=100 Tier 2B Layer 3 slice.")
    run_mode.add_argument("--m2-l2", action="store_true", help="Run the Milestone 2 Layer 2 single-pass premise-veto diagnostic.")
    run_mode.add_argument("--m2-l2-diag", action="store_true", help="Run the 0.5B dual-polarity and direct-recall diagnostic.")
    run_mode.add_argument("--m2-l2-frozen", action="store_true", help="Run frozen-threshold 0.5B dual-polarity scoring from the Stage-1 cache.")
    run_mode.add_argument("--m2-l2-components", action="store_true", help="Audit cached valid answers and rerun only Rome, Pride and Prejudice, and oxygen Stage-1 diagnostics.")
    run_mode.add_argument("--m2-l2-stage1-compare", action="store_true", help="Run the locked Stage-1 split with another model using a model-specific cache.")
    run_mode.add_argument("--m2-l2-asym", action="store_true", help="Run a decoupled Stage-2 premise verifier on Stage-1 passers.")
    parser.add_argument("--offline", action="store_true", help="Force local-only Hugging Face model loading.")
    parser.add_argument("--stage1-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID for --m2-l2-stage1-compare.")
    parser.add_argument("--stage1-fp32", action="store_true", help="Override the selected --dtype with float32 for Stage 1.")
    parser.add_argument("--stage2-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID for --m2-l2-asym.")
    parser.add_argument("--stage1-cache-model", default=MODEL_ID, help="Stage-1 model ID whose cached split --m2-l2-asym should read.")
    parser.add_argument(
        "--stage1-cache-dtype",
        choices=("default", "float16", "bfloat16", "float32"),
        default="default",
        help="Optional dtype tag for the Stage-1 cache selected by --m2-l2-asym.",
    )
    parser.add_argument("--stage1-cache-dir", default=None, help="Optional directory containing the selected Stage-1 cache.")
    parser.add_argument("--stage1-cache-device", choices=("auto", "cpu", "cuda"), default="auto", help="Device tag of the Stage-1 cache selected by --m2-l2-asym.")
    parser.add_argument("--stage2-dual-polarity", action="store_true", help="Score positive and negative premise prompts in Stage 2.")
    parser.add_argument("--stage2-fp32", action="store_true", help="Override the selected --dtype with float32 for Stage 2.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Model device; auto selects CUDA when available.")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto", help="Model precision; auto uses float16 on pre-Ampere CUDA, bfloat16 on Ampere+ CUDA and CPU.")
    parser.add_argument("--stage2-cache-dir", default=None, help="Stage-2 cache directory; defaults to --stage1-cache-dir or .hrc_cache.")
    parser.add_argument(
        "--part",
        choices=("4a", "4b", "all"),
        default="all",
        help="Select a Step 4 baseline section when using --baselines (default: all).",
    )
    args = parser.parse_args()
    resolved_device, resolved_dtype = _resolve_device_dtype(args.device, args.dtype)
    ACTIVE_DEVICE = resolved_device.type
    ACTIVE_DTYPE = _dtype_name(resolved_dtype)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.dense_scale:
        run_dense_scale()
    elif args.unlearn_audit:
        run_unlearn_audit()
    elif args.baselines:
        run_baselines(part=args.part)
    elif args.m2_l3:
        run_m2_l3()
    elif args.m2_l3_slice_only:
        run_m2_l3_tier2b()
    elif args.m2_l2:
        run_m2_l2()
    elif args.m2_l2_diag:
        run_m2_l2_diagnostic()
    elif args.m2_l2_frozen:
        run_m2_l2_frozen_dual_polarity(force_offline=args.offline)
    elif args.m2_l2_components:
        run_m2_l2_component_diagnostic(force_offline=args.offline)
    elif args.m2_l2_stage1_compare:
        stage1_dtype = torch.float32 if args.stage1_fp32 else resolved_dtype
        run_m2_l2_stage1_model_compare(
            model_id=args.stage1_model,
            force_offline=args.offline,
            dtype=stage1_dtype,
            device_name=resolved_device.type,
            cache_dir=args.stage1_cache_dir,
        )
    elif args.m2_l2_asym:
        run_m2_l2_asymmetric(
            stage2_model_id=args.stage2_model,
            dual_polarity=args.stage2_dual_polarity,
            force_offline=args.offline,
            use_bf16=not args.stage2_fp32,
            stage1_model_id=args.stage1_cache_model,
            stage1_dtype_name=None if args.stage1_cache_dtype == "default" else args.stage1_cache_dtype,
            stage1_cache_dir=args.stage1_cache_dir,
            stage1_device_name=(resolved_device.type if args.stage1_cache_device == "auto" else args.stage1_cache_device),
            stage2_device_name=resolved_device.type,
            stage2_dtype_name=("float32" if args.stage2_fp32 else _dtype_name(resolved_dtype)),
            stage2_cache_dir=args.stage2_cache_dir,
        )
    else:
        run_track_c(phase1_only=args.phase1_only)
