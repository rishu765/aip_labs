# Lab 3: Retrieval Sweeps & Architectural Optimization

**Author:** AIP Engineering Team  
**Evaluation Set:** `rag_golden.jsonl` ($n=42$ evaluated; $Q36, Q38, Q39$ excluded)  
**Corpus:** 30 insurance policy and claims documents

---

## Executive Summary & Final Recommended Configurations

Through empirical sweeps across chunking strategies, retrieval algorithms, reranking stages, and vector indexing backends, we established optimal retrieval configurations for two distinct production workloads.

```
+----------------------------------------------------------------------------------------------------+
|                                    RECOMMENDED ARCHITECTURES                                       |
+----------------------------------------------------------------------------------------------------+
|  Workload A: Interactive Agent / Search Box                                                        |
|  Pipeline:   MarkdownChunker (size=400, heading_path=True) -> DenseRetriever (Exact BLAS, k=5)    |
|  Filter:     Metadata pre-filter (status == 'current')                                             |
|  Metrics:    nDCG@10: 0.8527 | Hit@1: 0.7857 | Recall@5: 0.9028 | MRR: 0.8800                      |
|  SLA / Cost: p95 Latency: 1.71 ms | Cost: $0.00 / 1k queries                                       |
+----------------------------------------------------------------------------------------------------+
|  Workload B: Offline / Overnight Batch Compliance Audit                                            |
|  Pipeline:   MarkdownChunker (size=400, heading_path=True) -> DenseRetriever (k=30)                |
|              -> LLMReranker (tier=SMALL, k=30->5)                                                  |
|  Filter:     Metadata pre-filter (status == 'current')                                             |
|  Metrics:    nDCG@5: 0.8600 | Hit@1: 0.8095 | Recall@5: 0.9286 | MRR: 0.8912                       |
|  SLA / Cost: p95 Latency: 28,000 ms | Cost: $1.50 / 1k queries                                     |
+----------------------------------------------------------------------------------------------------+
```

### Dataset & Evaluation Methodology Note ($n=42$)

The evaluation set contains 45 questions. Questions **Q36, Q38, and Q39** are unanswerable questions that have zero relevant documents in the golden set (`relevant_docs: []`). Because calculating recall ($\frac{|\text{retrieved} \cap \text{relevant}|}{|\text{relevant}|}$) and nDCG requires non-zero relevant targets, these metrics are mathematically undefined ($\frac{0}{0}$) for those three queries. They are excluded from retrieval ranking sweeps, yielding $n=42$ valid evaluation questions. The remaining two unanswerable questions (Q37, Q40) retain partial document references and are evaluated in the `unanswerable` kind. Refusal behaviors are deferred to Lab 4.

---

## Part A: Chunking Strategy & Size Curve

Chunking determines the semantic granularity of representation. We swept strategy, chunk size, and contextual breadcrumbs.

### A1: Strategy Comparison ($size=800$)

| Strategy                 | Hit Rate@1 | Hit Rate@5 |  Recall@5  |    MRR     |  nDCG@10   | Chunks | Build Latency |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :----: | :-----------: |
| `fixed-800`              |   0.7381   |   0.9524   |   0.8373   |   0.8387   |   0.7952   |   83   |    24.1 ms    |
| `sliding-800` (Baseline) |   0.7857   |   0.9286   |   0.8452   |   0.8451   |   0.8053   |   91   |    25.8 ms    |
| `recursive-800`          |   0.7619   |   0.9524   |   0.8750   |   0.8611   |   0.8251   |   98   |    28.3 ms    |
| **`markdown-800`**       | **0.7619** | **0.9762** | **0.8988** | **0.8720** | **0.8458** |  164   |    44.9 ms    |

_Finding:_ Structure-aware markdown chunking outperformed arbitrary boundary splitters, delivering $+0.0405$ nDCG@10 over baseline by preserving document hierarchy and table semantics.

### A2: Chunk Size Curve & The Dilution Effect

Holding the winning `markdown` strategy constant, we evaluated sizes 1600, 800, and 400 characters:

| Config             | Hit Rate@1 | Hit Rate@5 |  Recall@5  |    MRR     |  nDCG@10   | Chunks |
| :----------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
| `markdown-1600`    |   0.7143   |   0.9524   |   0.8750   |   0.8262   |   0.8075   |  150   |
| `markdown-800`     |   0.7619   |   0.9762   |   0.8988   |   0.8720   |   0.8458   |  164   |
| **`markdown-400`** | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** |  235   |

```
nDCG@10 Dilution Curve:
  0.86 |                * (markdown-400: 0.8527)
  0.84 |         * (markdown-800: 0.8458)
  0.82 |
  0.80 |  * (markdown-1600: 0.8075)
       +------------------------------------------------
         1600 chars         800 chars          400 chars
```

**The Dilution Explanation:** Dense embedding models compute a single fixed-dimensional pooled vector (768 dimensions) per chunk. In a 1600-character chunk, specific factual assertions (e.g., waiting period numbers, condition clauses) are averaged over hundreds of tokens of surrounding background prose. This "dilutes" the cosine similarity score against tight, factoid queries. Smaller chunks (400 characters) isolate individual policy clauses into focused semantic units, increasing peak similarity at retrieval time without sacrificing document coverage.

### A3: Context Injection via Heading-Path Prefixes

Testing `markdown-800` with vs. without the `[heading > path]` prefix:

| Variant     | Hit Rate@1 | Hit Rate@5 | Recall@5 |    MRR     |  nDCG@10   | $\Delta$ Hit@1 | $\Delta$ nDCG@10 |
| :---------- | :--------: | :--------: | :------: | :--------: | :--------: | :------------: | :--------------: |
| With Prefix | **0.7619** |   0.9762   |  0.8988  | **0.8720** | **0.8458** |  **+14.29%**   |   **+0.0543**    |
| No Prefix   |   0.6190   |   1.0000   |  0.9048  |   0.7837   |   0.7915   |       —        |        —         |

_Mechanism:_ While chunks without prefixes still land in the top 5 (Hit@5 reaches 1.0), their rank-1 precision collapses from 76.2% to 61.9%. Isolated paragraphs lose situational context (e.g., whether an exclusion applies to "Pre-Existing Conditions" or "Maternity Benefits"). Injecting breadcrumbs anchors the chunk's scope, enabling the retriever to place the exact section at Rank 1.

### A4: Diagnostic Case Study: Q08 Chunking Failure

- **Question:** _"Can I claim for IVF treatment?"_ (Target: `exclusions`)
- **Under `fixed-800`:** `Hit Rate@5 = 0.0` (Failed entirely). Top retrieved chunks were irrelevant generic clauses from unrelated policy documents. Fixed slicing bisected the exclusion list mid-sentence across arbitrary offsets, separating "IVF and assisted reproduction" from its governing exclusion header.
- **Under `markdown-400`:** **Rank 1** (Score: `0.6890`, Success). The heading `[exclusions > Specific Treatments]` was prepended directly to the isolated bullet point, allowing the embedding to match the user's intent with high confidence.

---

## Part B: Retrieval Mechanics & Hybrid Fusion Analysis

We evaluated Dense retrieval (`gemini-embedding-001`), Lexical retrieval (`BM25Okapi`), and Hybrid Reciprocal Rank Fusion (`RRF-60`) on the winning `markdown-400` index.

### B1: Aggregate Retrieval Comparison

| Retriever       | Hit Rate@1 | Hit Rate@5 |  Recall@5  |    MRR     |  nDCG@10   | p95 Latency |
| :-------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| **Dense**       | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** |   3.46 ms   |
| BM25            |   0.4762   |   0.9286   |   0.7956   |   0.6698   |   0.6978   |   2.92 ms   |
| Hybrid (RRF-60) |   0.6667   |   0.9762   |   0.8631   |   0.7976   |   0.7949   |   6.32 ms   |

### B2: Per-Kind Breakdown (MRR)

Aggregates mask categorical failures. Notice that `hit_rate@5` is saturated (~0.93–0.98), hiding critical quality differences. Using **MRR** reveals the true ranking dynamics:

| Question Kind   | Count ($n$) | Dense MRR  |  BM25 MRR  | Hybrid MRR | Dominant Retriever  |
| :-------------- | :---------: | :--------: | :--------: | :--------: | :-----------------: |
| `single_hop`    |     18      |   0.9074   |   0.8519   | **0.9444** |   Hybrid / Dense    |
| `multi_hop`     |     10      | **1.0000** |   0.6500   |   0.8167   | **Dense (+0.1833)** |
| `paraphrase`    |      5      | **0.8000** |   0.4867   |   0.6500   | **Dense (+0.1500)** |
| `aggregation`   |      4      | **0.8750** |   0.3750   |   0.5833   | **Dense (+0.2917)** |
| `trap_archived` |      3      | **0.8333** |   0.5111   |   0.6667   | **Dense (+0.1666)** |
| `unanswerable`  |      2      |   0.3125   | **0.4167** |   0.3750   |        BM25         |

### Case Studies: Q44 vs. Q41 Mechanism

1. **Q44 (Exact Identifier):** _"What is the policy code for the Silver Plan?"_ (`AUR-HI-SIL-2026`)
   - `Dense MRR = 0.5000` (Rank 2) | `BM25 MRR = 1.0000` (Rank 1) | `Hybrid MRR = 1.0000` (Rank 1)
   - _Mechanism:_ Subword tokenizers fragment alphanumeric identifiers into arbitrary sub-tokens, weakening semantic density. BM25 indexes the exact token string with a high inverse document frequency (IDF), guaranteeing rank 1. Fusion successfully rescues the dense lag.
2. **Q41 (Lexical Mismatch / Paraphrase):** _"If I skip paying on time, how long before I lose everything?"_
   - `Dense MRR = 1.0000` (Rank 1) | `BM25 MRR = 0.0000` (Unranked in top 5) | `Hybrid MRR = 0.2500` (Rank 4)
   - _Mechanism:_ The query shares zero lexical overlap with the corpus text ("grace period of thirty days"). Dense retrieval matches the conceptual semantics effortlessly. However, BM25 assigns zero scores to relevant chunks; when fused via RRF, the non-retrieval from BM25 penalizes the correct chunk, demoting it from Rank 1 to Rank 4.

### B3 & B4: RRF Sensitivity & Fusion Weight Sweeps

- **RRF Constant $k \in \{10, 30, 60, 100\}$:**
  - $k=10$: nDCG@10 = 0.8156, Hit@1 = 0.6667
  - $k=30$: nDCG@10 = 0.7949, Hit@1 = 0.6667
  - $k=60$: nDCG@10 = 0.7949, Hit@1 = 0.6667
  - $k=100$: nDCG@10 = 0.7901, Hit@1 = 0.6667
  - _Observation:_ RRF is remarkably insensitive to $k$ within reasonable ranges ($[30, 100]$), demonstrating algorithmic robustness.
- **Fusion Weighting (Dense : BM25):**
  - $1:1$ ratio: nDCG@10 = 0.7949, Hit@1 = 0.6667
  - $2:1$ ratio: nDCG@10 = 0.8068, Hit@1 = 0.6905
  - $3:1$ ratio: nDCG@10 = 0.8136, Hit@1 = 0.6905
  - $1:2$ ratio: nDCG@10 = 0.7926, Hit@1 = 0.7143
  - Pure Dense: **nDCG@10 = 0.8527, Hit@1 = 0.7857**

### B5: Why Hybrid Lost on This Corpus

Industry rule-of-thumb states that "Hybrid is the single best upgrade for RAG." Here, **Hybrid is significantly worse than pure Dense** ($\Delta \text{nDCG} = -0.0578$).  
Dense outperforms BM25 across 85% of queries where they differ. Because BM25 fails completely on colloquial phrasings, conceptual questions, and cross-section aggregations, fusing BM25 injects noise that drags down high-confidence dense rankings far more often than it rescues rare alphanumeric codes. _A technique that is right on average can be wrong on your data._

---

## Part C: Two-Stage Reranking

We retrieved the top $k=30$ candidates using Dense retrieval and reranked down to top $k=5$ using a local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) and an LLM Reranker (`gemini-2.5-flash-lite`).

### C1–C3: Reranking Decision Matrix

| Configuration                           |   nDCG@5   | Hit Rate@1 |  Recall@5  | p95 Latency | Cost / 1k Queries |
| :-------------------------------------- | :--------: | :--------: | :--------: | :---------: | :---------------: |
| **Dense alone ($k=5$)**                 |   0.8313   |   0.7857   |   0.9028   | **1.71 ms** |     **$0.00**     |
| Cross-Encoder ($k=30 \rightarrow 5$)    |   0.8174   |   0.7619   |   0.8810   | 1,459.2 ms  |       $0.00       |
| **LLM Reranker ($k=30 \rightarrow 5$)** | **0.8600** | **0.8095** | **0.9286** | 28,000.0 ms |       $1.50       |

### Deployment Architectural Recommendations

1. **Interactive Agent-Facing Search Box: Deploy Dense Alone ($k=5$)**
   - _Rationale:_ User-facing interactive applications operate under strict SLA constraints ($<200\text{ ms}$ total round-trip). Dense retrieval provides an exceptional 0.8313 nDCG@5 and 78.6% Hit@1 in just **1.71 ms** at zero cost. Cross-Encoder reranking introduces +1.45 seconds of latency while _degrading_ accuracy by $-0.0139$ nDCG. The LLM reranker adds 28 seconds of p95 latency, which completely destroys interactive user experience.
2. **Overnight Batch Compliance Audit: Deploy LLM Reranker ($k=30 \rightarrow 5$)**
   - _Rationale:_ In batch auditing, queries run asynchronously offline where latency SLAs do not exist, and precision is paramount to avoid human audit overhead. The LLM reranker achieves the highest quality in the lab (0.8600 nDCG@5, 80.95% Hit@1, 92.86% Recall@5). At $1.50 per 1,000 queries, processing 10,000 policy checks costs only $15.00—a negligible expense for compliance verification.

### C4: Failure Diagnosis on Degraded Query Q01 (Failure Mode 5)

- **Question Q01:** _"How many days do I have to submit a reimbursement claim after discharge?"_
- **Dense alone:** Rank 1 (`MRR = 1.0000`, correct doc `claims-timelines` retrieved first).
- **Cross-Encoder reranking:** Rank 2 (`MRR = 0.5000`, demoted).
- **Mechanism (Domain Mismatch):** `ms-marco-MiniLM-L-6-v2` is pretrained on generic MS-MARCO search queries. It heavily weights surface-level lexical question-answer alignment. It selected a chunk containing the words "claim submission" from a general procedural overview rather than the strict tabular timeline clause. Dense embedding captured the regulatory semantics correctly; the generic cross-encoder overrode it with superficial keyword matching.

---

## Part D: Index Scaling & Metadata Filtering

### D1: Exact Search vs. Approximate Nearest Neighbors (Chroma HNSW)

On the production corpus ($N=235$ chunks), Chroma HNSW produced **identical retrieval quality** to exact NumPy BLAS dot-product:

- nDCG@10: `0.8527` (0.0000 gap)
- Hit Rate@1: `0.7857` (0.0000 gap)
- Recall@5: `0.9028` (0.0000 gap)
- MRR: `0.8800` (0.0000 gap)

### D2: Scaling Benchmark & Crossover Point

| Index Size              | Exact BLAS (NumPy) Latency | Chroma HNSW Latency |     Speedup Factor      |       Winner        |
| :---------------------- | :------------------------: | :-----------------: | :---------------------: | :-----------------: |
| **235 chunks** (Corpus) |        **0.80 ms**         |       6.60 ms       |   $8.25\times$ slower   |   **Exact BLAS**    |
| **4,000 chunks**        |          1.85 ms           |       1.42 ms       |   $1.30\times$ faster   | **Crossover (~4k)** |
| **40,000 chunks**       |          18.20 ms          |     **2.10 ms**     | **$8.67\times$ faster** |      **HNSW**       |

```
Query Latency Scaling:
  ms
  20 |                                        * (Exact: 18.2 ms)
  15 |
  10 |
   5 |  * (HNSW: 6.6 ms)      (Crossover ~4k)
   0 |  . (Exact: 0.8 ms)---*---------------* (HNSW: 2.1 ms)
     +--------------------------------------------------
         235 chunks           4,000 chunks     40,000 chunks
```

**Mechanism:** At small scale ($N < 1,000$), computing exact cosine similarity via matrix multiplication (`np.dot`) executes within optimized CPU L1/L2 cache lines with vector SIMD instructions. In contrast, HNSW incurs fixed Python-to-C++ binding overhead and non-contiguous memory pointer dereferencing during graph traversal. Only when linear scan complexity $\mathcal{O}(N \cdot d)$ exceeds graph traversal complexity $\mathcal{O}(\log N \cdot M \cdot d)$ does HNSW become advantageous. Adopting ANN prematurely adds latency, memory, and complexity without benefit.

### D3: The Trap Questions & Ingest Metadata Filtering

- **Target Trap Queries:** Q29, Q30, and Q31 involve policy claims deadlines. The corpus contains two documents: `claims-timelines` (current active terms) and `claims-timelines-2024-ARCHIVED` (superseded terms).
- **Before Filter:** `Hit Rate@1 = 0.6667`. The retriever ranked the archived document at Rank 1 on Q30 because the archived text had slightly higher lexical overlap with the query prompt.
- **After Filter (`where={"status": "current"}`):** `Hit Rate@1 = 1.0000` (**+33.33% gain**).

> **Architectural Takeaway:** _The best retrieval fix is often not retrieval._  
> When retrieval fails, engineering teams frequently rush to deploy more complex embedding models, exotic fusion schemes, or expensive rerankers. However, semantic models cannot infer temporal validity or business lifecycle status from prose alone. High-leverage data hygiene, document lifecycle metadata, and deterministic pre-filtering solve systemic ranking errors that no semantic retriever can fix.

---

## Methodological Reflection & Limitations

### 1. The Single Biggest Surprise: Hybrid Retrieval Degraded Quality

The most counter-intuitive result of this investigation was that Reciprocal Rank Fusion (BM25 + Dense) degraded retrieval performance across nearly every metric and subcategory compared to Dense alone. Standard literature treats hybrid retrieval as an unmitigated upgrade. Here, our embedding model (`gemini-embedding-001`) demonstrated sufficient semantic grasp of domain terminology that BM25 functioned primarily as an injection of lexical noise. This proves that architectural patterns must be empirically validated on the target domain rather than accepted on faith.

### 2. Limitations of Greedy One-Axis-At-A-Time Sweeps

Our experimental pipeline followed a greedy sequential optimization:
$$\text{Strategy} \longrightarrow \text{Size} \longrightarrow \text{Retriever} \longrightarrow \text{Reranker}$$
While practical, greedy coordinate ascent can miss non-linear interactions across axes:

- **Chunk Size $\times$ Retrieval Algorithm Interaction:** A 1600-character chunk might severely dilute Dense embeddings but perform exceptionally well with BM25, which benefits from rich term frequencies across longer passages.
- **Reranker $\times$ Candidate Pool Coupling:** Cross-encoders are trained to rerank heterogeneous candidates; their utility might be significantly higher over a candidate pool generated by a larger chunk size or higher retrieval cutoff ($k=100$).
  A full factorial grid search or Bayesian hyperparameter optimization would be required to verify that the combination of optimal individual choices constitutes the global optimum.
