# autoresearch — Knowledge Graph Adversarial Training

> **Historical provenance only.** This prompt records the autonomous procedure used during the original experiments. It is not a supported publication runner: several Stage 1 files it names were removed after provenance review, and its host-specific/destructive keep-or-discard instructions must not be executed on a current checkout. Use `README.md` for supported artifact commands.

> Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
> Last updated: 2026-04-09T23:00+08:00

This is an experiment to have the LLM do its own research — autonomously and indefinitely.

## Architecture

- **Controller (you, Claude Code)**: Mac mini — outer loop, edits code, searches papers, manages wiki
- **Training Engine**: NVIDIA DGX Spark (GB10, 128GB) at `jayc@192.168.10.105`
- **Knowledge Base**: `wiki/` — ingested papers and concepts
- **LLM Backend**: Ollama `qwen3:32b` on Spark (knowledge distillation)

## Setup

1. **Run tag**: propose a tag (e.g. `apr9`). Branch `autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b autoresearch/<tag>`
3. **Read in-scope files**:
   - `program.md` — this file
   - `models/transformer.py` — shared backbone (read-only)
   - `models/encoder.py` — KGEncoder (editable)
   - `models/decoder.py` — KGDecoder + RealismCritic (editable)
   - `train_adversarial.py` — primary file you modify
   - `prepare.py` — fixed data prep (read-only)
4. **Read the wiki**: `wiki/index.md` and papers in `wiki/raw/`
5. **Verify SSH**: `ssh -o ConnectTimeout=5 jayc@192.168.10.105 "echo OK"`
6. **Init results.tsv** with header row if needed
7. **Confirm and go**

## What you CAN / CANNOT do

**CAN**: Modify `train_adversarial.py`, `models/encoder.py`, `models/decoder.py`
**CANNOT**: Modify `prepare.py`, `models/transformer.py`, install new packages

**Goal: get the lowest L_rec.** Simplicity criterion: simpler is better at equal performance.

## Deploy and run

```bash
bash scripts/deploy.sh && bash scripts/run_train.sh && bash scripts/fetch_results.sh
```

Results: `grep "L_rec" results/run.log | tail -5`

## Logging

Log to `results.tsv` (tab-separated, untracked by git):
```
commit	L_rec_final	L_critic_final	status	description
```

## The experiment loop

LOOP FOREVER (with strategic breakpoints):

1. **Pre-flight**:
   - Time after 07:00 + no `reports/daily/morning_YYYY-MM-DD.md`? → **Morning Review** (pause)
   - Total experiments % 10 == 0? → **Checkpoint** (no pause, `python scripts/gen_report.py checkpoint`)
2. Look at git state
3. Tune code with an experimental idea
4. git commit
5. Deploy and run (see above)
6. Read results: `grep "L_rec" results/run.log | tail -5`
7. If empty → crash. `tail -n 50 results/run.log` to debug.
8. Record in results.tsv
9. **Evaluate**:
   - L_rec improved → **keep** commit
   - L_rec worse → **discard**, `git reset --hard HEAD~1`
   - L_rec crosses 0.5 / 0.1 / 0.01 → **Milestone** (mode-aware, see below)
     - Always: `python scripts/gen_report.py milestone <value>` and `git tag milestone-lrec-<value>`
     - If `$AUTORESEARCH_MODE == interactive` (default, daytime): **PAUSE** for human review
     - If `$AUTORESEARCH_MODE == overnight`: **DO NOT PAUSE**. Follow this ordered procedure:
       1. `/research-lookup` with a query specific to the current bottleneck (REQUIRED per plan §8). Ingest any useful paper to `wiki/raw/`.
       2. `/scientific-brainstorming` to list 3 candidate next directions.
       3. **In Stage 2 with the adversarial loop active (stage2c/d onwards), "mid-radical" variants are permitted** — reward shaping, LoRA rank, KL constraint, curriculum changes. Still not allowed: whole new dataset, new loss family, new architecture class. Those remain the human's call at Morning Review.
       4. **In Stage 1 or pre-adversarial Stage 2**, stay conservative: only hyperparam sweeps within the current research family (margin / encoder / KBGAN / data tweaks).
       5. Pick the strongest ROI candidate and resume LOOP. The human reviews at Morning Review (07:00).
10. Stuck **2+** experiments → **Knowledge Pipeline** (see below). *Lowered from 3+ per STAGE2_MIGRATION_PLAN §8: Stage 1 triggered KP only once in 92 experiments — the old threshold was too lax. 2+ means we actually use the literature arsenal.*
11. Knowledge Pipeline triggered 2+ times without improvement → **Stagnation** (pause)

## Knowledge Pipeline — Scientific Skills Integration

**This is the key extension beyond Karpathy's original design.** When stuck, you have a powerful arsenal of scientific skills. USE THEM.

### When to trigger
- **2+** consecutive experiments without L_rec improvement (lowered from 3+; see plan §8)
- After a Milestone breakpoint (to find next research direction)
- When you run out of ideas
- **Overnight mode: Milestone AND Stagnation breakpoints MUST run `/research-lookup` before picking a next direction.** The brainstorming call alone is no longer sufficient. Rationale: Stage 1 ran 92 experiments and ingested only 3 papers — brainstorming without fresh literature regresses to the same priors.

### Step 1: Search literature (USE SKILLS, NOT just WebSearch)

**Primary — use the `/research-lookup` skill:**
```
/research-lookup knowledge graph embedding <your specific problem> 2024 2025
```
This invokes Perplexity sonar-pro academic search. Results auto-save to `sources/`.

**Secondary — use `/paper-lookup` for targeted database search:**
```
/paper-lookup arxiv semantic_scholar "knowledge graph contrastive learning negative sampling"
```
Searches 10+ academic databases (arXiv, Semantic Scholar, PubMed, etc.)

**Tertiary — for systematic reviews, use `/literature-review`:**
```
/literature-review "adversarial training stabilization techniques for knowledge graph embeddings"
```

### Step 2: Ingest papers into wiki
```bash
python scripts/ingest_paper.py <paper_path_or_url>
```
Then run knowledge distillation:
```bash
python scripts/distill_knowledge.py
```

### Step 3: Read wiki insights and apply
- Read `wiki/raw/`, `wiki/concepts/`, `wiki/entities/`
- Extract actionable architecture changes
- Modify `train_adversarial.py` based on findings
- Continue the experiment loop

### Available Scientific Skills Reference

**Research & Discovery (找論文)**

| Skill | Command | Use when |
|-------|---------|----------|
| `research-lookup` | `/research-lookup <query>` | **首選** — 學術搜尋 (Perplexity sonar-pro) |
| `paper-lookup` | `/paper-lookup <database> <query>` | 針對特定資料庫 (arXiv, Semantic Scholar 等 10+) |
| `literature-review` | `/literature-review <topic>` | 系統性文獻回顧（比 research-lookup 更深入） |
| `bgpt-paper-search` | `/bgpt-paper-search <query>` | 從論文中提取實驗數據與方法論 |

**Thinking & Analysis (想方向)**

| Skill | Command | Use when |
|-------|---------|----------|
| `scientific-brainstorming` | `/scientific-brainstorming` | **連續 discard 沒靈感時** — 生成新實驗假設與方向 |
| `scientific-critical-thinking` | `/scientific-critical-thinking` | 評估一篇論文的技術是否適用於我們的 GAN+KG 架構 |

**Writing & Visualization (寫報告)**

| Skill | Command | Use when |
|-------|---------|----------|
| `scientific-writing` | `/scientific-writing` | **Milestone 報告** — 結構化研究摘要 (IMRaD 格式) |
| `scientific-visualization` | `/scientific-visualization` | **Checkpoint 報告** — 將 results.tsv 的 L_rec 趨勢視覺化 |
| `scientific-schematics` | `/scientific-schematics` | 生成 Encoder/Decoder/Critic 架構圖 |
| `citation-management` | `/citation-management` | 驗證 DOI、格式化引用 |
| `database-lookup` | `/database-lookup` | 交叉比對 78+ 科學資料庫 |

**IMPORTANT**: Always save search results to `sources/` folder. Check `sources/` before making new queries to avoid duplicate API calls.

## Breakpoints + Skill Triggers

Each breakpoint type has specific skills that SHOULD be invoked:

| Type | Trigger | Pause? | Report | Skills to invoke |
|------|---------|--------|--------|-----------------|
| **Milestone** | L_rec < 0.5 / 0.1 / 0.01 | **Mode-aware**: Yes if `interactive`, No if `overnight` | `reports/milestones/` | `/scientific-writing` for structured report; `overnight` mode also runs `/scientific-brainstorming` to pick a conservative next direction |
| **Morning** | After 07:00, no today's report | Yes | `reports/daily/` | `/scientific-visualization` for L_rec trend chart |
| **Stagnation** | Knowledge Pipeline 2+ times, no improvement | Yes | `reports/milestones/` | `/scientific-brainstorming` first, then `/research-lookup` to validate ideas |
| **Checkpoint** | Every 10 experiments | No | `reports/checkpoints/` | `/scientific-visualization` for progress chart |

### Skill usage flow by situation

**Normal iteration (experiments progressing):**
- No skills needed, just keep experimenting

**3+ consecutive discards (losing momentum):**
1. `/scientific-brainstorming` — generate 5+ new experimental hypotheses
2. Pick the most promising one, try it

**Knowledge Pipeline triggered (plateau detected):**
1. `/research-lookup` — find relevant papers
2. `/scientific-critical-thinking` — evaluate if techniques apply to our architecture
3. Ingest papers → distill → apply insights

**Milestone reached (L_rec breakthrough or Triple F1 jump):**
1. `/scientific-writing` — write structured milestone report
2. `/scientific-visualization` — generate trajectory plot
3. **`/research-lookup` — REQUIRED** (plan §8). Find papers for next research direction. Ingest promising ones to `wiki/raw/`.
4. **Mode-aware continuation** (check `$AUTORESEARCH_MODE`):
   - `interactive` (default): PAUSE for human review
   - `overnight`: DO NOT pause. Run `/scientific-brainstorming` to list 3 candidate next directions. **In Stage 2 adversarial phase**, mid-radical variants (reward shape, LoRA rank, KL constraint, curriculum) are permitted. **In Stage 1 or pre-adversarial Stage 2**, stay within the current research family only. Whole-paradigm pivots are reserved for the human at Morning Review.

**Morning Review (daily 07:00 breakpoint):**
The daily report in `reports/daily/morning_YYYY-MM-DD.md` **MUST** include (plan §8):
- A "Literature ingested overnight" section listing papers searched and added to `wiki/raw/`, with key insights per paper.
- If 0 papers were searched during the night, the report must explain why (e.g. "no Knowledge Pipeline triggered; 12 experiments all improved").
- The usual trajectory plot + experiment summary.

**Stagnation (deep stuck):**
1. `/scientific-brainstorming` — radical ideation session
2. `/literature-review` — systematic review of the subfield
3. Write stagnation report with proposed pivots

## Autonomous behavior

Between breakpoints, you are fully autonomous. Do NOT pause to ask the human. Do NOT ask "should I keep going?". The human might be asleep. Think harder — use the scientific skills above, read wiki, try radical changes. The loop runs until a breakpoint fires or the human interrupts.

## Operating modes

Read the `AUTORESEARCH_MODE` environment variable at session start (default: `interactive`).

- **`interactive`** (daytime, human in the loop): Milestone breakpoint pauses so the human can review and steer. This is the default and matches Karpathy's original design.
- **`overnight`** (set by `scripts/run_overnight.sh`): No human will respond until 07:00. Milestone is **soft** — write the report, tag the commit, then **required** `/research-lookup` → `/scientific-brainstorming` → pick next direction → resume the loop. The only hard stop overnight is the Morning Review breakpoint at 07:00. Per plan §8:
  - **Stage 1 / pre-adversarial Stage 2**: conservative pivots only (within current research family).
  - **Stage 2 adversarial (stage2c/d onwards)**: mid-radical variants permitted (reward shape, LoRA rank, KL, curriculum). Whole-paradigm pivots remain human-only.
  - Every Morning Review report must contain a literature-ingested section (0 papers requires an explanation).
