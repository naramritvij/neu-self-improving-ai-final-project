# Part 1 roadmap (due Nov 23)

Goal: reproduce three Bamboogle rows of AgentFlow Table 1, using LoRA for training.
Graded on **simplicity**: every line must be explainable in the walkthrough.

| Row | Paper | Ours |
|---|---|---|
| Qwen-2.5-7B-Instruct (direct answer, no tools) | 12.0 | – |
| AgentFlow (frozen Qwen planner) | 58.4 | – |
| AgentFlow w/ Flow-GRPO (LoRA planner) | 69.6 | – |

## Design (paper setting → our simplification)

- **Models:** Qwen2.5-7B-Instruct for every module and tool, served with vLLM on Modal. Only the planner gets a LoRA adapter.
- **Loop:** planner → executor → verifier, repeated until the verifier says STOP or the turn limit is hit (10 at eval, 3 in training), then the solution generator answers. Prompts are copied from paper §E.1.
- **Batching:** all questions advance through the loop together, one batched vLLM call per module per turn. That's what makes 125 questions × 10 turns finish in minutes.
- **Tools:** Base Generator, Python Coder, Wikipedia Search and Google Search (§E.2). We drop Web Search and the OpenAI-embedding RAG to keep the code small.
- **Judge:** the paper's analyze-then-judge prompt (§E.3). The paper uses GPT-4o; we use Qwen2.5-7B unless we get an OpenAI key. *(decision pending)*
- **Flow-GRPO (§3.2):** for each question, run G=8 rollouts. Give every planner turn the trajectory's 0/1 reward, normalize advantages within the group (Eq. 7), and do one policy-gradient step on the LoRA weights plus a KL penalty (β=0.001) to the base model. With one update per batch the PPO ratio is 1, so the clip term drops out. Learning rate, temperature and max turns follow §C.1.
- **Training data:** Search-R1's NQ+HotpotQA training split (paper mixes in DeepMath, but Part 1 only evaluates Bamboogle). Never Bamboogle.
- **Eval:** 125 questions, 3 trials, mean ± std, like the paper.

## Files and owners

| File | What | Owner |
|---|---|---|
| `scripts/xlsx_to_csv.py`, `data/` | Bamboogle CSV from the course workbook | Ritvij ✅ |
| `src/llm.py` | `generate(chats, temperature, max_tokens, lora=None) -> list[str]` on vLLM | Ritvij |
| `src/judge.py` | Paper's judge prompt → True/False per answer | Ritvij |
| `src/eval.py`, `modal_app.py` | Run a method on Bamboogle on a Modal GPU, save `results/*.json` | Ritvij |
| `src/tools.py` | Base Generator, Python Coder, Wikipedia Search, Google Search | **Ghouse** |
| `src/agentflow.py` | `run_agentflow(questions, max_turns, temperature, lora=None)` → answers + per-turn planner (prompt, output) | **Ghouse** |
| `scripts/prepare_train_data.py` | Download and subsample Search-R1 train data → `data/train.jsonl` | **Ghouse** |
| `src/flow_grpo.py` | Rollouts with `run_agentflow`, rewards, LoRA update, save adapter | Ritvij |
| `README.md` results table, demo script | Final numbers and walkthrough order | Both |

**Interface contract:** `agentflow.py` only calls `llm.generate(...)`, and `flow_grpo.py` only calls `run_agentflow(...)`. As long as these signatures hold, we can work in parallel without touching each other's files.

## Timeline

- **Thu Nov 20:** baseline row done (Ritvij). Tools and agent loop working on 5 questions (Ghouse).
- **Fri Nov 21:** AgentFlow row, 3 trials (Ghouse runs `eval.py`). Flow-GRPO trainer written (Ritvij). Training data ready (Ghouse).
- **Sat Nov 22:** training run on Modal, then evaluate the trained planner.
- **Sun Nov 23:** results table, README, and a demo rehearsal where each of us can explain every file.

## Workflow

- Work on a branch, keep commits small, and pull before you push.
- Run everything through `modal run modal_app.py::<function>`, so nobody needs a local GPU.
