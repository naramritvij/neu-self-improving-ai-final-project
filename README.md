# neu-self-improving-ai-final-project

INFO 7375 Final Project - AgentFlow and Flow-GRPO replication on Bamboogle using LoRA - By Ritvij Naram and Ghouse Mohiuddin Mohammed

Paper: *In-the-Flow Agentic System Optimization for Effective Planning and Tool Use* (AgentFlow, ICLR 2026).

## Results (Bamboogle, 125 questions, mean ± std over 3 trials)

| Method | Paper | Ours |
|---|---|---|
| Qwen-2.5-7B-Instruct | 12.0 | 14.9 ± 0.8 |
| AgentFlow | 58.4 | in progress |
| AgentFlow w/ Flow-GRPO (LoRA) | 69.6 | in progress |

Answers are judged with the paper's prompt (§E.3) by Qwen2.5-7B-Instruct instead of GPT-4o. See `ROADMAP.md` for design choices and caveats.

## Run

```bash
pip install -r requirements.txt
python -m modal setup
python -m modal run modal_app.py::eval_bamboogle --method direct --trials 3
```
