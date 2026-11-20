"""Bamboogle evaluation: run a method on all 125 questions, judge, average over trials (paper §C.2)."""
import csv
import statistics
from pathlib import Path

from src.judge import judge
from src.llm import generate

DIRECT_PROMPT = "Answer the following question. Give only the final answer, no explanation.\n\nQuestion: {question}"


def load_bamboogle():
    with open(Path(__file__).parent.parent / "data" / "bamboogle.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def answer(method, questions, lora=None):
    if method == "direct":  # Table 1 row "Qwen-2.5-7B-Instruct": no tools, one call
        return generate([DIRECT_PROMPT.format(question=q) for q in questions], temperature=0.7, max_tokens=256)
    from src.agentflow import run_agentflow  # rows "AgentFlow" and "w/ Flow-GRPO" (lora = trained planner)

    return [r["answer"] for r in run_agentflow(questions, max_turns=10, temperature=0.7, lora=lora)]


def evaluate(method, trials=3, limit=None, lora=None):
    rows = load_bamboogle()[:limit]
    questions = [r["question"] for r in rows]
    truths = [r["answer"] + (f" (also accepted: {r['answer_2']})" if r["answer_2"] else "") for r in rows]
    accuracies, records = [], []
    for trial in range(trials):
        responses = answer(method, questions, lora)
        correct = judge(questions, responses, truths)
        accuracies.append(100 * sum(correct) / len(correct))
        records += [dict(trial=trial, question=q, truth=t, response=r, correct=c)
                    for q, t, r, c in zip(questions, truths, responses, correct)]
    summary = dict(method=method, lora=lora, n=len(questions), accuracies=accuracies,
                   mean=statistics.mean(accuracies), std=statistics.pstdev(accuracies))
    return summary, records
