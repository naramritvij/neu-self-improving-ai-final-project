"""LLM-as-judge with the paper's analyze-then-judge prompt (§E.3). Used for eval accuracy and the training reward."""
import re

from src.llm import generate

JUDGE_PROMPT = """Task: Determine if the Model Response is equivalent to the Ground Truth.

Instructions:
1. Extract: Isolate the final answer from the Model Response, ignoring all reasoning steps. Look specifically for content within \\boxed{{...}} or the concluding statement.
2. Normalize & Compare: Assess equivalence after normalization:
3. Mathematical Answers: Must be mathematically identical (e.g., 1/2 is equivalent to 0.5).
4. Numerical/Textual Answers: Ignore formatting (commas, spaces), case sensitivity, and extraneous units/currency (e.g., "1,000" == "1000", "Paris" == "PARIS").
5. Multiple Choice Questions (MCQ): The answer must match either the correct option's content (e.g., "Paris") or its identifier (e.g., "A" or "1st").
6. Verdict: Return "True" only if the normalized answers are semantically or mathematically equivalent.

Inputs:
Question: {question}
Model Response: {response}
Ground Truth: {truth}

Output Format: Present your response in the following structured format. Do not include any extra text or explanations.
<analysis>: Brief analysis of the comparison.
<true_false>: "True" or "False"."""


def judge(questions, responses, truths):
    """Return one bool per (question, response, truth) triple."""
    prompts = [JUDGE_PROMPT.format(question=q, response=r, truth=t) for q, r, t in zip(questions, responses, truths)]
    verdicts = generate(prompts, temperature=0.0, max_tokens=512)
    return [bool(re.search(r"<true_false>\W*true", v, re.IGNORECASE)) for v in verdicts]
