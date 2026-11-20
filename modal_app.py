"""Modal entry points. Example: modal run modal_app.py::eval_bamboogle --method direct --trials 3"""
import json

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1")  # vLLM 0.11 breaks on transformers 5.x
    .add_local_dir("data", "/root/data")
    .add_local_python_source("src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # keeps the 15 GB model download between runs
app = modal.App("agentflow-bamboogle", image=image)


@app.function(gpu="H100", volumes={"/root/.cache/huggingface": hf_cache}, timeout=4 * 3600)
def run_eval(method, trials, limit, lora):
    from src.eval import evaluate

    return evaluate(method, trials, limit, lora)


@app.local_entrypoint()
def eval_bamboogle(method: str = "direct", trials: int = 3, limit: int = 0, lora: str = ""):
    summary, records = run_eval.remote(method, trials, limit or None, lora or None)
    print(json.dumps(summary, indent=2))
    name = f"results/{method}{'_lora' if lora else ''}{f'_n{limit}' if limit else ''}.json"
    with open(name, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, records=records), f, indent=2, ensure_ascii=False)
    print(f"saved {name}")
