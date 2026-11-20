"""One shared vLLM engine for every module, tool and the judge (paper §4.1: all Qwen2.5-7B-Instruct)."""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "Qwen/Qwen2.5-7B-Instruct"
_engine = None
_lora_ids = {}  # vLLM needs a unique integer id per adapter path


def generate(chats, temperature=0.0, max_tokens=2048, lora=None):
    """Batch-generate one reply per chat. A chat is a message list or a plain user string.

    `lora` is a path to a saved planner adapter; None means the frozen base model.
    """
    global _engine
    if _engine is None:
        _engine = LLM(MODEL, enable_lora=True, max_lora_rank=16, gpu_memory_utilization=0.85)
    chats = [[{"role": "user", "content": c}] if isinstance(c, str) else c for c in chats]
    request = None
    if lora:
        _lora_ids.setdefault(lora, len(_lora_ids) + 1)
        request = LoRARequest(lora, _lora_ids[lora], lora)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = _engine.chat(chats, params, lora_request=request, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]
