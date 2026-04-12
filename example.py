import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    prompts = [
        ("introduce yourself", SamplingParams(temperature=0.6, max_tokens=256)),
        ("list all prime numbers within 100", SamplingParams(temperature=0.6, max_tokens=256)),
        ("一个月后我要参加托业考试，帮我制定复习计划", SamplingParams(temperature=0.6, max_tokens=1024)),
    ]
    prompts = [
        (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            sp,
        )
        for prompt, sp in prompts
    ]

    for prompt, sp in prompts:
        outputs = llm.generate([prompt], sp)
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
