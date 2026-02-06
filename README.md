<p align="center">
  <img
    src="docs/aigco_logo.png"
    alt="aigco Logo"
    width="200"
  />
</p>

<span align="center">

## Quick Start

### Installation

> aigco requires **Python 3.12** or higher

#### From PyPI

```bash
pip install aigco[flash_attn]
```

or

with uv:

```bash
uv pip install aigco[flash_attn]

or add to deependencies
"flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl"
```

#### From Source

```bash
# pull the source code from Github
git clone --depth 1 https://github.com/TorrentBrave/aigco.git

# Install the package in editable mode
cd aigco

pip install -e .
# or with uv
# uv pip install -e .
```

#### As a submodule in Src

```bash
mkdir src && git -C src clone https://github.com/TorrentBrave/aigco.git

git submodule add --force https://github.com/TorrentBrave/aigco.git src/aigco

uv add --editable ./src/aigco/ <!-- will update aigco.egg.info -->
cd src/aigco
uv lock
uv sync
```

## Example

### Inference Qwen3-0.6B like vllm

```python
import time
import aigco
from random import randint, seed
from aigco.inference import LLM, SamplingParams
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-0.6B"

logger = aigco.logger(name="qwen3_benchmark")


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    try:
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        logger.info(f"📍 Finded model path: {model_path}")
    except Exception as e:
        logger.error(f"❌ Can't fine model in cache: {e}")
        return

    logger.info("Starting init LLM Engine...")
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]

    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    logger.info("🚀 Warmuping...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10))

    logger.info(f"🔥 Start test {num_seqs} random sequence...")
    start_time = time.time()

    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)

    total_time = time.time() - start_time

    total_gen_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_gen_tokens / total_time

    result_msg = (
        f"\n{'=' * 30}\n"
        f"✅ Benchmark Result:\n"
        f"Total Generated Tokens: {total_gen_tokens} tok\n"
        f"Total Time: {total_time:.2f} s\n"
        f"Throughput: {throughput:.2f} tok/s\n"
        f"{'=' * 30}"
    )

    logger.info(result_msg)


if __name__ == "__main__":
    main()
```

```md
==============================
✅ Benchmark Result:
Total Generated Tokens: 133966 tok
Total Time: 101.90 s
Throughput: 1314.70 tok/s
==============================
```