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
