import time
import aigco
from random import randint, seed
from aigco.inference import LLM, SamplingParams
from huggingface_hub import snapshot_download

# 模型在 HF 上的 ID
REPO_ID = "Qwen/Qwen3-0.6B"

# 初始化 logger
logger = aigco.logger(name="qwen3_benchmark")


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    # --- 1. 对齐路径获取逻辑 ---
    try:
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        logger.info(f"📍 找到模型路径: {model_path}")
    except Exception as e:
        logger.error(f"❌ 无法找到模型缓存: {e}")
        return

    # --- 2. 初始化 LLM ---
    logger.info("正在加载模型并初始化引擎...")
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)

    # --- 3. 准备随机数据 ---
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

    # --- 4. 热身 (Warmup) ---
    logger.info("🚀 正在热身 (Warmup)...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10))

    # --- 5. 正式测试 (Benchmarking) ---
    logger.info(f"🔥 开始测试 {num_seqs} 条随机序列...")
    start_time = time.time()

    # 传递 prompt_token_ids 列表和对应的 sampling_params 列表
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)

    total_time = time.time() - start_time

    # --- 6. 计算吞吐量 ---
    total_gen_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_gen_tokens / total_time

    # 格式化输出结果
    result_msg = (
        f"\n{'=' * 30}\n"
        f"✅ Benchmark Result:\n"
        f"Total Generated Tokens: {total_gen_tokens} tok\n"
        f"Total Time: {total_time:.2f} s\n"
        f"Throughput: {throughput:.2f} tok/s\n"
        f"{'=' * 30}"
    )

    # 最终结果记录到日志中
    logger.info(result_msg)


if __name__ == "__main__":
    main()
