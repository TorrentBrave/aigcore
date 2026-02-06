import aigco
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

# 模型在 HF 上的 ID
REPO_ID = "Qwen/Qwen3-0.6B"

# 确保 logger 配置了文件输出
# 如果 aigco.logger 支持，可以直接在这里指定 filename
logger = aigco.logger(name="qwen3_inference")


def main():
    # 自动获取缓存中的真实绝对路径
    try:
        # local_files_only=True 确保它只从本地找，不会去联网下载
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        logger.info(f"📍 找到模型路径: {model_path}")
    except Exception as e:
        logger.error(f"❌ 无法在缓存中找到模型 {REPO_ID}，原因: {e}")
        return

    # 使用自动获取的路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 初始化 LLM
    logger.info("开始初始化 LLM 引擎...")
    llm = aigco.inference.LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = aigco.inference.SamplingParams(temperature=0.6, max_tokens=256)
    prompts_text = ["introduce yourself", "list all prime numbers within 100"]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts_text
    ]

    # 执行生成
    logger.info(f"正在生成响应，样本数量: {len(prompts)}...")
    outputs = llm.generate(prompts, sampling_params)

    # 遍历并记录结果
    for prompt, output in zip(prompts, outputs):
        log_message = f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}"

        # 同时在控制台打印和记入日志文件
        # print(log_message)
        logger.info(log_message)

    logger.info("推理任务完成。")


if __name__ == "__main__":
    main()
