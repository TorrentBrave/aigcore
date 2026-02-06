import aigco
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()
REPO_ID = "Qwen/Qwen3-0.6B"

logger = aigco.logger(name="qwen3_inference")


def main():
    try:
        model_path = snapshot_download(repo_id=REPO_ID, local_files_only=True)
        logger.info(f"📍 Find model path: {model_path}")
    except Exception as e:
        logger.error(f"❌ Can't find model in cache {REPO_ID}, Reason: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Starting init LLM Engine...")
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

    logger.info(f"Generating reponse, The number of samples: {len(prompts)}...")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        log_message = f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}"
        logger.info(log_message)

    logger.info("Finished Inference Task")


if __name__ == "__main__":
    main()
