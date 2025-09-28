import transformers as tr
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-3B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

amateur_model = tr.AutoModelForCausalLM.from_pretrained(
    amateur_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
expert_model = tr.AutoModelForCausalLM.from_pretrained(
    expert_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

amateur_model.eval()
expert_model.eval()

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    return ""