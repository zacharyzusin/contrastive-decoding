import transformers as tr
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-3B-Instruct"

amateur_tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
expert_tokenizer = tr.AutoTokenizer.from_pretrained(expert_path)
tokenizer = amateur_tokenizer

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

def get_next_token_logits(model: tr.PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Get logits for the next token from the model."""
    with torch.no_grad():
        outputs = model(input_ids)
        return outputs.logits[:, -1, :]

def contrastive_generation(amateur, expert, prompt, max_tokens, alpha=1.0, plausibility_threshold=0.1, temperature=1.0) -> str:
    """ Implement contrastive decoding algorithm."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if amateur_tokenizer.vocab != expert_tokenizer.vocab:
        expert_input_ids = expert_tokenizer.encode(prompt, return_tensors="pt")
    else:
        expert_input_ids = input_ids.clone()
    
    device = next(amateur.parameters()).device
    input_ids = input_ids.to(device)
    expert_input_ids = expert_input_ids.to(device)
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        amateur_logits = get_next_token_logits(amateur, input_ids)
        expert_logits = get_next_token_logits(expert, expert_input_ids)
        
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        
        contrastive_scores = expert_log_probs - alpha * amateur_log_probs
        
        expert_probs = F.softmax(expert_logits / temperature, dim=-1)
        plausible_mask = expert_probs > plausibility_threshold
        
        contrastive_scores = torch.where(
            plausible_mask,
            contrastive_scores,
            torch.full_like(contrastive_scores, -float('inf'))
        )
        
        contrastive_probs = F.softmax(contrastive_scores / temperature, dim=-1)
        
        next_token_id = torch.multinomial(contrastive_probs, num_samples=1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
        generated_tokens.append(next_token_id.item())
        
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        if amateur_tokenizer.vocab != expert_tokenizer.vocab:
            expert_token = next_token_id
            expert_input_ids = torch.cat([expert_input_ids, expert_token], dim=1)
        else:
            expert_input_ids = torch.cat([expert_input_ids, next_token_id], dim=1)
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    print("Starting contrastive generation...")
    print(f"Prompt: {prompt[:100]}...")
    
    result = contrastive_generation(
        amateur=amateur_model,
        expert=expert_model, 
        prompt=prompt,
        max_tokens=100,
        alpha=1.0,
        plausibility_threshold=0.1,
        temperature=1.0
    )
    
    print("\n" + "="*50)
    print("CONTRASTIVE DECODING RESULT:")
    print("="*50)
    print(result)
    print("="*50)