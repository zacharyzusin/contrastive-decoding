import transformers as tr
import torch
import torch.nn.functional as F

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(expert_path)

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

def contrastive_generation(amateur, expert, prompt, max_tokens, alpha=1.0, plausibility_threshold=0.1, temperature=1.0, do_sample=True) -> str:    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(amateur.parameters()).device
    input_ids = input_ids.to(device)
    
    original_length = input_ids.size(1)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            amateur_logits = amateur(input_ids).logits[:, -1, :]
            expert_logits = expert(input_ids).logits[:, -1, :]
            
            cd_logits = expert_logits - alpha * amateur_logits
            
            expert_probs = F.softmax(expert_logits, dim=-1)
            max_expert_prob = expert_probs.max(dim=-1, keepdim=True)[0]
            plausibility_mask = expert_probs >= (plausibility_threshold * max_expert_prob)
            
            cd_logits = cd_logits.masked_fill(~plausibility_mask, float('-inf'))
            
            if torch.all(torch.isinf(cd_logits)):
                cd_logits = expert_logits
            
            if do_sample and temperature > 0:
                probs = F.softmax(cd_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = cd_logits.argmax(dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_tokens = input_ids[0, original_length:].tolist()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    print("Starting contrastive generation...")
    
    result = contrastive_generation(
        amateur=amateur_model,
        expert=expert_model, 
        prompt=prompt,
        max_tokens=100,
        alpha=1.0,
        plausibility_threshold=0.1,
        temperature=1.0,
        do_sample=True
    )
    
    print("="*50)
    print("CONTRASTIVE DECODING RESULT:")
    print("="*50)
    print(result)
    print("="*50)