import torch

def generate_action_response(command, keywords, tokenizer, model):
    prompt = f"""너는 로봇이야. 사용자의 지시를 이해하고 적절한 응답을 생성해.
다음 지시에 따라 행동을 시작한다고 생각하고 자연스럽게 대답해.

지시: "{command}"
핵심 키워드: {keywords}
응답:"""

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return result.replace(prompt, "").strip()
