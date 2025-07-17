import torch

def detect_mode(text, tokenizer, model):
    prompt = f"""다음 문장을 '지시' 또는 '대화'로 분류하세요.
- '지시': 로봇이 실제로 무언가를 수행해야 하는 명령 (예: "빨간 공을 정리해", "노란색 상자에 넣어줘")
- '대화': 정보 요청, 인사, 일상적인 대화 (예: "오늘 날씨 어때?", "안녕", "점심 뭐 먹지?")

문장: "{text}"
분류:"""

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "지시" in result:
        return "지시"
    else:
        return "대화"
