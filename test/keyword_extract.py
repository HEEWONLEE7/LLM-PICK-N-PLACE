# keyword_extract.py
import torch

def extract_keywords(command, tokenizer, model):
    prompt = f"""다음 지시 문장에서 핵심 키워드(action, object, color 등)를 뽑아줘. 예시는 다음과 같아:

예시:
문장: "파란 공을 정리해줘"
키워드: ['정리', '공', '파란']

문장: "{command}"
키워드:"""

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 대괄호 안에 있는 키워드만 추출
    start = result.find("[")
    end = result.find("]")
    if start != -1 and end != -1:
        return result[start:end+1]
    else:
        return "[]"
