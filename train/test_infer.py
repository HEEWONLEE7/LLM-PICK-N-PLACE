import torch
from transformers import AutoTokenizer
from model import ModeClassifier  # ✅ 분류 전용 모델로 변경

def predict(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        class_logits = outputs["logits"]
    
    pred_class = torch.argmax(class_logits, dim=1).item()
    mode = "대화" if pred_class == 0 else "지시"
    return mode

# 🔽 테스트 예시
if __name__ == "__main__":
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = ModeClassifier(model_name)
    model.load_state_dict(torch.load("saved_models/mode_classifier.pt"))
    model.cuda()

    while True:
        text = input("문장을 입력하세요 (exit 입력 시 종료): ")
        if text.lower() == "exit":
            break

        mode = predict(text, model, tokenizer)
        print(f"🧠 분류 결과: {mode}")
