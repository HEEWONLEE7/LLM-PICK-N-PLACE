# main.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mode_detect import detect_mode
from keyword_extract import extract_keywords
from action_response import generate_action_response

def main():
    model_path = "/home/robotis/.cache/huggingface/hub/models--beomi--KoAlpaca-Polyglot-12.8B/snapshots/5f225e9c5ae6c7238fc2316da0b8a9922019674d"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ 프로그램 실행됨: 자연어 입력. 'exit' 입력 시 종료됩니다.")

    while True:
        user_input = input("👤 사용자: ")
        if user_input.strip().lower() == "exit":
            break

        mode = detect_mode(user_input, tokenizer, model)
        print(f"[DEBUG] 판단된 모드: {mode}")

        if mode == "대화":
            prompt = f"""당신은 친절하고 똑똑한 AI 비서입니다.
사용자가 어떤 질문을 하든 자연스럽고 유용하게 한국어로 응답해야 합니다.

사용자: {user_input}
AI:"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = result.replace(prompt, "").strip()
            print(f"🤖 AI: {answer}")

        elif mode == "지시":
            keywords = extract_keywords(user_input, tokenizer, model)
            print(f"🔍 추출된 키워드: {keywords}")
            action_response = generate_action_response(user_input, keywords, tokenizer, model)
            print(f"🤖 AI 응답: {action_response}")

# ✅ main 함수 실행
if __name__ == "__main__":
    main()
