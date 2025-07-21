import re
import json

def extract_keywords(text):
    result = {
        "action": None,
        "direction": None,
        "value": None,
        "unit": None
    }

    # 방향 추출
    direction_map = {
        "왼쪽": "left",
        "오른쪽": "right",
        "앞": "forward",
        "앞으로": "forward",
        "뒤": "backward",
        "뒤로": "backward",
        "위": "up",
        "위로": "up",
        "아래": "down",
        "아래로": "down"
    }
    for keyword, value in direction_map.items():
        if keyword in text:
            result["direction"] = value
            break

    # 회전 관련 키워드
    if any(word in text for word in ['회전', '돌려', '돌아']):
        result["action"] = "rotate"
        result["unit"] = "degree"
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            result["value"] = float(match.group(1))

    # 이동 관련 키워드
    elif any(word in text for word in ['이동', '가', '전진', '내려', '내려와', '내려가', '올려', '올라', '올려줘']):
        result["action"] = "move"
        result["unit"] = "cm"
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            result["value"] = float(match.group(1))

    # 기타 동작
    elif any(word in text for word in ['집', '집어', '잡']):
        result["action"] = "pick"
    elif any(word in text for word in ['놔', '놓']):
        result["action"] = "place"

    return result

def main():
    print("명령을 입력하세요 (예: 왼쪽으로 10cm 회전해)")
    print("종료하려면 'exit' 또는 '종료'를 입력하세요.")

    while True:
        text = input(">>> ").strip()
        if text.lower() in ["exit", "종료"]:
            print("종료합니다.")
            break

        structured_command = extract_keywords(text)
        print("\n[구조화된 명령]")
        print(json.dumps(structured_command, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
