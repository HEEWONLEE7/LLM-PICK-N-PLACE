import re
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def extract_keywords(text):
    result = {
        "action": None,
        "direction": None,
        "value": None,
        "unit": None
    }

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

    # 방향 감지
    for keyword, value in direction_map.items():
        if keyword in text:
            result["direction"] = value
            break

    # 값과 단위 추출 (예: 10도, 20cm, 0.3m 등)
    value_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(도|degree|cm|센티|m|미터)?', text)
    if value_match:
        val = float(value_match.group(1))
        unit = value_match.group(2)

        # 단위 변환
        if unit in ['도', 'degree']:
            result["unit"] = "degree"
            result["value"] = val
            result["action"] = "rotate"
        elif unit in ['cm', '센티']:
            result["unit"] = "cm"
            result["value"] = val
            result["action"] = "move"
        elif unit in ['m', '미터']:
            result["unit"] = "cm"
            result["value"] = val * 100  # meter → cm
            result["action"] = "move"

    # 명시적 키워드가 없는 경우 추론
    if result["action"] is None:
        if any(word in text for word in ['회전', '돌려', '돌아']):
            result["action"] = "rotate"
            result["unit"] = result["unit"] or "degree"
        elif any(word in text for word in ['이동', '가', '전진', '올려', '올라', '내려', '내려와']):
            result["action"] = "move"
            result["unit"] = result["unit"] or "cm"

    # pick/place 처리
    if any(word in text for word in ['집', '집어', '잡']):
        result["action"] = "pick"
    elif any(word in text for word in ['놔', '놓']):
        result["action"] = "place"

    return result


class CommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')
        self.publisher_ = self.create_publisher(String, '/command_topic', 10)

    def publish_command(self, command_dict):
        msg = String()
        msg.data = json.dumps(command_dict, ensure_ascii=False)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main():
    rclpy.init()
    node = CommandPublisher()

    print("명령을 입력하세요 (예: 왼쪽으로 10cm 회전해)")
    print("종료하려면 'exit' 또는 '종료'를 입력하세요.")

    while rclpy.ok():
        text = input(">>> ").strip()
        if text.lower() in ["exit", "종료"]:
            print("종료합니다.")
            break

        structured_command = extract_keywords(text)
        print("\n[구조화된 명령]")
        print(json.dumps(structured_command, indent=2, ensure_ascii=False))

        # 퍼블리시
        node.publish_command(structured_command)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
