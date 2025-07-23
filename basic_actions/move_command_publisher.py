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

    # ▶️ 다양한 초기화 표현들
    init_keywords = [
        "initial pose", "reset position", "go back to start", "return to initial",
        "home position", "back to home", "original position", "go to default"
    ]
    if any(phrase in text.lower() for phrase in init_keywords):
        return {
            "action": "initialize",
            "direction": None,
            "value": 0,
            "unit": None
        }

    direction_map = {
        "left": "left",
        "right": "right",
        "forward": "forward",
        "backward": "backward",
        "up": "up",
        "down": "down"
    }

    reverse_direction = {
        "left": "right",
        "right": "left",
        "forward": "backward",
        "backward": "forward",
        "up": "down",
        "down": "up"
    }

    # 🔍 detect direction
    for keyword, value in direction_map.items():
        if keyword in text.lower():
            result["direction"] = value
            break

    # 🔍 extract value and unit
    value_match = re.search(r'(-?[0-9]+(?:\.[0-9]+)?)\s*(degrees?|rads?|cm|centimeter|m|meter|inch|inches)?', text.lower())
    if value_match:
        val = float(value_match.group(1))
        unit = value_match.group(2)

        if unit in ['degree', 'degrees']:
            result["unit"] = "degree"
            result["value"] = abs(val)
            result["action"] = "rotate"
        elif unit in ['rad', 'rads']:
            result["unit"] = "rad"
            result["value"] = abs(val)
            result["action"] = "rotate"
        elif unit in ['cm', 'centimeter']:
            result["unit"] = "cm"
            result["value"] = abs(val)
            result["action"] = "move"
        elif unit in ['m', 'meter']:
            result["unit"] = "cm"
            result["value"] = abs(val * 100)
            result["action"] = "move"
        elif unit in ['inch', 'inches']:
            result["unit"] = "cm"
            result["value"] = abs(val * 2.54)
            result["action"] = "move"

        # 방향 뒤집기
        if val < 0 and result["direction"] in reverse_direction:
            result["direction"] = reverse_direction[result["direction"]]

    # infer action if still missing
    if result["action"] is None:
        if any(word in text.lower() for word in ['rotate', 'turn']):
            result["action"] = "rotate"
            result["unit"] = result["unit"] or "degree"
        elif any(word in text.lower() for word in ['move', 'go', 'forward', 'backward', 'up', 'down']):
            result["action"] = "move"
            result["unit"] = result["unit"] or "cm"

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

    print("Enter a command (e.g., 'rotate left 10 degrees' or 'move forward 20 cm')")
    print("Type 'exit' to quit.")

    while rclpy.ok():
        text = input(">>> ").strip()
        if text.lower() in ["exit"]:
            print("Exiting...")
            break

        structured_command = extract_keywords(text)
        print("\n[Structured Command]")
        print(json.dumps(structured_command, indent=2, ensure_ascii=False))

        # Publish
        node.publish_command(structured_command)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
