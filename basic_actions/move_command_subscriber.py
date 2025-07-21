# 파일명: command_subscriber.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class CommandSubscriber(Node):
    def __init__(self):
        super().__init__('command_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/command_topic',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        try:
            command = json.loads(msg.data)
            self.get_logger().info(f'받은 명령: {command}')
        except json.JSONDecodeError:
            self.get_logger().error('JSON 디코딩 실패')

def main():
    rclpy.init()
    node = CommandSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
