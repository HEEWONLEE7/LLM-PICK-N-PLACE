import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
import math

# 링크 길이 (m)
L2 = 0.128
L3 = 0.124
L_GRIPPER = 0.126  # joint4 ~ gripper 끝까지 수평 길이

class CommandSubscriber(Node):
    def __init__(self):
        super().__init__('command_subscriber')

        self.subscription = self.create_subscription(
            String,
            '/command_topic',
            self.listener_callback,
            10
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0  # 고정
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_z = L2 + L3 * math.sin(self.current_joint3_pos)

    def listener_callback(self, msg):
        try:
            command = json.loads(msg.data)
            self.get_logger().info(f'Received command: {command}')

            action = command.get("action")
            direction = command.get("direction")
            value = command.get("value")
            unit = command.get("unit")

            if action == "initialize":
                self.reset_pose()
                return

            traj = JointTrajectory()
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            point = JointTrajectoryPoint()

            # ✅ joint3 각도에 따라 회전 반지름 계산
            radius = L3 * math.cos(self.current_joint3_pos) + L_GRIPPER

            # 단위 변환 → 각도 (rad)
            delta = self.get_delta(value, unit, radius)

            if action == "rotate":
                if direction in ["left", "right"]:
                    self.current_joint1_pos += delta if direction == "left" else -delta

                elif direction in ["up", "down"]:
                    sign = -1 if direction == "up" else 1
                    target_joint3 = self.current_joint3_pos + sign * delta
                    if not (-math.pi/2 <= target_joint3 <= math.pi/2):
                        self.get_logger().warn("❌ joint3 범위 초과")
                        return
                    self.current_joint3_pos = target_joint3
                    self.current_joint4_pos = -target_joint3
                    self.current_z = L2 + L3 * math.sin(target_joint3)

                else:
                    self.get_logger().warn("❌ rotate 방향 오류")
                    return

            elif action == "move":
                if direction in ["left", "right"]:
                    self.current_joint1_pos += delta if direction == "left" else -delta

                elif direction in ["up", "down"]:
                    sign = -1 if direction == "up" else 1
                    delta_z = sign * (value / 100.0)
                    target_z = self.current_z + delta_z
                    success, theta3 = self.inverse_kinematics_z(target_z)
                    if not success:
                        self.get_logger().warn("❌ IK 실패")
                        return
                    self.current_joint3_pos = theta3
                    self.current_joint4_pos = -theta3
                    self.current_z = target_z

                elif direction in ["forward", "backward"]:
                    sign = 1 if direction == "forward" else -1
                    target_joint3 = self.current_joint3_pos + sign * delta
                    if not (-math.pi/2 <= target_joint3 <= math.pi/2):
                        self.get_logger().warn("❌ joint3 범위 초과")
                        return
                    self.current_joint3_pos = target_joint3
                    self.current_joint4_pos = -target_joint3
                    self.current_z = L2 + L3 * math.sin(target_joint3)

                else:
                    self.get_logger().warn("❌ move 방향 오류")
                    return

            else:
                self.get_logger().warn("❌ 지원되지 않는 action")
                return

            point.positions = [
                self.current_joint1_pos,
                self.current_joint2_pos,
                self.current_joint3_pos,
                self.current_joint4_pos
            ]
            point.time_from_start.sec = 2
            traj.points.append(point)
            self.trajectory_pub.publish(traj)

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

    def get_delta(self, value, unit, radius):
        if unit == "degree":
            return math.radians(value)
        elif unit == "rad":
            return value
        elif unit == "cm":
            arc_length = value / 100.0  # cm → m
            if radius == 0:
                self.get_logger().warn("⚠️ 반지름 0 → 회전 불가")
                return 0.0
            return arc_length / radius  # θ = s / r
        else:
            self.get_logger().warn(f"⚠️ 지원되지 않는 단위: {unit}")
            return 0.0

    def inverse_kinematics_z(self, z_target):
        try:
            delta = z_target - L2
            if abs(delta / L3) > 1.0:
                return False, 0.0
            theta3 = math.asin(delta / L3)
            return True, theta3
        except Exception as e:
            self.get_logger().error(f"IK 계산 오류: {e}")
            return False, 0.0

    def reset_pose(self):
        self.current_joint1_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_z = L2

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 2
        traj.points.append(point)
        self.trajectory_pub.publish(traj)

        self.get_logger().info("✅ 초기자세로 이동 완료.")

def main():
    rclpy.init()
    node = CommandSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
