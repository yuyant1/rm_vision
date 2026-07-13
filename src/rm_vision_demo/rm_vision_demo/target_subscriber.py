import rclpy
from rclpy.node import Node
from rm_msg.msg import Target

class TargetSubscriber(Node):
    def __init__(self):
        super().__init__('target_subscriber')
        self.subscription = self.create_subscription(
            Target,
            'targets',
            self.callback,
            10)
        self.get_logger().info('Subscriber 已启动，等待消息...')

    def callback(self, msg):
        self.get_logger().info(f'收到: id={msg.id}, x={msg.x:.2f}, y={msg.y:.2f}, confidence={msg.confidence:.2f}')

        if msg.confidence > 0.8:
            self.get_logger().warning(f' 高置信度目标！ id={msg.id}, confidence={msg.confidence:.2f}')

def main():
    rclpy.init()
    node = TargetSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()