import  random
import  rclpy
from  rclpy.node  import  Node
from  rm_msg.msg  import  Target

class  TargetPublisher(Node):
    def  __init__(self):
        super().__init__('target_publisher')
        self.publisher = self.create_publisher(Target, 'targets', 10)
        timer_period = self.declare_parameter('timer_period', 0.5).value
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
        self.get_logger().info('Pulisher 已启动 ')

    def  timer_callback(self):
        msg = Target()
        msg.id = self.count
        msg.x = random.uniform(-5.0, 5.0)
        msg.y = random.uniform(-5.0, 5.0)
        msg.confidence = random.uniform(0.5, 1.0)
        self.publisher.publish(msg)
        self.get_logger().info(f'发送: id={msg.id}, x={msg.x:.2f}, y={msg.y:.2f}, confidence={msg.confidence:.2f}')
        self.count += 1

def main():
    rclpy.init()
    node = TargetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()