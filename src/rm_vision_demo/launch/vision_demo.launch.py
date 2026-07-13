from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    publisher_rate = DeclareLaunchArgument(
        'publisher_rate',
        default_value='0.5',
        description='消息发布间隔 （秒）'
    )

    publisher = Node(
        package='rm_vision_demo',
        executable='target_publisher',
        name='target_publisher',
        output='screen',
        parameters=[{'timer_period': LaunchConfiguration('publisher_rate')}]
    )

    subscriber = Node(
        package='rm_vision_demo',
        executable='target_subscriber',
        name='target_subscriber',
        output='screen'
    )

    return LaunchDescription([
        publisher_rate,
        publisher,
        subscriber
    ])