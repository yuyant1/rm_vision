from setuptools import find_packages, setup

package_name = 'rm_vision_phase3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/vision_pipeline.launch.py']),
        ('share/' + package_name + '/config',
            ['config/params.yaml']),
        ('share/' + package_name + '/rviz',
            ['rviz/vision_pipeline.rviz']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuyant1',
    maintainer_email='yuyant1@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = rm_vision_phase3.camera_node:main',
            'detector_node = rm_vision_phase3.detector_node:main',
            'tracker_node = rm_vision_phase3.tracker_node:main',
            'visualizer_node = rm_vision_phase3.visualizer_node:main',
        ],
    },
)
