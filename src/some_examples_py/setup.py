from setuptools import find_packages, setup

package_name = 'some_examples_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        '': ['*.pkl', '*.pth', '*.npz'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maryam-mahmood',
    maintainer_email='maryam-mahmood@todo.todo',
    description='Miscellaneous DAADBot example, learning, IK, and data collection scripts.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'IK_traj = some_examples_py.ik_tools.IK_traj:main',
            'circle_IK = some_examples_py.ik_tools.circle_IK:main',
            'ellipse_IK = some_examples_py.ik_tools.ellipse_IK:main',
            'elliptical_data = some_examples_py.ik_tools.elliptical_data:main',
            'collect_data = some_examples_py.data_collection.collect_data:main',
            'z_torque_aggregator = some_examples_py.data_collection.z_torque_aggregator:main',
            'rviz_torque_text = some_examples_py.data_collection.rviz_torque_text:main',
            'train_dynamics = some_examples_py.learning.train_dynamics:main',
            'train_ellipse_dynamics = some_examples_py.learning.train_ellipse_dynamics:main',
            'simple_publisher = some_examples_py.demos.simple_publisher:main',
            'my_action_server = some_examples_py.demos.my_action_server:main',
            'gui_trajectory = some_examples_py.demos.gui_trajectory:main',
            'gui_trajectory_2 = some_examples_py.demos.gui_trajectory_2:main',
            'gui_trajectory_3 = some_examples_py.demos.gui_trajectory_3:main',
            'gui_trajectory_hw = some_examples_py.demos.gui_trajectory_hw:main',
            'mpipe = some_examples_py.demos.mpipe:main',
            'sim_driver = some_examples_py.demos.sim_driver:main',
            'tf_visualizer = some_examples_py.demos.tf_visualizer:main',
            'ctc_controller = some_examples_py.demos.ctc_controller:main',
            'ctc_gui_ros = some_examples_py.demos.ctc_gui_ros:main',
        ],
    },
)
