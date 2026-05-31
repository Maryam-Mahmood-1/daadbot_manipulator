import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'daadbot_clf_cbf'

launch_files = []
for launch_file in glob(os.path.join(package_name, '**', '*.launch.py'), recursive=True):
    launch_dir = os.path.relpath(os.path.dirname(launch_file), package_name)
    launch_files.append((
        os.path.join('share', package_name, 'launch', launch_dir),
        [launch_file],
    ))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + launch_files,
    package_data={
        '': ['*.launch.py', '*.npz', '*.pkl', '*.pth'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maryam-mahmood',
    maintainer_email='maryam-mahmood@todo.todo',
    description='CLF/CBF and conformally robust CLF/CBF controllers for DAADBot.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'CRCLF = daadbot_clf_cbf.CRCLF:main',
            'compare_clf = daadbot_clf_cbf.compare_clf:main',
            'clf_oper = daadbot_clf_cbf.clf_oper:main',
            'main_controller = daadbot_clf_cbf.CLF_CBF_traj.main_controller:main',
            'cr_main_controller = daadbot_clf_cbf.CRCLF_CRCBF_traj.cr_main_controller:main',
            'collect_calibration_data = daadbot_clf_cbf.CRCLF_CRCBF_traj.collect_calibration_data:main',
            'compare_controllers_robust = daadbot_clf_cbf.CRCLF_CRCBF_traj.compare_controllers_robust:main',
            'main_node = daadbot_clf_cbf.CLF_CBF.main_node:main',
            'cr_main_node = daadbot_clf_cbf.CRCLF_CRCBF.cr_main_node:main',
            'trajectory_visualizer = daadbot_clf_cbf.CLF_CBF.trajectory_visualizer:main',
            'safety_visualizer = daadbot_clf_cbf.CLF_CBF.safety_visualizer:main',
            'compare_robust = daadbot_clf_cbf.CRCLF_CRCBF.compare_robust_models:main',
            'pinocchio_main_node = daadbot_clf_cbf.CLF_CBF.pinocchio_main_node:main',
            'cr_pinocchio_main = daadbot_clf_cbf.CRCLF_CRCBF.cr_pinocchio_main:main',
            'main_2_link = daadbot_clf_cbf.CLF_CBF_2_link.main_node:main',
            'pinocchio_main_node_2_link = daadbot_clf_cbf.CLF_CBF_2_link.pinocchio_main_node:main',
            'cr_pinocchio_2_link = daadbot_clf_cbf.CRCLF_CRCBF_2_link.pinocchio_main_node:main',
            'trajectory_visualizer_2_link = daadbot_clf_cbf.CLF_CBF_2_link.trajectory_visualizer:main',
            'safety_visualizer_2_link = daadbot_clf_cbf.CLF_CBF_2_link.safety_visualizer:main',
            'cr_main_2_link = daadbot_clf_cbf.CRCLF_CRCBF_2_link.main_node:main',
            'cr_pinocchio_main_2_link = daadbot_clf_cbf.CRCLF_CRCBF_2_link.pinocchio_main_node:main',
            'pinocchio_main_2_link = daadbot_clf_cbf.CLF_CBF_2_link.pinocchio_main_node:main',
            'quantile_data = daadbot_clf_cbf.CRCLF_CRCBF_2_link.quantile_data:run_pipeline',
            'compare_2_link_models = daadbot_clf_cbf.CRCLF_CRCBF_2_link.compare_models:main',
            'compare_2_link_robust = daadbot_clf_cbf.CRCLF_CRCBF_2_link.compare_models:main',
            'compare_models_fbl_crclf = daadbot_clf_cbf.CRCLF_CRCBF_2_link.compare_models_fbl_crclf:main',
            'pend_main_node = daadbot_clf_cbf.CLF_CBF_pend.main_node:main',
            'cr_7dof_main = daadbot_clf_cbf.CRCLF_CRCBF_7_dof.main_node:main',
            'quantile_data_7_dof = daadbot_clf_cbf.CRCLF_CRCBF_7_dof.quantile_data:run_pipeline',
            'calculate_quantile = daadbot_clf_cbf.CRCLF_CRCBF_2_link.calculate_quantile:main',
            'cr_main_2_dof = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.main_node:main',
            'cr_pinocchio_main_2_dof = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.pinocchio_main:main',
            'compare_2_dof_models = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.compare_models_fbl_crclf:main',
            'traj_sim_multi = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.traj_sim:main',
            'task_space_stab_main = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.task_space_main:main',
            'task_space_stab_pin = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.task_space_pinocchio:main',
            'task_space_traj_multi = daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.traj_sim:main',
        ],
    },
)
