import os
from setuptools import setup, find_packages

package_name = 'vehicle_dynamics_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yxk-vtd',
    maintainer_email='740365168@qq.com',
    description='Vehicle dynamics simulation with ROS2 Galactic and Python 3.8',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            #'start_env_node = vehicle_dynamics_sim.scripts.start_env_node:main',
            #'start_action_node = vehicle_dynamics_sim.scripts.start_action_node:main',
        ],
    },
)