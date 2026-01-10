import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'vehicle_dynamics_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    scripts=glob('scripts/*.py'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yxk-vtd',
    maintainer_email='740365168@qq.com',
    description='Vehicle dynamics simulation with ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_env_node = start_env_node:main',
            'start_action_node = start_action_node:main',
        ],
    },
)
