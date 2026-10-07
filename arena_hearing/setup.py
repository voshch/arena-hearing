import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_hearing'


def _tree(directory: str) -> list[tuple[str, list[str]]]:
    files: dict[str, list[str]] = {}
    for root, _, names in os.walk(directory):
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                files.setdefault(os.path.join('share', package_name, root), []).append(path)
    return sorted(files.items())


setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *_tree('config'),
        (os.path.join('share', package_name, 'launch'), [path for path in glob('launch/*.launch.py') if os.path.isfile(path)]),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Arena robot hearing',
    license='TODO',
    entry_points={
        'console_scripts': [
            'hearing_belief_node = arena_hearing.belief_node:main',
            'hearing_policy = arena_hearing.policy_node:main',
            'hearing_seld_frontend = arena_hearing.seld_frontend_node:main',
            'hearing_srp_frontend = arena_hearing.srp_frontend_node:main',
            'hearing_setup = arena_hearing.weights:main',
            'hearing_audio_replay = arena_hearing.audio_replay:main',
        ]
    },
)
