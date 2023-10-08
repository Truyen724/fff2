

from pathlib import Path
from setuptools import setup, find_packages


SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / 'requirements.txt') as f:
    required = f.read().splitlines()

with open(SETUP_DIR / 'requirements_ovms.txt') as f:
    ovms_required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
packages.remove('visualizers')
package_dir = {'openvino': str(SETUP_DIR / 'openvino')}

setup(
    name='openmodelzoo-modelapi',
    version='0.0.0',
    author='Intel® Corporation',
    license='OSI Approved :: Apache Software License',
    url='https://github.com/openvinotoolkit/open_model_zoo/tree/develop/demos/common/python/openvino',
    description='Model API: model wrappers and pipelines from Open Model Zoo',
    python_requires = ">=3.7",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    packages=packages,
    package_dir=package_dir,
    install_requires=required,
    extras_require={'ovms': ovms_required}
)
