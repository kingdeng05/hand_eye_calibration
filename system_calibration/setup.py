from setuptools import setup, find_packages

setup(
    name='system_calibration',
    version='0.1.0',
    description='kinetic System Calibration',
    author='Fuheng Deng',         
    author_email='kingdeng05@gmail.com',
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "PyYaml",
        "shapely",
        "apriltag"
    ],
)
