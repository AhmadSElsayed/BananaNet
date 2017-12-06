## Not working yet 
from setuptools import setup, find_packages

setup(name='BananaNet',
    version='5.0',
    description='Keras + Caffee training. mxnet predict. OpenCV Camera + Draw',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'mxnet', 
        'opencv-python', 
        'numpy'
    ]   
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'BananaNet = __init__.py'
        ]

    }
)
