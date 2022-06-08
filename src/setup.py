from setuptools import setup, find_packages
import my_wrapper

requirements = [
    'jupyter==1.0.0',
    'numpy==1.22.2',
    'matplotlib==3.4',
    'requests==2.25.1',
    'pandas==1.2.4',
    'scikit-learn==1.0.2'
]

setup(
    name='my_wrapper',
    version=my_wrapper.__version__,
    python_requires='>=3.5',
    author='Miguel García',
    author_email='miguel.garglez@gmail.com',
    description='Wrapped gluon custom package',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)