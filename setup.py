from setuptools import setup, find_packages

setup(
    name='named_arrays',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
    ],
    url='https://github.com/Kankelborg-Group/named_arrays',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Library that implements named tensors with astropy.units.Quantity support.'
)
