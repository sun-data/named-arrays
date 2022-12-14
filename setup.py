from setuptools import setup, find_packages

setup(
    name='named_arrays',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'typing-extensions',
        'pytest',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy @ git+https://github.com/byrdie/astropy.git@bugfix/array_equal-incompatible-unitsd#egg=astropy',
        'sphinx-autodoc-typehints',
        'astropy-sphinx-theme',
        'jupyter-sphinx',
    ],
    url='https://github.com/Kankelborg-Group/named_arrays',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Library that implements named tensors with astropy.units.Quantity support.'
)
