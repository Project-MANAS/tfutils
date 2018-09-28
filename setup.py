from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tfutils',
    version='0.0',
    description='Tensorflow utilites',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/pypa/sampleproject',
    author='Project MANAS',  # Optional
    author_email='projectmanas.mit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Development Utilities',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='tensorflow utilities',
    packages=['tfutils'],
    install_requires=['numpy'],
    extras_require={
        "tensorflow": ["tensorflow>=1.0.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.0.0"],
    },
    project_urls={  # Optional
        'Bug Reports': '',
        'Source': '',
    },
)