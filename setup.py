from setuptools import setup, find_packages

with open('INFO.md', 'r') as fh:
    long_desc = fh.read()

setup(
    name='pypair',
    version='2.0.0',
    author='Jee Vang',
    author_email='vangjee@gmail.com',
    packages=find_packages(exclude=('*.tests', '*.tests.*', 'tests.*', 'tests')),
    description='Pairwise association measures of statistical variable types',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/oneoffcoder/py-pair',
    keywords=' '.join(
        ['statistics', 'pairwise', 'association', 'correlation', 'concordance', 'measurement', 'strength', 'pyspark']),
    install_requires=['scipy', 'numpy', 'pandas', 'scikit-learn', 'pyspark'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable'
    ],
    include_package_data=True,
    test_suite='nose.collector'
)
