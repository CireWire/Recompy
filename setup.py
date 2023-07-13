from setuptools import setup, find_packages

setup(
    name='recompy',
    version='0.1.0',
    description='A Python library for building and training neural networks for recommender systems.',
    author='CireWire',
    author_email='dreaded.sushi@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'keras'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)
