from setuptools import setup, find_packages

setup(
    name='tf_tools',
    version='0.1.7',
    author='Nivratti Boyane',
    author_email='your.email@example.com',
    description='A set of tools for enhancing TensorFlow 2 projects with custom metrics, loss functions, visualizations, and attention layers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Nivratti/tf-tools',
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
