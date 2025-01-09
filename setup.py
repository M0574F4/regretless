from setuptools import setup, find_packages

setup(
    name="RegretLess",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
    ],
    author="Mostafa Naseri",
    author_email="mostafa.naseri1991@gmail.com",
    description="A short description of the project.",
    license="MIT",
    url="https://github.com/yourusername/RegretLess",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
