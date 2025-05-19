from setuptools import setup

setup(
    name="pyroid",
    version="0.7.0",
    description="High-performance Rust functions for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="pyroid Team",
    author_email="support@ataiva.com",
    url="https://github.com/ao/pyroid",
    packages=["pyroid"],
    package_dir={"": "python"},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

