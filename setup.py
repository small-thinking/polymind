from setuptools import setup, find_packages
from pathlib import Path

# Read from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

repo = Path(__file__).resolve().parent
long_description = (repo / "README.md").read_text(encoding="utf-8")

setup(
    name="polymind",
    version="0.0.3",
    packages=find_packages(),
    install_requires=requirements,
    author="Small Thinking",
    author_email="yjiang@small-thinking.org",
    description="""
    PolyMind is a cutting-edge multi-agent framework focused on leveraging collective intelligence
    to solve complex problems.
    """,
    long_description=long_description,
    license="MIT",
    python_requires=">=3.10",
    keywords="multi-agent, collective intelligence, problem solving, polymind",
    long_description_content_type="text/markdown",
    url="https://github.com/small-thinking/polymind",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)