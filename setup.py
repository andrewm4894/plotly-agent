from setuptools import setup, find_packages

setup(
    name="plotly_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "langchain-core",
        "langchain",
        "langchain-openai",
        "pydantic",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered agent for creating Plotly visualizations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plotly-agent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 