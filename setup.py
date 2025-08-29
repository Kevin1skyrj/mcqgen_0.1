from setuptools import setup, find_packages

setup(
    name="mcqgenerator",
    version="0.1",
    author="Rajat Pandey",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "langchain",
        "langchain-community",
        "langchain-google-genai",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
    ],
)
