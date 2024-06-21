from setuptools import setup, find_packages

setup(
    name="transcribe",
    version="0.1.0",
    author="Aeron Laffere",
    author_email="ajlaffere@gmail.com",
    description="Utilities for transcribing audio files using the Whisper API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aeronjl/transcribe",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "openai",
        "numpy",
        "ffmpeg-python",
        "pydub",
        "tiktoken",
        "python-dotenv"
        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
