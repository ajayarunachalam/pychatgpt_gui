#! /usr/bin/env python


# Copyright (C) 2022-2023 Ajay Arunachalam <ajay.arunachalam08@gmail.com>
# License: MIT, ajay.arunachalam08@gmail.com

import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

__version__ = '0.0.1'

def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='pychatgpt_gui',
    version=__version__,
    packages=["pychatgpt_gui"],
    description='pyChatGPT GUI - is an open-source, low-code python GUI wrapper providing easy access and swift usage of Large Language Models (LLMs) such as ChatGPT, AutoGPT, LLaMa, GPT-J, and GPT4All with custom-data and pre-trained inferences.',
    long_description = readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/ajayarunachalam/pychatgpt_gui',
    install_requires=[
        "numpy",
        "pandas",
        "warnings",
        "scipy",
        "tqdm",
        "openai",
        "gpt_index==0.4.24",
        "PyPDF2",
        "PyCryptodome",
        "gradio",
        "scikit-learn",
        "openai_whisper",
        "TTS",
        "gTTS",
        "ffmpeg",
        "ffprobe",
        "SpeechRecognition",
        "pyaudio",
        "playsound",
        "langchain",
        "chromadb",
        "tiktoken",
        "pypdf",
        "unstructured[local-inference]",
        "nomic",
        "pyllamacpp",
        "pygpt4all",
    ],
    license='MIT',
    include_package_data=True,
    author='Ajay Arunachalam',
    author_email='ajay.arunachalam08@gmail.com')

