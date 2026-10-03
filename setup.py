import tensorflow as tf
from tf_text_tokenizer_suite import AdvancedWordPieceTokenizer

vocab = ["[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s"]
tokenizer = AdvancedWordPieceTokenizer(vocab)

text = tf.constant(["hello worlds"])
tokens = tokenizer.tokenize(text)
print(tokens)

---


```python
from setuptools import setup, find_packages

setup(
    name="tf-text-tokenizer-suite",
    version="0.1.0",
    author="Your Name",
    description="Low-level BPE and WordPiece tokenizers using tensorflow-text",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tf-text-tokenizer-suite",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.14.0",
        "tensorflow-text>=2.14.0"
    ],
)
