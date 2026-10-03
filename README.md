# TF-Text-Tokenizer-Suite 🧠

A low-level, highly optimized implementation of Byte-Pair Encoding (BPE) and WordPiece tokenizers built entirely using the `tensorflow-text` library.

Designed for production environments, these tokenizers use native TensorFlow operations, making them 100% Graph-compatible. You can embed these tokenizers directly into your `SavedModel` to eliminate training-serving skew.

## Features
- **Graph-Compatible**: Decorated with `tf.function` for XLA optimization.
- **WordPiece Tokenization**: Advanced wrapper handling custom UNK tokens, unknown word fallbacks, and suffix indicators.
- **BPE Tokenization**: Handles merges and vocabulary lookups via TF lookup tables.
- **Text Normalization**: Native Unicode folding, lowercasing, and regex-based punctuation handling.

## Installation
```bash
pip install -r requirements.txt
python setup.py install
