import tensorflow as tf
import tensorflow_text as tf_text
from .normalizer import TextNormalizer

class AdvancedWordPieceTokenizer(tf.Module):
    """
    Low-level wrapper around FastWordpieceTokenizer providing
    end-to-end normalization, regex splitting, and tokenization.
    """
    def __init__(self, vocab, unknown_token="[UNK]", max_bytes_per_word=100, name=None):
        super().__init__(name=name)
        self.unknown_token = unknown_token
        
        # Initialize Lookup Table for Vocabulary
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(vocab, dtype=tf.string),
            values=tf.range(len(vocab), dtype=tf.int64)
        )
        self.vocab_table = tf.lookup.StaticVocabularyTable(
            initializer, num_oov_buckets=1
        )
        
        # Core TF Text Tokenizer
        self.tokenizer = tf_text.FastWordpieceTokenizer(
            vocab=vocab,
            token_out_type=tf.int64,
            unknown_token=self.unknown_token,
            max_bytes_per_word=max_bytes_per_word,
            support_detokenization=True
        )
        
        # Basic whitespace/punctuation splitter
        self.whitespace_tokenizer = tf_text.WhitespaceTokenizer()
        self.normalizer = TextNormalizer(lowercase_text=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, texts):
        """Normalizes, splits into words, then applies WordPiece."""
        # 1. Normalize
        normalized = self.normalizer.normalize(texts)
        
        # 2. Split words
        words = self.whitespace_tokenizer.tokenize(normalized)
        
        # 3. Apply WordPiece
        # returns RaggedTensor: [batch, word, wordpieces]
        subwords = self.tokenizer.tokenize(words)
        
        # Merge the word and wordpiece dimensions
        # Output shape: [batch, flat_wordpieces]
        flat_subwords = subwords.merge_dims(1, 2)
        return flat_subwords

    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)])
    def detokenize(self, token_ids):
        """Detokenizes subwords back into strings."""
        return self.tokenizer.detokenize(token_ids)
