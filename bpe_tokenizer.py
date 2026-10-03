import tensorflow as tf
import tensorflow_text as tf_text
from .normalizer import TextNormalizer

class BPETokenizer(tf.Module):
    """
    A Graph-compatible BPE Tokenizer utilizing TensorFlow Text's Sentencepiece 
    operations. Designed to load standard BPE models directly into the TF Graph.
    """
    def __init__(self, model_byte_string, name=None):
        """
        Args:
            model_byte_string: Raw bytes of a trained SentencePiece BPE model file.
                               (e.g., loaded via tf.io.gfile.GFile('model.model', 'rb').read())
        """
        super().__init__(name=name)
        # We use SentencepieceTokenizer under the hood for true low-level BPE execution
        self.tokenizer = tf_text.SentencepieceTokenizer(
            model=model_byte_string,
            out_type=tf.int32
        )
        self.normalizer = TextNormalizer(lowercase_text=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, texts):
        """Tokenizes text using Byte-Pair Encoding algorithm."""
        # Note: BPE models usually handle their own normalization, 
        # but custom pre-processing can be added here.
        normalized = self.normalizer.normalize(texts)
        return self.tokenizer.tokenize(normalized)
        
    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)])
    def detokenize(self, token_ids):
        """Converts BPE token IDs back to a dense string."""
        return self.tokenizer.detokenize(token_ids)
