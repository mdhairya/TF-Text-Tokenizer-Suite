import os
import tempfile
import tensorflow as tf
from tf_text_tokenizer_suite import AdvancedWordPieceTokenizer, TextNormalizer

class TestTokenizers(tf.test.TestCase):

    def test_text_normalizer(self):
        normalizer = TextNormalizer(lowercase_text=True)
        texts = tf.constant(["Hello   World!", "TENSORFLOW"])
        normalized = normalizer.normalize(texts)
        
        self.assertAllEqual(
            normalized.numpy(), 
            [b"hello world!", b"tensorflow"]
        )

    def test_wordpiece_tokenizer(self):
        # Create a tiny vocab
        vocab = ["[UNK]", "hello", "world", "ten", "##sor", "##flow"]
        tokenizer = AdvancedWordPieceTokenizer(vocab)
        
        texts = tf.constant(["Hello world", "tensorflow", "unknown_word"])
        tokens = tokenizer.tokenize(texts)
        
        # Ensure it creates a Ragged Tensor of expected structure
        self.assertIsInstance(tokens, tf.RaggedTensor)
        self.assertEqual(tokens.shape[0], 3)
        
        # Test serialization compatibility
        concrete_func = tokenizer.tokenize.get_concrete_function()
        self.assertTrue(concrete_func is not None)

if __name__ == '__main__':
    tf.test.main()
