import tensorflow as tf
import tensorflow_text as tf_text

class TextNormalizer(tf.Module):
    """Native TF Text Normalizer for Graph-compatible preprocessing."""
    
    def __init__(self, lowercase_text=True, keep_whitespace=False, name=None):
        super().__init__(name=name)
        self.lowercase = lowercase_text
        self.keep_whitespace = keep_whitespace

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def normalize(self, text_input):
        """Applies Unicode normalization (NFKC) and basic stripping."""
        # NFKC Normalization via TF Text
        normalized = tf_text.normalize_utf8(text_input, "NFKC")
        
        if self.lowercase:
            normalized = tf_text.case_fold_utf8(normalized)
            
        if not self.keep_whitespace:
            # Replace multiple spaces with a single space
            normalized = tf.strings.regex_replace(normalized, r"\s+", " ")
            normalized = tf.strings.strip(normalized)
            
        return normalized
