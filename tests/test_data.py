import tensorflow_hub as hub
import tensorflow as tf
from mrpc.model import build_model
from mrpc.data import data_loader
import transformers

class BertLayerTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        """tests required bert module stored at `smrt-hub/bert-base-uncased-tfhub/1` """
        dropout_prob = 0.1
        num_labels = 2
        initializer_range = 0.02
        tf_hub_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
        cls.max_seq_length = 7
        cls.features = {"token_type_ids": tf.constant([[0, 0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0, 0]]),
                        "input_word_ids": tf.constant([[2673, 2025, 5552, 2097, 2022, 2439, 1012],
                                                       [100,    0,    0,    0,    0,    0,    0]]),
                        "attention_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1],
                                                       [1, 0, 0, 0, 0, 0, 0]])}
        cls.model = build_model(tf_hub_path=tf_hub_path, max_seq_length=cls.max_seq_length,
        dropout_prob=dropout_prob, num_labels=num_labels, initializer_range=initializer_range)
    
    def test_data_loader(self):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        _, _, test_ds = data_loader(tfds_task_name="glue/mrpc", tokenizer=tokenizer, max_seq_length=self.max_seq_length)

        features, labels = next(iter(test_ds))
        pred = self.model(features)
        self.assertAllEqual(pred.shape, [32, 2])



