import tensorflow_datasets as tfds
import transformers
from transformers import glue_convert_examples_to_features


def data_loader(tfds_task_name='glue/mrpc', tokenizer=None, max_seq_length=128):
    """generate training and validation data for a tfds task

    Args:
        tfds_task_name (string): task name in tensorflow_datasets
        tokenizer (PreTrainedTokenizer): a transformer tokenizer
        max_seq_length (int): max sequence length for input

    Returns:
        train_dataset (tf.dataset): batched dataset for training
        valid_dataset (tf.dataset): batched dataset for validation
    """
    data = tfds.load(tfds_task_name)
    task_name = 'mrpc'
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=max_seq_length, task=task_name)
    valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=max_seq_length, task=task_name)
    test_dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length=max_seq_length, task=task_name)
    train_dataset = train_dataset.shuffle(100).batch(16).repeat(1)
    valid_dataset = valid_dataset.batch(32)
    test_dataset = test_dataset.batch(32)
    return train_dataset, valid_dataset, test_dataset

