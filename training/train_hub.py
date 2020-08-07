import transformers
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import argparse
import sys
from mrpc.model import build_model
from mrpc.data import data_loader

# data parameters
tfds_task_name = "glue/mrpc"

# model parameters
max_seq_length = 128
dropout_prob = 0.1
num_labels = 2
initializer_range = 0.02
tf_hub_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'

# training parameters
learning_rate = 3e-5
epsilon = 1e-08
clipnorm = 1.0
train_steps = 115

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog(every_n_iter=5)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int, help='training epochs')
parser.add_argument('--train_steps', default=10, type=int,
                    help='number of training steps')


def main(argv):
    with mlflow.start_run():
        # load data
        tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased')
        train_ds, valid_ds, test_ds = data_loader(
            tfds_task_name=tfds_task_name, tokenizer=tokenizer, max_seq_length=max_seq_length)

        # build model
        tfhub_model = build_model(tf_hub_path=tf_hub_path,
                                  max_seq_length=max_seq_length,
                                  dropout_prob=dropout_prob,
                                  num_labels=num_labels,
                                  initializer_range=initializer_range)

        args = parser.parse_args(argv[1:])

        # Train and evaluate using tf.keras.Model.fit()
        tfhub_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        tfhub_history = tfhub_model.fit(train_ds, epochs=args.epochs, steps_per_epoch=args.train_steps,
                                        validation_data=valid_ds, validation_steps=7)

        # evaluate
        result = tfhub_model.evaluate(valid_ds)

        mlflow.log_metric("test_acc_0", result[0])
        mlflow.log_metric("test_acc_1", result[1])


if __name__ == '__main__':
    main(sys.argv)
