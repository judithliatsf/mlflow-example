import transformers
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import argparse
import sys
import os

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

# Train and evaluate using tf.keras.Model.fit()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define metrics
train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
eval_loss = tf.keras.metrics.Mean("eval_loss", dtype=tf.float32)
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('eval_accuracy')

# Define training and testing functions
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        logits = model(x_train, training=True)
        loss = loss_object(y_train, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    predictions = tf.nn.softmax(logits)

    train_loss(loss)
    train_accuracy(y_train, predictions)


def eval_step(model, x_valid, y_valid):
    logits = model(x_valid)
    loss = loss_object(y_valid, logits)
    predictions = tf.nn.softmax(logits)

    eval_loss(loss)
    eval_accuracy(y_valid, predictions)


# Define input args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int, help='training epochs')
parser.add_argument('--train_steps', default=10, type=int,
                    help='maximum number of training steps per epoch')


def main(argv):
    with mlflow.start_run(run_name="training"):
        # step 1: build and install package
        mlflow.run(".", "build")
        from mrpc.model import build_model
        from mrpc.data import data_loader
        
        # step 2: main
        # load data
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        train_ds, valid_ds, test_ds = data_loader(
            tfds_task_name=tfds_task_name, tokenizer=tokenizer, max_seq_length=max_seq_length)

        # build model
        tfhub_model = build_model(tf_hub_path=tf_hub_path,
                                max_seq_length=max_seq_length,
                                dropout_prob=dropout_prob,
                                num_labels=num_labels,
                                initializer_range=initializer_range)

        args = parser.parse_args(argv[1:])
        mlflow.log_params({
            "dropout_prob": dropout_prob,
            "initializer_range": initializer_range,
            "max_seq_length": max_seq_length
        })

        # training
        for epoch in range(args.epochs):
            for step, (x_train, y_train) in enumerate(train_ds):
                global_step = optimizer.iterations.numpy()
                train_step(tfhub_model, optimizer, x_train, y_train)

                if step > args.train_steps:
                    break

                if global_step % 5 == 0:
                    mlflow.log_metric(
                        "training_loss_per_step", train_loss.result().numpy(), step=global_step)
                    mlflow.log_metric(
                        "training_acc_per_step", train_accuracy.result().numpy(), step=global_step)

                    # reset eval loss and acc
                    eval_loss.reset_states()
                    eval_accuracy.reset_states()

                    for (x_valid, y_valid) in valid_ds:
                        eval_step(tfhub_model, x_valid, y_valid)

                    mlflow.log_metric(
                        "eval_loss_per_step", eval_loss.result().numpy(), step=global_step)
                    mlflow.log_metric(
                        "eval_acc_per_step", eval_accuracy.result().numpy(), step=global_step)
                    print("epoch : {:03d}, step: {:03d}, avg training loss: {:.6f}, avg training acc: {:.4%}, validation loss: {:.6f}, validation acc: {:.4%}".format(
                        epoch,
                        step,
                        train_loss.result().numpy(),
                        train_accuracy.result().numpy(),
                        eval_loss.result().numpy(),
                        eval_accuracy.result().numpy()
                    ))

            # record epoch loss
            mlflow.log_metric("train_loss_per_epoch",
                              train_loss.result().numpy(), step=epoch)
            mlflow.log_metric("train_acc_per_epoch",
                              train_accuracy.result().numpy(), step=epoch)

            # reset at the end of epoch
            train_loss.reset_states()
            train_accuracy.reset_states()

            # save model and checkpoints
            local_artifacts_path = "/tmp/artifacts"
            local_model_path = os.path.join(local_artifacts_path, "savedmodel")
            local_ckpt_path = os.path.join(local_artifacts_path, "checkpoints")
            ckpt_name = "model"

            tfhub_model.save_weights(os.path.join(local_ckpt_path, ckpt_name))
            tfhub_model.save(local_model_path, save_format="tensorflow")

            # log model and checkpoints to mlflow run as artifacts
            mlflow.log_artifacts(local_ckpt_path, artifact_path="checkpoints")
            mlflow.tensorflow.log_model(
                tf_saved_model_dir=local_model_path,
                tf_meta_graph_tags=["serve"],
                tf_signature_def_key="serving_default",
                artifact_path="models")

        # evaluate on test dataset
        eval_loss.reset_states()
        eval_accuracy.reset_states()

        for (x_test, y_test) in test_ds:
            eval_step(tfhub_model, x_test, y_test)

        mlflow.log_metric("test_loss", eval_loss.result().numpy())
        mlflow.log_metric("test_acc", eval_accuracy.result().numpy())


if __name__ == '__main__':
    main(sys.argv)
