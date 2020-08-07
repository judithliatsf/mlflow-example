import tensorflow as tf
import tensorflow_hub as hub

# define hyperparameters
max_seq_length = 128
dropout_prob = 0.1
num_labels = 2
initializer_range = 0.02
tf_hub_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'

def build_model(tf_hub_path, max_seq_length, dropout_prob, num_labels, initializer_range):
    # creat bert_layer
    
    bert_layer = hub.KerasLayer(tf_hub_path, trainable=True)

    # load module as a Keras Layer and create a classifier
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="attention_mask")
    token_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                name="token_type_ids")
    pooled_output, _ = bert_layer([input_ids, attention_mask, token_type_ids])
    x  = tf.keras.layers.Dropout(dropout_prob)(pooled_output)
    output  = tf.keras.layers.Dense(num_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range))(x)

    tfhub_model = tf.keras.models.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    
    return tfhub_model