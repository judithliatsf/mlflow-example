# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import tensorflow as tf
import transformers
import warnings
import mlflow

from transformers import InputExample

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('layer_name', 'bert-base-uncased', 'pre-trained model name')

text1 = "penguin are flightless gerbs"
text2 = "why can bird fly"

example = InputExample(
  guid = 111,
  text_a = text1,
  text_b = text2,
  label = "1"
)

def evaluate(example, tokenizer, model):
    features = transformers.glue_convert_examples_to_features([example], tokenizer, max_length=128, task='mrpc')
  
    bert_input = {"input_ids": tf.constant([f.input_ids for f in features]),
         "token_type_ids": tf.constant([f.token_type_ids for f in features]),
         "attention_mask": tf.constant([f.attention_mask for f in features])}
    pred = model(bert_input)

    prob = pred[0][0,0].numpy()
    return prob



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.random.set_seed(1234)
    
    print("tf version: {}".format(tf.__version__))
    print("mlflow version: {}".format(mlflow.__version__))

    bert_tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.layer_name)
    bert_model = transformers.TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    with mlflow.start_run():
        class_prob = evaluate(example, bert_tokenizer, bert_model)

        print("Model Prediction: %s" % class_prob)

        mlflow.log_param("class_prob", class_prob)
