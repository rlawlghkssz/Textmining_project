import tensorflow as tf
from bert import run_classifier
import config
from BERT.src.model import model_fn_builder
from BERT.src.create_tokenizer import create_tokenizer_from_hub_module as create_tokenizer


MODEL_DIR = config.BERT_HAIR_MODEL_PATH
MAX_SEQ_LENGTH = 200
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100


BERT_MODEL_HUB = config.BERT_MODEL_PATH
tokenizer = create_tokenizer(BERT_MODEL_HUB)


DATA_COLUMN = 'description'
LABEL_COLUMN = 'hair_color'
label_list = [0, 1, 2, 3, 4, 5, 6, 7]

# 202: train dataset 의 길이
num_train_steps = int(202 / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR,
                                    save_summary_steps=SAVE_SUMMARY_STEPS,
                                    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(num_labels=len(label_list),
                            learning_rate=LEARNING_RATE,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps,
                            bert_model_hub=BERT_MODEL_HUB)
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   config=run_config,
                                   params={"batch_size": BATCH_SIZE})


def get_prediction(in_sentences):
    labels = ['brown', 'red', 'gray', 'black',
              'blonde', 'white', 'orange', 'idk']
    labels_list = [0, 1, 2, 3, 4, 5, 6, 7]
    input_examples = [run_classifier.InputExample(
        guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
    input_features = run_classifier.convert_examples_to_features(
        input_examples, labels_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)

    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
