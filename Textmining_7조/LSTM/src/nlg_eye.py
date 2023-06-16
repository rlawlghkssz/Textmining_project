import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import config

df_eye = pd.read_csv(config.LSTM_EYE_DATA_TRAIN_PATH, sep='\t')

"""NULL값은 없으므로 결측값 제거 과정은 생략"""

headline = []
headline.extend(list(df_eye.text.values))


def repreprocessing(raw_sentence):
    preproceseed_sentence = raw_sentence.encode(
        "utf8").decode("ascii", 'ignore')
    # 구두점 제거와 동시에 소문자화
    return ''.join(word for word in preproceseed_sentence if word not in punctuation).lower()


preporcessed_headline = [repreprocessing(x) for x in headline]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preporcessed_headline)
vocab_size = len(tokenizer.word_index) + 1

sequences = list()

for sentence in preporcessed_headline:
    # 각 샘플에 대한 정수 인코딩
    encoded = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)


index_to_word = {}
for key, value in tokenizer.word_index.items():     # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

# print('빈도수 상위 1번 단어 : {}'.format(index_to_word[1]))

max_len = max(len(l) for l in sequences)

sequences = pad_sequences(sequences, maxlen=max_len,
                          padding='pre')  # zero padding

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)

"""### model"""

model = load_model(config.LSTM_EYE_MODEL_PATH)

# 모델, 토크나이저, 현재 단어, 반복할 횟수


def sentence_generation(current_word, n, model=model, tokenizer=tokenizer):
    init_word = current_word
    sentence = ''
    temperature = 0.025  # adjust the temperature to control randomness

    # n번 반복
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre')

        # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        # result = np.argmax(result, axis=1)
        next_word_probs = result[0]
        next_word_probs = next_word_probs ** (1.0 / temperature)
        next_word_probs /= np.sum(next_word_probs)
        next_word_index = np.random.choice(
            len(next_word_probs), p=next_word_probs)

        for word, index in tokenizer.word_index.items():
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
            # if index == result:
            if index == next_word_index:
                break

        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' ' + word

        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence


# print(sentence_generation('eye color is blue.', 10))
