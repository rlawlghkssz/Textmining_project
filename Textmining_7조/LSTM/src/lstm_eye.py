from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import config

df_eye = pd.read_csv(config.LSTM_EYE_DATA_TRAIN_PATH, sep='\t')
df_eye.head()

"""NULL값은 없으므로 결측값 제거 과정은 생략"""

headline = []
headline.extend(list(df_eye.text.values))

# 전처리


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

# sequences[:11]

index_to_word = {}
for key, value in tokenizer.word_index.items():  # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

# print('빈도수 상위 1번 단어 : {}'.format(index_to_word[1]))

max_len = max(len(l) for l in sequences)
# print('샘플의 최대 길이 : {}'.format(max_len))

sequences = pad_sequences(sequences, maxlen=max_len,
                          padding='pre')  # zero padding
# print(sequences[:3])

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)

"""### model"""


embedding_dim = 10
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

model.save(config.LSTM_EYE_MODEL_PATH)
