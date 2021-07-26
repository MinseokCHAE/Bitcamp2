import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


docs = ['너무 재밌습니다', '참 최고입니다', '참 잘 만든 영화입니다', '추천하고 싶은 영화입니다', '한 번 더 보고 싶습니다',
            '글쎄요', '별로입니다', '생각보다 지루합니다', '연기가 어색합니다', '재미없습니다', ' 너무 재미없다', 
            '참 재밌습니다', '청순이가 잘 생기긴 했어요']

# 라벨링 -> 긍정: 1, 부정: 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index) 
'''
{'참': 1, '너무': 2, '재밌습니다': 3, '잘': 4, '영화입니다': 5, '최고입니다': 6, '만든': 7, '추천
하고': 8, '싶은': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶습니다': 14, '글쎄요': 15, '별 
로입니다': 16, '생각보다': 17, '지루합니다': 18, '연기가': 19, '어색합니다': 20, '재미없습니다': 
21, '재미없다': 22, '청순이가': 23, '생기긴': 24, '했어요': 25}
'''

x = token.texts_to_sequences(docs)
# print(x) # [[2, 3], [1, 6], [1, 4, 7, 5], [8, 9, 5], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 3], [23, 4, 24, 25]]
# 데이터 별로 사이즈 제각각 -> 가장 큰 사이즈 기준으로 나머지 0 채움(*앞에서부터)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # pre <-> post, 특이치 배제를 위해 maxlen 조절가능(앞에서부터 잘림)
# print(pad_x, pad_x.shape)
'''
[[ 0  0  0  2  3][ 0  0  0  1  6][ 0  1  4  7  5][ 0  0  8  9  5][10 11 12 13 14][ 0  0  0  0 15][ 0  0  0  0 16]
[ 0  0  0 17 18][ 0  0  0 19 20][ 0  0  0  0 21][ 0  0  0  2 22][ 0  0  0  1  3][ 0 23  4 24 25]] 
(13, 5)
 '''

from tensorflow.keras.utils import to_categorical
# x = to_categorical(pad_x)
# print(x.shape) # (13, 5, 26) -> 라벨이 많을수록 데이터의 크기가 과도하게 커짐
# 대안으로 Embedding 사용 -> 벡터화

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Conv1D
# print(len(token.word_index)) # 25: 단어 종류 (input_dim)
# print(np.unique(pad_x)) # 0~25 -> 26: 단어 종류 (25) +1 (0포함)
input = Input(shape=(5,)) # 5: 문장길이, 단어수 (input_length), shape=(None, )으로 생략가능(자동으로 설정)
a = Embedding(26, 10)(input) # 26: 단어 종류+1 (input_dim), 10: output node갯수 (output_dim)
a = Conv1D(2, 2, activation='relu')(a)
output = Dense(1, activation='sigmoid')(a)
model = Model(inputs=input, outputs=output)
# model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
embedding (Embedding)        (None, 5, 10)             260
_________________________________________________________________
conv1d (Conv1D)              (None, 4, 2)              42
_________________________________________________________________
dense (Dense)                (None, 4, 1)              3
=================================================================
Total params: 305
Trainable params: 305
Non-trainable params: 0
'''

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(pad_x, labels, epochs=100, batch_size=1)

# acc = model.evaluate(pad_x, labels)[1]
# print('acc = ', acc)
# # acc =  1.0 과적합
