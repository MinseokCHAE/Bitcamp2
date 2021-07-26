from tensorflow.keras.preprocessing.text import Tokenizer


text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])
# print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7} # 빈도수에 따라 정렬
x = token.texts_to_sequences([text])
# print(x) # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]

# 다중분류 -> one hot encoding 필요
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
# print(word_size) #7
x = to_categorical(x)
# print(x, x.shape) # (1, 9, 8) -> 0 자동채움으로 라벨 0~7 8개
