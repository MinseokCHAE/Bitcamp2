from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array(range(100)) # 0~99
y = np.array(range(1,101)) # 1~100

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66) 
# shuffle=True 디폴트
# test or train_size 노상관
# 공홈설명 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 구성원리 https://rfriend.tistory.com/519 

print(x_test)
print(y_test)

'''
shuffle 활용해서 x,y 동등하게 랜덤, 7:3 분할
np.random.shuffle 활용, 매개 함수 s 할당해서 인덱스로 활용
x, y 각 두 함수가 동일한 인덱스가 셔플됨
s = np.arange(x.shape[0])
np.random.shuffle(s)
x1 = x[s]
y1 = y[s]
x_train = x1[:70]
y_train = y1[:70]
x_test = x1[-30:]
y_test = y1[-30:]
'''
