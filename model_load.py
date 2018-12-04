# -*- coding: utf-8 -*-
from captcha_keras import gen, decode, evaluate
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from tqdm import tqdm


width, height, n_len, n_class = 170, 80, 4, 36

input_tensor = Input((height, width, 3))
x = input_tensor

for i in range(4):
    x = Conv2D(2 * 2 ** i, 3, 3, activation='relu')(x)
    x = Conv2D(32 * 2 ** i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.load_weights("./model_.h5")

X, y = next(gen(1))

y_pred = model.predict(X)
plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.show()

print("模型的整体准确率： ", evaluate(model,batch_num=200))