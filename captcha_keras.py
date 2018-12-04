# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
import string
from tqdm import tqdm



def gen(batch_size=16, height=80, width=170, n_len=4, n_class=36):
    """
    生成器，默认每次生成16张验证码样本
    :param batch_size: 每次生成验证码个数
    :param height: 图片高度
    :param width: 图片宽度
    :param n_len: 验证码中数字/字母的个数
    :param n_class: 类别数

    :yield:
        :X: 图片数据，of shape (batch_size=16, height=80, width=170, 3)
        :y: one-hot 标签数据, of shape (4, batch_size=16, number of classes=36)
    """
    characters = string.digits + string.ascii_uppercase

    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)

    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    """
    将模型输出的 one-hot 解码成相应的4个数字/字母
    :param y: one-hot
    :return:
    """
    characters = string.digits + string.ascii_uppercase

    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def model(height, width):
    """
    神经网络模型
    :param height: 图片高度
    :param width: 图片宽度
    :return:
    """
    input_tensor = Input((height, width, 3))
    x = input_tensor

    # VGG-16
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

    # 保存可视化结构图
    plot_model(model, to_file="model1.png", show_shapes=True)

    model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
                        nb_worker=3, pickle_safe=True,
                        validation_data=gen(), nb_val_samples=1280)

    # 保存模型权重
    model.save_weights("model_.h5")

    return model


def evaluate(model, batch_num=200):
    """
    计算模型的总体准确率
    :param model:
    :param batch_num:
    :return:
    """
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(list(map(np.array_equal, y_true, y_pred)))

    return batch_acc / batch_num


if __name__ == '__main__':
    # 0-9，A-Z，总共36个类别
    characters = string.digits + string.ascii_uppercase
    # print(characters)

    width, height, n_len, n_class = 170, 80, 4, len(characters)

    model = model(height, width)

    X, y = next(gen(1))

    # 测试一个样本
    y_pred = model.predict(X)
    plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.show()

    print("模型总体准确率：", evaluate(model))
