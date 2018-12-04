# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import random
import string
from utils import mkdir


if __name__ == '__main__':
    characters = string.digits + string.ascii_uppercase
    width, height, n_len, n_class = 170, 80, 4, len(characters)
    generator = ImageCaptcha(width=width, height=height)

    mkdir("./data_examples")

    for i in range(16):

        random_str = ''.join([random.choice(characters) for j in range(4)])
        img = generator.generate_image(random_str)

        plt.imshow(img)
        plt.title(random_str)
        # plt.show()
        print("Saving {}.jpg".format(random_str))
        plt.savefig("./data_examples/{}.jpg".format(random_str))