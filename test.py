import os
import time

import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar

from keras.applications.resnet50 import ResNet50
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='predict images with your model.')
    parser.add_argument('model', help='model file path')
    args = parser.parse_args()

    model = ResNet50(weights=None, classes=3)
    model.load_weights(args.model, by_name=True)

    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    out = open('result.txt', 'a')

    test_dir = "image/test"
    paths = []
    for current_dir, dirs, files in os.walk(test_dir):
        for file in files:
            paths.append(os.path.join(test_dir, file))
    num_samples = len(paths)

    correct_count = 0
    wrong_count = 0

    for idx, path in enumerate(paths):
        bgr_img = cv.imread(path)
        bgr_img = cv.resize(bgr_img, (224, 224))
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(bgr_img)
        prob = np.max(preds)
        class_id = np.argmax(preds) + 1

        correct_ans = 1
        if "racoon_dog" in path:
            correct_ans = 2
        elif "red_panda" in path:
            correct_ans = 3

        if class_id == correct_ans:
            correct_count += 1
        else:
            wrong_count += 1

        out.write('{}\n'.format(str(class_id)))
        pb.print_progress_bar((idx + 1) * 100 / num_samples)

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))
    print('acc : {}'.format(str(correct_count / (correct_count + wrong_count))))

    out.close()
    K.clear_session()
