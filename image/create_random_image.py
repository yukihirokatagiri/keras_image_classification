import os
import cv2
import random
import numpy as np

script_dir = os.path.abspath(os.path.dirname(__file__))
test_dir = os.path.join(script_dir, "test")
train_dir_base = os.path.join(script_dir, "train")
valid_dir_base = os.path.join(script_dir, "valid")

labels = {
    "red": 1,
    "green": 2,
    "blue": 3,
}

for i in range(100):
    for label in labels.keys():
        train_dir = os.path.join(train_dir_base, str(labels[label]))
        valid_dir = os.path.join(valid_dir_base, str(labels[label]))

        for target_dir in [test_dir, train_dir, valid_dir]:
            image = np.zeros((224, 224, 3), np.uint8)

            rand = random.randrange(256)
            if label == "red":
                color = (0, 0, rand)  # Note this tupple is b-g-r order
            elif label == "green":
                color = (0, rand, 0)  # Note this tupple is b-g-r order
            elif label == "blue":
                color = (rand, 0, 0)  # Note this tupple is b-g-r order

            image[:] = color

            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, f"{label}_{i}.png")
            cv2.imwrite(path, image)
