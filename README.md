This repository shows how you train and test keras image classification model, using randomly generated red, green and blue images.

### Setup random images
python image\create_random_image.py

This script creates [red, green, blue] images and store them into 3 different directories, following the keras's ImageDataGenerator directory rule.

After running the script, you'll gain the directory tree as shown below.
```
images
├───test	(blue_0.png, greeen_0.png, red_0.png, blue_1.png, ...)
├───train
│   ├───1	(red_0.png, red_1.png, red_2.png, ...)
│   ├───2	(green_0.png, green_1.png, green_2.png, ...)
│   └───3	(blue_0.png, blue_1.png, blue_2.png, ...)
└───valid
    ├───1	(red_0.png, red_1.png, red_2.png, ...)
    ├───2	(green_0.png, green_1.png, green_2.png, ...)
    └───3	(blue_0.png, blue_1.png, blue_2.png, ...)
```
As you see, train\1 directory represents red images dir.(train\2 is for green images and train\3 is for blue images) The keras's ImageDataGenerator automatically loads train's and valid's subdirectories to train a model.

The train.py will train a Resnet50 model with those imagse.
The test.py will test the model created with train.py with test direcory's images.

### Train
python train.py

### Test
python test.py

### Train with your iamges.
If you want to train a model with your images, replace those single color images. You need to make sure which test directory image corresponds to each class(such as 1,2,3) in the test.py.

