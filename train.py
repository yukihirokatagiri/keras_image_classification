import os
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

IMG_WIDTH = 224
IMG_HEIGHT = 224

# augmentation
SHIFT_RANGE = 0.1
ROTATION_RANGE = 10
ZOOM_RANGE = 0.2
H_FLIP = True

CHANNELS = 3
TRAIN_DIR = 'image/train'
VALID_DIR = 'image/valid'
VERBOSE = 1
BATCH_SIZE = 8
EPOCH = 10000
PATIENCE = 50


class Trainer:
    def __init__(self):
        self.callbacks = None
        self.train_gen = None
        self.valid_gen = None
        self.num_train_samples = Trainer.get_file_count(TRAIN_DIR)
        self.num_valid_samples = Trainer.get_file_count(VALID_DIR)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def get_file_count(dir):
        count = 0
        for current_dir, dirs, files in os.walk(dir):
            count += len(files)

        return count

    def setup_callback(self):
        tb = keras.callbacks.TensorBoard(log_dir='./logs',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)
        log = CSVLogger('logs/training.log', append=False)
        es = EarlyStopping('val_acc', patience=PATIENCE)
        lr = ReduceLROnPlateau('val_acc', factor=0.1,
                               patience=int(PATIENCE / 4), VERBOSE=1)
        model_names = 'models/model' + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        mc = ModelCheckpoint(model_names, monitor='val_acc',
                             verbose=1, save_best_only=True)

        self.callbacks = [
            tb,
            mc,
            log,
            es,
            lr
        ]

    def setup_generator(self):
        # prepare data augmentation configuration
        train_idg = ImageDataGenerator(rotation_range=ROTATION_RANGE,
                                       width_shift_range=SHIFT_RANGE,
                                       height_shift_range=SHIFT_RANGE,
                                       zoom_range=ZOOM_RANGE,
                                       horizontal_flip=H_FLIP)
        valid_idg = ImageDataGenerator()

        self.train_gen = train_idg.flow_from_directory(TRAIN_DIR,
                                                       (IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='categorical'
                                                       )
        self.valid_gen = valid_idg.flow_from_directory(VALID_DIR,
                                                       (IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='categorical'
                                                       )

    def train(self):
        self.setup_callback()
        self.setup_generator()

        # create resnet model with random initial weights
        model = ResNet50(weights=None, classes=3)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit_generator(
            self.train_gen,
            steps_per_epoch=self.num_train_samples / BATCH_SIZE,
            validation_data=self.valid_gen,
            validation_steps=self.num_valid_samples / BATCH_SIZE,
            epochs=EPOCH,
            callbacks=self.callbacks,
            verbose=VERBOSE)


if __name__ == "__main__":
    t = Trainer()
    t.train()
