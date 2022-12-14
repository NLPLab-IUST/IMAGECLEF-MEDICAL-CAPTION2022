# -*- coding: utf-8 -*-
"""ConceptDetection1-data generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y-rL_yAJjbcRIPmenAcNHo991cU_cswX

# Getting Files

imports
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras import backend as K


"""addresses of files"""

path = "/home/MohammadMahdiJavid/mvqa/mvqa/IMAGECLEF/"
train_path = path + "IMAGECLEF_dataset/train/train"
valid_path = path + "IMAGECLEF_dataset/valid/valid/valid"
test_path = path + "IMAGECLEF_dataset/test/test/test"

"""create necessary folders"""
ysmn_path = path + "yasaman"
os.makedirs(ysmn_path + "/results", exist_ok=True)
os.makedirs(ysmn_path + "/weights", exist_ok=True)

train_limit = 83275
valid_limit = 7645
test_limit = 7601

dataset_batchsize = 64
densenet_weights_path = (
    path + "yasaman-caption/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
)

"""# Preprocessing

## Read All Concepts and their Names
"""

concept_ids = pd.read_csv(path + "IMAGECLEF_dataset/concepts.csv", sep="\t")[
    "concept"
].tolist()
print(f"1- read {len(concept_ids)} concepts")


train_df = (
    pd.read_csv(
        path + "IMAGECLEF_dataset/ImageCLEFmedCaption_2022_concept_train.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(train_limit)
    .sort_values(by=["ID"])
)
print(f"2- read {len(train_df)} train images")

valid_df = (
    pd.read_csv(
        path + "IMAGECLEF_dataset/ImageCLEFmedCaption_2022_concept_valid.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(valid_limit)
    .sort_values(by=["ID"])
)
print(f"3- read {len(valid_df)} valid images")

"""Reading existing images in train_df_limited"""


def load_images_from_list(pixel_x, pixel_y, img_names, folder_path, apply_clahe=True):
    """normalize and loads image in array format"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_list = []

    for img in img_names:
        img_path = folder_path + "/" + img + ".jpg"
        image = cv2.imread(img_path, 0)
        if apply_clahe:
            image = clahe.apply(image)  # apply CLAHE
        # image = cv2.equalizeHist(image) #apply Histograms Equalization
        image = cv2.resize(
            image, (pixel_y, pixel_x), interpolation=cv2.INTER_AREA
        )  # resize image
        images_list.append(image)

    images_list_np = np.array(images_list)
    return images_list_np


train_img_names = train_df.index.values.tolist()
valid_img_names = valid_df.index.values.tolist()

# Use a multilabelbinarizer to transform the concepts into a binary format for training
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=concept_ids)
mlb.fit([concept_ids])


class My_Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, batch_size, folder_path, concepts_df):
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.concepts_df = concepts_df

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(
            np.int
        )

    def __getitem__(self, idx):
        batch_names = self.image_filenames[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path)
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(batch_x), name=None)
        batch_y = self.get_labels(batch_names)
        return batch_x, batch_y

    def get_labels(self, img_names):
        if self.concepts_df is None:
            return np.zeros((len(img_names), len(mlb.classes_)))
        return mlb.transform(
            self.concepts_df.loc[img_names]["cuis"]
            .apply(lambda x: x.strip(";").split(";"))
            .values
        )

        return batch_x, batch_y


gen_train = My_Custom_Generator(
    train_img_names, dataset_batchsize, folder_path=train_path, concepts_df=train_df
)
gen_valid = My_Custom_Generator(
    valid_img_names, dataset_batchsize, folder_path=valid_path, concepts_df=valid_df
)
print("4- created train and valid data generators")

"""# Using DenseNet 121"""

densenet = tf.keras.applications.ResNet50(
    include_top=True,
    weights=densenet_weights_path,
)

"""## Fine-tuned Model

### Training

Deleting and adding layers
"""

densenet.trainable = False
x = densenet.layers[-2].output
x = tf.keras.layers.Dropout(0.5)(x)
prediction_layer = tf.keras.layers.Dense(len(mlb.classes_), activation="sigmoid")(x)
fine_tuned_densenet = tf.keras.models.Model(
    inputs=densenet.input, outputs=prediction_layer
)
# fine_tuned_densenet.load_weights(ysmn_path + "/weights/e48_weights.h5")
"""Learning rate decay and compiling model, adding f1 metric"""


def f1_weighted(true, pred):
    ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1


initial_learning_rate = 0.001
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.90, staircase=True)
fine_tuned_densenet.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=["acc", f1_weighted],
)

"""Early stopping checkpoint"""


class Test_Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, batch_size, folder_path):
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.folder_path = folder_path

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_names = self.image_filenames[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path)
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(batch_x), name=None)
        return batch_x


import glob

test_img_names = [
    os.path.splitext(os.path.basename(x))[0] for x in glob.glob(test_path + "/*.jpg")
][:test_limit]
print(f"found {len(test_img_names)} test images")
gen_test = Test_Custom_Generator(
    test_img_names, dataset_batchsize, folder_path=test_path
)

"""Train"""

import re


def predict_and_save(valid_res, test_res, epoch):
    def evaluate(truth, th_pred):
        return f1_weighted(
            tf.convert_to_tensor(truth, "float32"),
            tf.convert_to_tensor(th_pred, "float32"),
        )

    def classify(th_pred, img_names, filename):
        valid_cui_predictions = np.where(
            th_pred, mlb.classes_, ""
        )  # np.take(mlb.classes_, th_pred)
        # print(valid_cui_predictions.shape, th_pred.shape, re.sub(';+', ';', ";".join(valid_cui_predictions[0]))),
        res = np.apply_along_axis(
            lambda x: re.sub(";+", ";", ";".join(x)).strip(";"),
            1,
            valid_cui_predictions,
        )
        # print(res[0], res)
        valid_df_pred = pd.DataFrame(
            data={
                "ID": img_names,
                "cuis": res,
            }
        )
        valid_df_pred.to_csv(
            ysmn_path + "/results/" + filename, index=False, header=None, sep="\t"
        )

    threshold = [
        0.02,
        0.1,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.3,
        0.4,
    ]
    valid_truth = mlb.transform(
        valid_df["cuis"].apply(lambda x: x.strip(";").split(";")).values
    )
    valid_f1s = []
    for th in threshold:
        th_pred = (valid_res > th).astype(int)
        print(f"Using threshold {th}...")
        valid_f1 = evaluate(valid_truth, th_pred).numpy()
        valid_f1s.append(valid_f1)
        print("valid f1: ", valid_f1)
        th_pred_test = (test_res > th).astype(int)
        classify(
            th_pred_test, test_img_names, f"e-{epoch}-threshold-{th}-classification.csv"
        )

    pd.DataFrame({"threshold": threshold, "f1": valid_f1s}).to_csv(
        ysmn_path + f"/results/valid_f1s_e{epoch}.csv", index=False
    )


class SaveResultsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 4 != 0 and (epoch + 1) != 1:
            return
        test_pred = self.model.predict(gen_test)
        valid_pred = self.model.predict(gen_valid)

        predict_and_save(valid_pred, test_pred, epoch + 1)

        self.model.save_weights(
            ysmn_path + f"/weights/e{epoch+1}_weights.h5", overwrite=True
        )


history = fine_tuned_densenet.fit(
    x=gen_train,
    epochs=50,
    validation_data=gen_valid,
    verbose=1,
    callbacks=[SaveResultsCallback()],
)

"""Save weights at the end"""
