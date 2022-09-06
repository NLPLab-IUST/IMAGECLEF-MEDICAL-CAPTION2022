# -*- coding: utf-8 -*-
"""imports"""
import os
import csv
import h5py
import math
import cv2
import spacy, numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial import distance
import pandas as pd
import sys, argparse, string
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.preprocessing import MultiLabelBinarizer


"""files paths"""
path = "/content/drive/MyDrive/ImageCLEF/"
train_path = path + "train"
valid_path = path + "valid"
test_path = path + "test"
"""create necessary folders"""
model_path = path + "yasaman"
os.makedirs(model_path + "/results", exist_ok=True)
os.makedirs(model_path + "/weights", exist_ok=True)
resnet_weights_path = model_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5"

"""define limit for data loading"""
train_limit = 1000
valid_limit = 300
test_limit = 300

dataset_batchsize = 64

"""Loading csvs"""

"""Read train concepts csv and apply limit"""
train_df = (
    pd.read_csv(
        model_path + "/ImageCLEFmedCaption_2022_caption_train_prepro.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(train_limit)
    .sort_values(by=["ID"])
)
print(f"1- read {len(train_df)} train images")

"""Read valid concepts csv and apply limit"""
valid_df = (
    pd.read_csv(
        model_path + "/ImageCLEFmedCaption_2022_caption_valid_prepro.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(valid_limit)
    .sort_values(by=["ID"])
)
print(f"2- read {len(valid_df)} valid images")


def load_images_from_list(pixel_x, pixel_y, img_names, folder_path, apply_clahe=True):
    """Load images with img_names from folder_path and apply needed preprocessing

    Args:
        pixel_x (int): resize width to
        pixel_y (int): resize height to
        img_names (list[str]): names of images to read from folder
        folder_path (str): folder path containing images
        apply_clahe (bool, optional): whether clahe should be applied or not. Defaults to True.

    Returns:
        np.ndarray: preprocessed images in size (len(img_names), pixel_x, pixel_y)
    """
    """normalize and loads image in array format"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_list = []

    for img in img_names:
        img_path = folder_path + "/" + img + ".jpg"
        image = cv2.imread(img_path, 0)
        if apply_clahe:
            image = clahe.apply(image)  # apply CLAHE
        image = cv2.resize(
            image, (pixel_y, pixel_x), interpolation=cv2.INTER_AREA
        )  # resize image
        images_list.append(image)

    images_list_np = np.array(images_list)
    return images_list_np


"""Save image names as list"""
train_img_names = train_df.index.values.tolist()
valid_img_names = valid_df.index.values.tolist()

"""Read and create vocabulary"""
vocab_stat = pd.read_csv(
    model_path + "/caption_tokens_occurences.csv", index_col="token"
)
vocab_size = len(vocab_stat)

"""Use MultiLabelBinarizer from sklearn to turn vocab words to
one-hot vectors for multi-class classification"""
vocab_words = vocab_stat.index.values.tolist()
mlb = MultiLabelBinarizer(classes=vocab_words)
mlb.fit([vocab_words])


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for train and valid data"""

    def __init__(self, img_names, batch_size, folder_path, concepts_df):
        """Initialize data generator params

        Args:
            img_names (list[str]): names of images to read
            batch_size (int): mini batch sizze
            folder_path (str): path of the folder containing data files
            concepts_df (Pandas DataFrame): the dataframe containing labels
        """
        self.img_names = img_names
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.concepts_df = concepts_df

    def __len__(self):
        return (np.ceil(len(self.img_names) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Return minibatch of sequence idx

        Args:
            idx (int): sequence of the mini batch

        Returns:
            np.ndarray, np.ndarray: model input, model output
        """
        batch_names = self.img_names[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path)
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(batch_x), name=None)
        batch_y = self.get_labels(batch_names)
        return batch_x, batch_y

    def get_labels(self, img_names):
        """Return labels corresponding to img_names in one-hot vectors

        Args:
            img_names (list[str]): image names to return the labels for

        Returns:
            np.ndarray: one-hot vectors of labels
        """
        if self.concepts_df is None:
            return np.zeros((len(img_names), len(mlb.classes_)))
        return mlb.transform(
            self.concepts_df.loc[img_names]["caption"]
            .apply(lambda x: x.split(";"))
            .values
        )


"""Create a data generator for training and validation data"""
gen_train = DataGenerator(
    train_img_names, dataset_batchsize, folder_path=train_path, concepts_df=train_df
)
gen_valid = DataGenerator(
    valid_img_names, dataset_batchsize, folder_path=valid_path, concepts_df=valid_df
)
print("4- created train and valid data generators")

"""Using ResNet50 from keras applications"""
resnet = tf.keras.applications.ResNet50(
    include_top=True,
    weights=resnet_weights_path,
)

"""Fine-tuning models by deleting the last layer and
adding a dense layer as the output layer with sigmoid activation"""
x = resnet.layers[-2].output  # remove last layer
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.PReLU()(x)
x = tf.keras.layers.Dense(len(mlb.classes_), activation="sigmoid")(x)  # add layer
fine_tuned_resnet = tf.keras.models.Model(inputs=resnet.input, outputs=x)

"""Compile fine-tune model"""
lr = 5e-4
fine_tuned_resnet.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["acc"],
)

"""Early stopping checkpoint"""


class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_names, batch_size, folder_path):
        """Initialize data generator params

        Args:
            img_names (list[str]): names of images to read
            batch_size (int): mini batch sizze
            folder_path (str): path of the folder containing data files
        """
        self.img_names = img_names
        self.batch_size = batch_size
        self.folder_path = folder_path

    def __len__(self):
        return (np.ceil(len(self.img_names) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Return minibatch of sequence idx

        Args:
            idx (int): sequence of the mini batch

        Returns:
            np.ndarray, np.ndarray: model input, model output
        """
        batch_names = self.img_names[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path)
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(batch_x), name=None)
        return batch_x


"""Read test images from the path with glob"""
test_img_names = [
    os.path.splitext(os.path.basename(x))[0] for x in glob.glob(test_path + "/*.jpg")
][:test_limit]
print(f"found {len(test_img_names)} test images")
gen_test = TestDataGenerator(test_img_names, dataset_batchsize, folder_path=test_path)

"""Train"""


def predict_and_save(valid_res, test_res, epoch):
    """Evaluate validation result, save test result

    Args:
        valid_res (np.ndarray): validation result predicted by model
        test_res (np.ndarray): test result predicted by model
        epoch (int): epoch
    """

    Ns = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    for N in Ns:
        """Find N words with highest probability"""
        valid_words_idx = np.argsort(-valid_res, axis=1)[:, :N]
        valid_words = np.take(mlb.classes_, valid_words_idx)
        valid_captions_sorted = []
        """Sort based on the training set statistics"""
        for v in valid_words:
            v_stats = [int(vocab_stat.loc[v_token]["occurences"][0]) for v_token in v]
            valid_captions_sorted.append(v[np.argsort(v_stats)])

        pd.DataFrame(
            data={
                "ID": valid_img_names,
                "caption": [" ".join(tokens) for tokens in valid_captions_sorted],
            }
        ).to_csv(
            model_path + "/results" + f"/valid_res_N_{N}_e{epoch}_sorted.csv",
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            data={
                "ID": valid_img_names,
                "caption": [" ".join(tokens) for tokens in valid_words],
            }
        ).to_csv(
            model_path + "/results" + f"/valid_res_N_{N}_e{epoch}_normal.csv",
            index=False,
            sep="\t",
        )

    for N in Ns:
        valid_words_idx = np.argsort(-(test_res), axis=1)[:, :N]
        valid_words = np.take(mlb.classes_, valid_words_idx)
        valid_captions_sorted = []
        for v in valid_words:
            v_stats = [int(vocab_stat.loc[v_token]["occurences"][0]) for v_token in v]
            valid_captions_sorted.append(v[np.argsort(v_stats)])
        pd.DataFrame(
            data={
                "ID": test_img_names,
                "caption": [" ".join(tokens) for tokens in valid_captions_sorted],
            }
        ).to_csv(
            model_path + "/results" + f"/test_res_N_{N}_e{epoch}_sorted.csv",
            index=False,
            sep="\t",
        )
        pd.DataFrame(
            data={
                "ID": test_img_names,
                "caption": [" ".join(tokens) for tokens in valid_words],
            }
        ).to_csv(
            model_path + "/results" + f"/test_res_N_{N}_e{epoch}_normal.csv",
            index=False,
            sep="\t",
        )


class SaveResultsCallback(tf.keras.callbacks.Callback):
    """Custom callback"""

    def on_epoch_end(self, epoch, logs={}):
        """Save results after the first epoch and then, after each 4 epochs
            and save models weights
        Args:
            epoch (int): epoch
        """
        if (epoch + 1) % 4 != 0 and (epoch + 1) != 1:
            return
        test_pred = self.model.predict(gen_test)
        valid_res = self.model.predict(gen_valid)

        predict_and_save(valid_res, test_pred, epoch + 1)

        self.model.save_weights(
            model_path + f"/weights/e{epoch+1}_weights.h5", overwrite=True
        )


"""Train the model"""
history = fine_tuned_resnet.fit(
    x=gen_train,
    epochs=50,
    validation_data=gen_valid,
    verbose=1,
    callbacks=[SaveResultsCallback()],
)
