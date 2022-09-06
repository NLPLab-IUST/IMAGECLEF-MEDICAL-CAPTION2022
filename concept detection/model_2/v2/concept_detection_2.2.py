# -*- coding: utf-8 -*-
"""imports"""
import glob
import re
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
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

"""Read all concepts"""
concept_ids = pd.read_csv(path + "concepts.csv", sep="\t")["concept"].tolist()
print(f"1- read {len(concept_ids)} concepts")

"""Read train concepts csv and apply limit"""
train_df = (
    pd.read_csv(
        path + "ImageCLEFmedCaption_2022_concept_train.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(train_limit)
    .sort_values(by=["ID"])
)
print(f"2- read {len(train_df)} train images")

"""Read valid concepts csv and apply limit"""
valid_df = (
    pd.read_csv(
        path + "ImageCLEFmedCaption_2022_concept_valid.csv",
        delimiter="\t",
        index_col="ID",
    )
    .sample(valid_limit)
    .sort_values(by=["ID"])
)
print(f"3- read {len(valid_df)} valid images")


def load_images_from_list(pixel_x, pixel_y, img_names, folder_path, apply_clahe=True,data_aug=True):
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
        image = cv2.imread(img_path,0)
        if (data_aug):
        # 2
            clahe_image = clahe.apply(image) #apply CLAHE     
            clahe_image = cv2.resize(clahe_image, (pixel_y, pixel_x), interpolation = cv2.INTER_AREA) #resize image
            images_list.append(clahe_image)

            # 3
            hist_image = cv2.equalizeHist(image)
            hist_image = cv2.resize(hist_image, (pixel_y, pixel_x), interpolation = cv2.INTER_AREA) #resize image
            images_list.append(hist_image)

            # 4
            hflip_image = cv2.flip(image, 1)
            hflip_image = cv2.resize(hflip_image, (pixel_y, pixel_x), interpolation = cv2.INTER_AREA) #resize image
            images_list.append(hflip_image)

        else:
            image = clahe.apply(image)

        image = cv2.resize(image, (pixel_y, pixel_x), interpolation = cv2.INTER_AREA) #resize image 
        images_list.append(image)
        

    images_list_np = np.array(images_list)
    return images_list_np


"""Save image names as list"""
train_img_names = train_df.index.values.tolist()
valid_img_names = valid_df.index.values.tolist()

"""Use MultiLabelBinarizer from sklearn to turn concepts to
one-hot vectors for multi-class classification"""
mlb = MultiLabelBinarizer(classes=concept_ids)
mlb.fit([concept_ids])


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for train and valid data"""

    def __init__(self, img_names, batch_size, folder_path, concepts_df, is_train):
        """Initialize data generator params

        Args:
            img_names (list[str]): names of images to read
            batch_size (int): mini batch sizze
            folder_path (str): path of the folder containing data files
            concepts_df (Pandas DataFrame): the dataframe containing labels
            is_train (boolean): train/valid data
        """
        self.img_names = img_names
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.concepts_df = concepts_df
        self.is_train = is_train

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
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path, data_aug=self.is_train))
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(batch_x), name=None)
        batch_y = self.get_labels(batch_names)
        if self.is_train:
            y_rep = np.tile(batch_y,4).reshape(batch_x.shape[0],-1)
            return batch_x, y_rep
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
            self.concepts_df.loc[img_names]["cuis"]
            .apply(lambda x: x.strip(";").split(";"))
            .values
        )


"""Create a data generator for training and validation data"""
gen_train = DataGenerator(
    train_img_names, dataset_batchsize, folder_path=train_path, concepts_df=train_df, is_train=True
)
gen_valid = DataGenerator(
    valid_img_names, dataset_batchsize, folder_path=valid_path, concepts_df=valid_df, is_train=False
)
print("4- created train and valid data generators")

"""Using ResNet50 from keras applications"""
resnet = tf.keras.applications.ResNet50(
    include_top=True,
    weights=resnet_weights_path,
)

"""Fine-tuning models by deleting the last layer and
adding a dense layer as the output layer with sigmoid activation"""
resnet.trainable = False
x = resnet.layers[-2].output
prediction_layer = tf.keras.layers.Dense(len(mlb.classes_), activation="sigmoid")(x)
fine_tuned_resnet = tf.keras.models.Model(
    inputs=resnet.input, outputs=prediction_layer
)


def f1_weighted(true, pred):
    """Calculate weighted F1 score

    Args:
        true (np.ndarray): correct answers
        pred (np.ndarray): model predictions

    Returns:
        float: F1 score
    """
    ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1


"""Compile fine-tune model"""
lr = 0.001
fine_tuned_resnet.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["acc", f1_weighted],
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
        batch_x = load_images_from_list(224, 224, batch_names, self.folder_path, data_aug=False)
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

    def evaluate(truth, th_pred):
        """Evaluate validation result

        Args:
            truth (np.ndarray): validation truth
            th_pred (np.ndarray): validation prediction (after applying threshold)

        Returns:
            float: F1 score
        """
        return f1_weighted(
            tf.convert_to_tensor(truth, "float32"),
            tf.convert_to_tensor(th_pred, "float32"),
        )

    def classify(th_pred, img_names, filename):
        """Save predicted labels

        Args:
            th_pred (np.ndarray): test prediction (after applying threshold)
            img_names (list[str]): list of input images names
            filename (str): filename to save the result to
        """
        valid_cui_predictions = np.where(th_pred, mlb.classes_, "")
        res = np.apply_along_axis(
            lambda x: re.sub(";+", ";", ";".join(x)).strip(";"),
            1,
            valid_cui_predictions,
        )
        print(res[0], res)
        valid_df_pred = pd.DataFrame(
            data={
                "ID": img_names,
                "cuis": res,
            }
        )
        valid_df_pred.to_csv(
            model_path + "/results/" + filename, index=False, header=None, sep="\t"
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
        model_path + f"/results/valid_f1s_e{epoch}.csv", index=False
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
        valid_pred = self.model.predict(gen_valid)

        predict_and_save(valid_pred, test_pred, epoch + 1)

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
