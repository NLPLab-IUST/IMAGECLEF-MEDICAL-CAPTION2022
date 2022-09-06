import glob
from scipy.spatial import distance
import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
from tensorflow.keras import layers
from scipy import spatial
from tensorflow import keras
import cv2
import keras.backend as K
from keras.models import load_model
import random

path = '/IMAGECLEF2022/IMAGECLEF_dataset/'
train_path = path + 'train/'
valid_path = path + 'valid/'

# Create directories for saving model's weights and results
os.makedirs(path + "results", exist_ok=True)
os.makedirs(path + "encoders", exist_ok=True)

# You can change this values based on how many examples you want to test and train
train_count = 1000
valid_count = 300
cross_valid_count = 200

## Read concept file and make desired dicts
file = open(path + 'concepts.csv')
concept_reader = csv.reader(file)

concepts_dict = {}  # Have Concepts and their names
concept_index = {}  # Concepts and their corresponding index

idx = 0
for row in concept_reader:
    row_arr = row[0].split('\t')
    if row_arr[0] == 'concept':
        continue
    concept_index[row_arr[0]] = idx
    idx += 1
    concepts_dict[row_arr[0]] = row_arr[1]
concepts_dict.pop('concept', None)
print(" ** Number of Concepts: " + str(len(concepts_dict.keys())) + " **\n")  ## We have 8374 different concepts!

## Read train csv file (images and their corresponding concepts)
train_df = pd.read_csv(path + 'ImageCLEFmedCaption_2022_concept_train.csv')
img_cuis = list(train_df.loc[:, 'ID\tcuis'])
train_image_concept_dict = {}  ## Contains Image name and Concepts in Image
for ic in img_cuis:
    ic = ic.split('\t')
    cuis = ic[1].split(';')
    train_image_concept_dict[ic[0] + '.jpg'] = cuis

## Read valid csv file (images and their coresponding concepts)
valid_df = pd.read_csv(path + 'ImageCLEFmedCaption_2022_concept_valid.csv')
img_cuis = list(valid_df.loc[:, 'ID\tcuis'])
valid_image_concept_dict = {}  ## Contains Image name and Concepts in Image
for ic in img_cuis:
    ic = ic.split('\t')
    cuis = ic[1].split(';')
    valid_image_concept_dict[ic[0] + '.jpg'] = cuis


## normalizing images (normal_count indicates number of images you want to process)
def load_images(pixel_x, pixel_y, img_set_path, normal_count):
    """normalize and loads image in array format"""
    jpg_images = glob.glob(img_set_path + '/*.jpg')
    random.shuffle(jpg_images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_list = [None] * normal_count
    images_name_list = [None] * normal_count
    counter = 0

    for img in jpg_images:
        if counter >= normal_count:
            break
        image_id = str(img.split('/')[-1])

        image = cv2.imread(img, 0)
        image = clahe.apply(image)  # apply CLAHE

        image = cv2.resize(image, (pixel_y, pixel_x), interpolation=cv2.INTER_AREA)  # resize image

        images_list[counter] = image
        images_name_list[counter] = image_id

        counter += 1

    x_img = np.array(images_list)
    return x_img, images_name_list


img_train, img_train_name = load_images(224, 224, train_path, train_count)
print("successfully load train images")

img_valid, img_valid_name = load_images(224, 224, valid_path, valid_count)
print("successfully load valid images")

input_shape = (224, 224, 3)

concepts_count = len(concepts_dict.keys())

## train set
train_set = []
train_y = []
for idx, img in enumerate(img_train):
    img_name = img_train_name[idx]
    img_arr = img
    img_arr = cv2.merge((img_arr, img_arr, img_arr))
    train_set.append(img_arr)
    cpts = train_image_concept_dict[img_name]
    cpt_arr = np.zeros((concepts_count, 1))
    for cpt in cpts:
        cpt_idx = concept_index[cpt]
        cpt_arr[cpt_idx, 0] = 1
    train_y.append(cpt_arr)
train_set = np.array(train_set)
train_y = np.array(train_y)
train_y = np.reshape(train_y, (train_count, concepts_count))

del img_train

## valid set
valid_set = []
valid_y = []
for idx, img in enumerate(img_valid):
    img_name = img_valid_name[idx]
    img_arr = img
    img_arr = cv2.merge((img_arr, img_arr, img_arr))
    valid_set.append(img_arr)
    cpts = valid_image_concept_dict[img_name]
    cpt_arr = np.zeros((concepts_count, 1))
    for cpt in cpts:
        cpt_idx = concept_index[cpt]
        cpt_arr[cpt_idx, 0] = 1
    valid_y.append(cpt_arr)
valid_set = np.array(valid_set)
valid_y = np.array(valid_y)
valid_y = np.reshape(valid_y, (valid_count, concepts_count))

del img_valid

## data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomWidth(0.2),
        layers.RandomHeight(0.2),
    ]
)

## Transfer Learning
### Resnet50
resnet50_model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
)
resnet50_model.trainable = False
x = resnet50_model.layers[-2].output
prediction_layer = tf.keras.layers.Dense(concepts_count, activation='sigmoid')(x)
fine_tuned_resnet50 = tf.keras.models.Model(inputs=resnet50_model.input, outputs=prediction_layer)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.90,
                                                             staircase=True)

fine_tuned_resnet50.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                            metrics=['acc'])

callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print("Fine tuning Resnet50..........")
history = fine_tuned_resnet50.fit(
    x=train_set,
    y=train_y,
    epochs=5,
    validation_split=0.1,
    verbose=1,
    batch_size=32,
    callbacks=[callback_earlystop]
)

## Save resnet50 weights
print("Save weights for fine-tuned Resnet50....")
fine_tuned_resnet50.save(path + 'encoders/best_resnet50_weights.h5')

del fine_tuned_resnet50

## Densenet201
densenet201_model = tf.keras.applications.DenseNet201(
    include_top=True,
    weights="imagenet",
)
densenet201_model.trainable = False
x = densenet201_model.layers[-2].output
prediction_layer = tf.keras.layers.Dense(concepts_count, activation='sigmoid')(x)
fine_tuned_densenet201 = tf.keras.models.Model(inputs=densenet201_model.input, outputs=prediction_layer)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.90,
                                                             staircase=True)

fine_tuned_densenet201.compile(loss='binary_crossentropy',
                               optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])

callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print("Fine tuning Densenet201..........")
history = fine_tuned_densenet201.fit(
    x=train_set,
    y=train_y,
    epochs=5,
    validation_split=0.1,
    verbose=1,
    batch_size=32,
    callbacks=[callback_earlystop]
)

## Save best weights of Densenet201
print("Save weights for fine-tuned Densenet201....")
fine_tuned_densenet201.save(path + 'encoders/best_densenet201_weights.h5')

del fine_tuned_densenet201

## EfficinetNet B0
EfficientNetB0_model = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
)
EfficientNetB0_model.trainable = False
x = EfficientNetB0_model.layers[-2].output
prediction_layer = tf.keras.layers.Dense(concepts_count, activation='sigmoid')(x)
fine_tuned_EfficientNetB0 = tf.keras.models.Model(inputs=EfficientNetB0_model.input, outputs=prediction_layer)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.90,
                                                             staircase=True)

fine_tuned_EfficientNetB0.compile(loss='binary_crossentropy',
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])

callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print("Fine tuning EfficientNet B0..........")
history = fine_tuned_EfficientNetB0.fit(
    x=train_set,
    y=train_y,
    epochs=5,
    validation_split=0.1,
    verbose=1,
    batch_size=32,
    callbacks=[callback_earlystop]
)

## Save best weights of EfficientNetB0
print("Save weights for fine-tuned EfficientNet B0....")
fine_tuned_EfficientNetB0.save(path + 'encoders/best_EfficientNetB0_weights.h5')

del fine_tuned_EfficientNetB0

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.90,
                                                             staircase=True)
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
number_of_models = 5

## Ensemble Learning : Create 5 different models from each fine tuned encoders and train them on cross validation data
## Resnet50
for i in range(number_of_models):
    print(f"Training model {i + 1} of Resnet50 on Cross-Validation....")
    main_resnet50 = load_model(path + 'encoders/best_resnet50_weights.h5')
    main_resnet50.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                          metrics=['acc'])
    history = main_resnet50.fit(
        x=valid_set[:cross_valid_count],
        y=valid_y[:cross_valid_count],
        epochs=3,
        verbose=1,
        batch_size=32,
        callbacks=[callback_earlystop])
    main_resnet50.save(path + f'encoders/Cross_Resnet_{i + 1}_weights.h5')

## Densenet201
densenet201_encoders = []
for i in range(number_of_models):
    print(f"Training model {i + 1} of densenet201 on Cross-Validation....")
    main_densenet201 = load_model(path + 'encoders/best_densenet201_weights.h5')
    main_densenet201.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                             metrics=['acc'])
    history = main_densenet201.fit(
        x=valid_set[:cross_valid_count],
        y=valid_y[:cross_valid_count],
        epochs=3,
        verbose=1,
        batch_size=32,
        callbacks=[callback_earlystop])
    main_densenet201.save(path + f'encoders/Cross_Densenet201_{i + 1}_weights.h5')

## EfficinetNet B0
for i in range(number_of_models):
    print(f"Training model {i + 1} of EfficinetNet B0 on Cross-Validation....")
    main_efficinetnetb0 = load_model(path + 'encoders/best_EfficientNetB0_weights.h5')
    main_efficinetnetb0.compile(loss='binary_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'])
    history = main_efficinetnetb0.fit(
        x=valid_set[:cross_valid_count],
        y=valid_y[:cross_valid_count],
        epochs=3,
        verbose=1,
        batch_size=32,
        callbacks=[callback_earlystop])
    main_efficinetnetb0.save(path + f'encoders/Cross_EfficientNet_{i + 1}_weights.h5')


## We use remains of validation data for testing model performance
test_x = valid_set[cross_valid_count:]
test_y = valid_y[cross_valid_count:]
similarity = []


## compute Cosine similarity
def cosine_similarity(test_embeddings, train_embeddings):
    test_count = test_embeddings.shape[0]
    train_idx = [None] * test_count
    for i in range(test_count):
        best_sim = -10000
        test_image = test_embeddings[i]
        for j in range(train_count):
            train_image = train_embeddings[j]
            cosine_sim = 1 - spatial.distance.cosine(test_image, train_image)
            if cosine_sim > best_sim:
                best_sim = cosine_sim
                train_idx[i] = j
    return train_idx


# Predicting test set concepts
## Resnet50
for i in range(number_of_models):
    print(f"Loading model {i + 1} of Resnet50 on Cross-Validation....")
    main_resnet50 = load_model(path + f'encoders/Cross_Resnet_{i + 1}_weights.h5')
    # predict and sim
    intermediate_model = keras.models.Model(inputs=main_resnet50.layers[0].input,
                                            outputs=[l.output for l in main_resnet50.layers[-2:]])
    embedding = intermediate_model.predict(train_set)
    embedding_resnet50 = embedding[0]

    ## predict
    embedding_test = intermediate_model.predict(test_x)[0]
    ## compute cosine similarity
    sim = cosine_similarity(embedding_test, embedding_resnet50)
    similarity.append(sim)

del main_resnet50
del embedding_resnet50
del embedding_test
del embedding

### Densenet201
for i in range(number_of_models):
    print(f"Loading model {i + 1} of densenet201 on Cross-Validation....")
    main_densenet201 = load_model(path + f'encoders/Cross_Densenet201_{i + 1}_weights.h5')

    print(f"Start to get Embedding of train images from Densenet{i + 1} and similarity .....")
    intermediate_model = keras.models.Model(inputs=main_densenet201.layers[0].input,
                                            outputs=[l.output for l in main_densenet201.layers[-2:]])
    embedding = intermediate_model.predict(train_set)
    embedding_densenet201 = embedding[0]

    ## predict
    embedding_test = intermediate_model.predict(test_x)[0]
    ## compute cosine similarity
    sim = cosine_similarity(embedding_test, embedding_densenet201)
    similarity.append(sim)

del main_densenet201
del embedding_densenet201
del embedding_test
del embedding

### EfficinetNet B0
efficinetnetb0_encoders = []
for i in range(number_of_models):
    print(f"Loading model {i + 1} of EfficinetNet B0 on Cross-Validation....")
    main_efficinetnetb0 = load_model(path + f'encoders/Cross_EfficientNet_{i + 1}_weights.h5')
    print(f"Start to get Embedding of train images from EfficinetNet{i + 1} and similarity.....")
    intermediate_model = keras.models.Model(inputs=main_efficinetnetb0.layers[0].input,
                                            outputs=[l.output for l in main_efficinetnetb0.layers[-2:]])
    embedding = intermediate_model.predict(train_set)
    embedding_efficientnetb0 = embedding[0]

    ## predict
    embedding_test = intermediate_model.predict(test_x)[0]
    ## compute cosine similarity
    sim = cosine_similarity(embedding_test, embedding_efficientnetb0)
    similarity.append(sim)

del main_efficinetnetb0
del embedding_efficientnetb0
del embedding_test
del embedding

del train_set

## Majority Voting for selecting most frequent concepts
majority_count = 8
predicted_test_y = []
for i in range(test_x.shape[0]):
    concept_sim = [0] * concepts_count
    for conc in range(concepts_count):
        concept_maj = 0
        for j in range(len(similarity)):
            similar = train_y[similarity[j][i]]
            if similar[conc] == 1:
                concept_maj += 1
        if concept_maj >= majority_count:
            concept_sim[conc] = 1
    predicted_test_y.append(concept_sim)

## Save results
idx_concept = {v: k for k, v in concept_index.items()}
conc_str = []
for y_pre in predicted_test_y:
    concept_present = [idx_concept[idx] for idx, c in enumerate(y_pre) if c == 1]
    conc_str.append(';'.join(concept_present))

valid_df_result = ["ID\tcuis"]
for i in range(len(conc_str)):
    result = img_valid_name[i] + "\t" + conc_str[i]
    valid_df_result.append(result)
    valid_df_pred = pd.DataFrame(data=valid_df_result)
    valid_df_pred.to_csv(path + "results/model_1_concepts_results.csv", index=False, header=None)

## compute F1 score
gt_file = path + "ImageCLEFmedCaption_2022_concept_valid.csv"
candidate_file = path + 'results/model_1_concepts_results.csv'

import sys, argparse, string
import csv
import warnings

from sklearn.metrics import f1_score

min_concepts = sys.maxsize
max_concepts = 0
total_concepts = 0
concepts_distrib = {}


# Read a Tab-separated ImageID - Caption pair file
def readfile(path):
    try:
        pairs = {}
        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                # We have an ID and a set of concepts (possibly empty)
                if len(row) == 2:
                    pairs[row[0]] = row[1]
                # We only have an ID
                elif len(row) == 1:
                    pairs[row[0]] = ''
                else:
                    print('File format is wrong, please check your run file')
                    exit(1)

        return pairs
    except FileNotFoundError:
        print('File "' + path + '" not found! Please check the path!')
        exit(1)


# Print 1-level key-value dictionary, sorted (with numeric key)
def print_dict_sorted_num(obj):
    keylist = [int(x) for x in list(obj.keys())]
    keylist.sort()
    for key in keylist:
        print(key, ':', obj[str(key)])


candidate_pairs = readfile(candidate_file)
gt_pairs = readfile(gt_file)

# Define max score and current score
max_score = len(candidate_pairs)
current_score = 0

# Check there are the same number of pairs between candidate and ground truth
# if len(candidate_pairs) != len(gt_pairs):
#     print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
#     exit(1)

# Evaluate each candidate concept list against the ground truth
print('Processing concept sets...\n********************************')

i = 0
for image_key in candidate_pairs:
    if (image_key not in candidate_pairs):
        continue
    # Get candidate and GT concepts
    candidate_concepts = candidate_pairs[image_key].upper()
    gt_concepts = gt_pairs[image_key.split('.')[0]].upper()

    # Split concept string into concept array
    # Manage empty concept lists
    if gt_concepts.strip() == '':
        gt_concepts = []
    else:
        gt_concepts = gt_concepts.split(';')

    if candidate_concepts.strip() == '':
        candidate_concepts = []
    else:
        candidate_concepts = candidate_concepts.split(';')

    # Manage empty GT concepts (ignore in evaluation)
    if len(gt_concepts) == 0:
        max_score -= 1
    # Normal evaluation
    else:
        # Concepts stats
        total_concepts += len(gt_concepts)

        # Global set of concepts
        all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

        # Calculate F1 score for the current concepts
        y_true = [int(concept in gt_concepts) for concept in all_concepts]
        y_pred = [int(concept in candidate_concepts) for concept in all_concepts]

        f1score = f1_score(y_true, y_pred, average='binary')

        # Increase calculated score
        current_score += f1score

    # Concepts stats
    nb_concepts = str(len(gt_concepts))
    if nb_concepts not in concepts_distrib:
        concepts_distrib[nb_concepts] = 1
    else:
        concepts_distrib[nb_concepts] += 1

    if len(gt_concepts) > max_concepts:
        max_concepts = len(gt_concepts)

    if len(gt_concepts) < min_concepts:
        min_concepts = len(gt_concepts)

    # Progress display
    i += 1
    if i % 1000 == 0:
        print(i, '/', len(gt_pairs), ' concept sets processed...')

# Print evaluation result
print('Final result\n********************************')
print('Obtained score :', current_score, '/', max_score)
print('Mean score over all concept sets :', current_score / max_score)
