"""
Created on Thu Oct 02

This script collects all data files, shuffle them, split them to training/test datasets
and save into numpy files that are compatabile with Dataset and Dataloader formats. 

- Add relevant stimuli/firing rate ti stimuli_file_names and fr_file_names.
- TEST_DATASET_PERCENTAGE defines which percentage of data should be split for testing.


@author: Ahmed Abdalfatah (@ahmedtarek-)
"""
import numpy as np
import random
import os

TEST_DATASET_PERCENTAGE = 10

# 1. Define stimuli and firing rate files to be used
foldername = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses'
stimuli_file_names = [
    '2022-12-20_15-08_1_stimulus_cb_200.npy',
    '2022-12-20_15-08_2_stimulus_sn_dark.npy',
    '2022-12-20_15-08_3_stimulus_sn_light.npy',

    '2023-03-15_11-05_1_stimulus_cb_200.npy',
    '2023-03-15_11-05_2_stimulus_sn_dark.npy',
    '2023-03-15_11-05_3_stimulus_sn_light.npy',

    '2023-03-15_15-23_1_stimulus_cb_200.npy',
    '2023-03-15_15-23_2_stimulus_sn_dark.npy',
    '2023-03-15_15-23_3_stimulus_sn_light.npy'
]

fr_file_names = [
    '2022-12-20_15-08_1_fr_cb_200.npy',
    '2022-12-20_15-08_2_fr_sn_dark.npy',
    '2022-12-20_15-08_3_fr_sn_light.npy',

    '2023-03-15_11-05_1_fr_cb_200.npy',
    '2023-03-15_11-05_2_fr_sn_dark.npy',
    '2023-03-15_11-05_3_fr_sn_light.npy',

    '2023-03-15_15-23_1_fr_cb_200.npy',
    '2023-03-15_15-23_2_fr_sn_dark.npy',
    '2023-03-15_15-23_3_fr_sn_light.npy'
]

images_dim = np.load(os.path.join(foldername, stimuli_file_names[0])).shape
responses_dim = np.load(os.path.join(foldername, fr_file_names[0])).shape

print("images_dim: ", images_dim)
print("responses_dim: ", responses_dim)

# 2. Load all images and concatenate into two lists
images = np.empty([0, images_dim[1], images_dim[2]])
responses = np.empty([0, responses_dim[1]])
for stimuli_file, fr_file in zip(stimuli_file_names, fr_file_names):
  loaded_images = np.load(os.path.join(foldername, stimuli_file))
  loaded_responses = np.load(os.path.join(foldername, fr_file))
  images = np.concat((images, loaded_images), axis=0)
  responses = np.concat((responses, loaded_responses), axis=0)


# 3. Shuffle the images/responses and take a sample for test data.
data_length = images.shape[0]
shuffled_indices = np.random.permutation(data_length)

images, responses = images[shuffled_indices], responses[shuffled_indices]

# 4. Reshape to get the right shape
images = images.reshape(data_length, 1, images_dim[1], images_dim[2])

# 5. Take 10% of data for testing
# Note that np.split needs to take the index_to_split between brackets, otherwise
#   it divides the array equally to n parts. Check docs.
index_to_split = data_length - int(data_length/TEST_DATASET_PERCENTAGE)
images, test_images = np.split(images, [index_to_split])
responses, test_responses = np.split(responses, [index_to_split])


# 6. Produce images_training, responses_training, images_test, responses_test
save_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Training'
training_images_file_path = os.path.join(save_folder, "training_images_2023-03-15_11-05.npy")
training_responses_file_path = os.path.join(save_folder, "training_responses_2023-03-15_11-05.npy")
test_images_file_path = os.path.join(save_folder, "test_images_2023-03-15_11-05.npy")
test_responses_file_path = os.path.join(save_folder, "test_responses_2023-03-15_11-05.npy")

with open(training_images_file_path, 'wb') as f:
    np.save(f, images)
with open(training_responses_file_path, 'wb') as f:
    np.save(f, responses)

with open(test_images_file_path, 'wb') as f:
    np.save(f, test_images)
with open(test_responses_file_path, 'wb') as f:
    np.save(f, test_responses)
