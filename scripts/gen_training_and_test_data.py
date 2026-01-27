"""
Created on Thu Oct 02

This script collects all data files, shuffle them, split them to training/test datasets
and save into numpy files that are compatabile with Dataset and Dataloader formats. 

- Add relevant stimuli/firing rate to stimuli_file_names and fr_file_names.
- TEST_DATASET_PERCENTAGE defines which percentage of data should be split for testing.

To Change:
    1. stimuli_file_names
    2. fr_file_names
    3. save_date

@author: Ahmed Abdalfatah (@ahmedtarek-)
"""
import numpy as np
import random
import os

TEST_DATASET_PERCENTAGE = 10

# 1. Define stimuli and firing rate files to be used
foldername = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli-Responses-Delay'
stimuli_file_names = [
    # '2022-12-20_15-08_1_stimulus_cb_200.npy',
    # '2022-12-20_15-08_2_stimulus_sn_dark.npy',
    # '2022-12-20_15-08_3_stimulus_sn_light.npy',

    # '2023-03-15_11-05_1_stimulus_cb_200.npy',
    # '2023-03-15_11-05_2_stimulus_sn_dark.npy',
    # '2023-03-15_11-05_3_stimulus_sn_light.npy',

    # '2023-03-15_15-23_1_stimulus_cb_200.npy',
    '2023-03-15_15-23_2_stimulus_sn_dark.npy',
    '2023-03-15_15-23_3_stimulus_sn_light.npy'
]

fr_file_names = [
    # '2022-12-20_15-08_1_fr_cb_200.npy',
    # '2022-12-20_15-08_2_fr_sn_dark.npy',
    # '2022-12-20_15-08_3_fr_sn_light.npy',

    # '2023-03-15_11-05_1_fr_cb_200.npy',
    # '2023-03-15_11-05_2_fr_sn_dark.npy',
    # '2023-03-15_11-05_3_fr_sn_light.npy',

    # '2023-03-15_15-23_1_fr_cb_200.npy',
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
reconstruct_meta_data = {}
reconstruct_data = []
for i, (stimuli_file, fr_file) in enumerate(zip(stimuli_file_names, fr_file_names)):
    # 2.1 Load the images
    loaded_images = np.load(os.path.join(foldername, stimuli_file))
    loaded_responses = np.load(os.path.join(foldername, fr_file))
    # 2.2 Concat images and responses
    images = np.concat((images, loaded_images), axis=0)
    responses = np.concat((responses, loaded_responses), axis=0)
    # 2.3 Concat data used for reconstruction
    stimulus_length = loaded_images.shape[0]
    reconstruct_meta_data[f"st{i}"] = '_'.join(fr_file.split('_fr_')).replace('.npy', '')
    reconstruct_data += [f"st{i}_{k}" for k in list(range(stimulus_length))]

# 3. Shuffle the images/responses and take a sample for test data.
data_length = images.shape[0]
print("number of data points: ", data_length)

shuffled_indices = np.random.permutation(data_length)
images, responses = images[shuffled_indices], responses[shuffled_indices]
reconstruct_data = np.array(reconstruct_data)[shuffled_indices]

# 4. Reshape to get the right shape
images = images.reshape(data_length, 1, images_dim[1], images_dim[2])

# 5. Take 10% of data for testing
# Note that np.split needs to take the index_to_split between brackets, otherwise
#   it divides the array equally to n parts. Check docs.
index_to_split = data_length - int(data_length/TEST_DATASET_PERCENTAGE)
images, test_images = np.split(images, [index_to_split])
responses, test_responses = np.split(responses, [index_to_split])
reconstruct_data, test_reconstruct_data = np.split(reconstruct_data, [index_to_split])


# 6. Produce images_training, responses_training, images_test, responses_test
save_folder = '/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Training-SparseNoise'
save_date = '2023-03-15_15-23'

training_images_file_path = os.path.join(save_folder, "training_images_{}.npy".format(save_date))
training_responses_file_path = os.path.join(save_folder, "training_responses_{}.npy".format(save_date))

test_images_file_path = os.path.join(save_folder, "test_images_{}.npy".format(save_date))
test_responses_file_path = os.path.join(save_folder, "test_responses_{}.npy".format(save_date))

training_reconstruct_file_path = os.path.join(save_folder, "training_reconstruct_{}.npy".format(save_date))
test_reconstruct_file_path = os.path.join(save_folder, "test_reconstruct_{}.npy".format(save_date))

with open(training_images_file_path, 'wb') as f:
    np.save(f, images)
with open(training_responses_file_path, 'wb') as f:
    np.save(f, responses)

with open(test_images_file_path, 'wb') as f:
    np.save(f, test_images)
with open(test_responses_file_path, 'wb') as f:
    np.save(f, test_responses)

with open(training_reconstruct_file_path, 'wb') as f:
    np.save(f, {'meta': reconstruct_meta_data, 'ids': reconstruct_data}, allow_pickle=True)
with open(test_reconstruct_file_path, 'wb') as f:
    np.save(f, {'meta': reconstruct_meta_data, 'ids': test_reconstruct_data}, allow_pickle=True)

print()
print("Created ", training_images_file_path)
print("Created ", training_responses_file_path)
print("Created ", test_images_file_path)
print("Created ", test_responses_file_path)
print("Created ", training_reconstruct_file_path)
print("Created ", test_reconstruct_file_path)
