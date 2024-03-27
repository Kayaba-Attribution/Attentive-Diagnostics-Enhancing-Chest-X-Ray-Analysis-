
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import set_global_policy

# Experimental mixed precision policy
set_global_policy('mixed_float16')

# Constants
SEED = 15
IMG_SIZE = 224
CHANNELS = 3 
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CORES = 6
NUM_CLASSES = 15
ALL_LABELS = ['No Finding', 'Hernia', 'Emphysema', 'Nodule', 'Pneumonia', 'Consolidation', 'Cardiomegaly', 'Effusion', 'Mass', 'Pleural_Thickening', 'Atelectasis', 'Pneumothorax', 'Fibrosis', 'Infiltration', 'Edema']

# Paths
base_path = '/home/kayaba_attribution/Documents/UoL/FINAL_PROJECT/Code/'
current_dir = os.path.dirname(os.path.abspath(__file__))

# Old Implementation
# train_list_path = os.path.join(current_dir, 'train_df.pkl')
# test_list_path = os.path.join(current_dir, 'test_df.pkl')

# Get Images paths
def get_all_image_paths(base_dir):
    """
    Retrieves all image paths within the specified base directory.

    Args:
    - base_dir (str): The base directory containing the image folders.

    Returns:
    - dict: A dictionary mapping image filenames to their full paths.
    """
    # Pattern to match all PNG images in nested 'images' directories
    pattern = os.path.join(base_dir, 'images*', 'images', '*.png')
    
    # Use glob to find all matching image paths
    image_paths = glob.glob(pattern)
    
    # Create a dictionary mapping from basename to full path
    image_paths = {os.path.basename(x): x for x in image_paths}
    
    return image_paths

image_paths = get_all_image_paths(base_path+'nih-chest')

# # Load data
# train_df = pd.read_pickle(train_list_path)
# test_df = pd.read_pickle(test_list_path)

# # Adjust paths
# train_df['Image Index'] = train_df['Image Index'].map(lambda index: image_paths[index])
# test_df['Image Index'] = test_df['Image Index'].map(lambda index: image_paths[index])

# train_data = train_df['Image Index']
# train_labels = train_df[train_df.columns[2:]].values

# X_train, X_val, y_train, y_val = train_test_split(train_data,
#                                      train_labels,
#                                      test_size=0.2, 
#                                      random_state=14)
    
# X_test = test_df['Image Index']
# y_test = test_df[test_df.columns[2:]].values

train_list_path = os.path.join(current_dir, 'train_df_v2.pkl')
test_list_path = os.path.join(current_dir, 'test_df_v2.pkl')
val_list_path = os.path.join(current_dir, 'val_df_v2.pkl')

train_df = pd.read_pickle(train_list_path)
test_df = pd.read_pickle(test_list_path)
val_df = pd.read_pickle(val_list_path)

train_df['Image Index'] = train_df['Image Index'].map(lambda index: image_paths[index])
test_df['Image Index'] = test_df['Image Index'].map(lambda index: image_paths[index])
val_df['Image Index'] = val_df['Image Index'].map(lambda index: image_paths[index])

X_test = test_df['Image Index']
y_test = test_df[test_df.columns[3:]].values

X_train = train_df['Image Index']
y_train = train_df[train_df.columns[3:]].values

X_val = val_df['Image Index']
y_val = val_df[val_df.columns[3:]].values




def parse_function(filename, label):
    """Read, decode, resize, and normalize image, returning image and label."""
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    image_normalized = image_resized / 255.0
    return image_normalized, label

def create_dataset(filenames, labels):
    """Create a TensorFlow dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=NUM_CORES)
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return dataset

# Create TensorFlow datasets
train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val)
test_ds = create_dataset(X_test, y_test)

if __name__ == "__main__":
    print(f'''Train size: X: {len(X_train)}, y: {len(y_train)}
    Validation size: X: {len(X_val)}, y: {len(y_val)}
    Test size: X: {len(X_test)}, y: {len(y_test)}
    Total size: X: {len(X_train) + len(X_val) + len(X_test)}''')
    
    print(f'y_train ex: {y_train[0]}')
    print(f'x_traom ex: {X_train[0]}')
    
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape} | '
      f'X_val: {X_val.shape}, y_val: {y_val.shape} | '
      f'X_test: {X_test.shape}, y_test: {y_test.shape}'
)   

    