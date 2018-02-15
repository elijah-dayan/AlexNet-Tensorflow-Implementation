import numpy as np
import pickle
import os
import sys
import urllib.request
import tarfile
import zipfile





auto_guess = -1
null = 0
.
size_of_image = 32


channels_of_image = 3

flat_size_of_image = size_of_image * size_of_image * channels_of_image


classes_of_image = 10


files_training_set = 5


file_each_count = 10000

images_training_set = files_training_set * file_each_count




homework_data_path = "homework2_3_&_homework3_1_data/cifar10"

u_toronto_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def fetch_path(assigned_name=""):
   

    return os.path.join(homework_data_path, "data_batches/", assigned_name)


def unpickle(assigned_name):
  

    real_path = fetch_path(assigned_name)

    print("Contents...." + real_path)

    with open(real_path, mode='rb') as contents:
      
        loaded = pickle.load(contents, encoding='bytes')

    return loaded


def load_data_wrapper(data_unprocessed):
  


    floating_point_data = np.array(data_unprocessed, dtype=float) / 255.0

   
    reshaped_data = floating_point_data.reshape([auto_guess, channels_of_image, size_of_image, size_of_image])

    
    reshaped_data = reshaped_data.transpose([0, 2, 3, 1])

    return reshaped_data


def load_data(assigned_name):
  data-file.
    contents = unpickle(assigned_name)

 
    images_unprocessed = contents[b'data']


    image_class_number = np.array(contents[b'labels'])


    wrapped_images = load_data_wrapper(images_unprocessed)

    return wrapped_images, image_class_number

def download(data_url, download_destination):
  
    assigned_name = data_url.split('/')[auto_guess]
    destination_path = os.path.join(download_destination, assigned_name)

  
    if not os.path.exists(destination_path):
        t.
        if not os.path.exists(download_destination):
            os.makedirs(download_destination)

      
        destination_path, _ = urllib.request.urlretrieve(url=data_url,
                                                  filename=destination_path,
                                                  reporthook=None)

        print()
        print("Download Job Done.")

        if destination_path.endswith(".zip"):
           
            zipfile.ZipFile(file=destination_path, mode="r").extractall(download_destination)
        elif destination_path.endswith((".tar.gz", ".tgz")):
        
            tarfile.open(name=destination_path, mode="r:gz").extractall(download_destination)

        print("Job All Done.")
    else:
        print("Error: Job Was Done Previously")


def download_wrapper():
  

    download(data_url=u_toronto_url, download_destination=homework_data_path)

def load_data_classes():


    # Load the class-names from the pickled file.
    data_unprocessed = unpickle(assigned_name="batches.meta")[b'label_names']

    # Convert from binary strings.
    assigned_names = [tmp.decode('utf-8') for tmp in data_unprocessed]

    return assigned_names



def data_vectors(cls_counts, classes=None):

    if classes is None:
        classes = np.max(cls_counts) + 1

    return np.eye(classes, dtype=float)[cls_counts]  



def load_data_training():
  

   
    dataset = np.zeros(shape=[images_training_set, size_of_image, size_of_image, channels_of_image], dtype=float)
    dataset_classes = np.zeros(shape=[images_training_set], dtype=int)

   
    initial = null

  
    for count in range(files_training_set):
       
        data_batch, class_batch = load_data(assigned_name="batch_" + str(count + 1))

        
        images_count = len(data_batch)

        
        final = initial + images_count

       
        dataset[initial:final, :] = data_batch

       
        dataset_classes[initial:final] = class_batch

        initial = final

    return dataset, dataset_classes, data_vectors(cls_counts=dataset_classes, classes=classes_of_image)



def load_data_testing():
   

    dataset, classes = load_data(assigned_name="testing_batch_data")

    return dataset, classes, data_vectors(cls_counts=classes, classes=classes_of_image)
                
