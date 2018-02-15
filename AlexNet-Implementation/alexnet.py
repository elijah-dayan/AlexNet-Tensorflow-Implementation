

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import cifar10_load
import prettytensor as pt


batches = 256
batch_training = 64
crop_sz = 24
cifar10_load.download_wrapper()
class_names = cifar10_load.load_data_classes()

im_tr, cl_tr, lb_tr = cifar10.load_training_data()
im_tt, cl_tt, lb_tt = cifar10.load_test_data()


from cifar10_load import size_of_image, channels_of_image, classes_of_image



 

input_im = tf.placeholder(tf.float32, shape=[None, size_of_image, size_of_image, channels_of_image], name='input_im')

im_true_lb = tf.placeholder(tf.float32, shape=[None, classes_of_image], name='im_true_lb')

im_true_cl = tf.argmax(im_true_lb, axis=1)

def data_prepro(im, tr):
  
    
    if tr:
        
        im = tf.random_crop(im, size=[crop_sz, crop_sz, channels_of_image])

      
        im = tf.image.random_flip_left_right(im)
        
      
        im = tf.image.random_hue(image, max_delta=0.03)
        im = tf.image.random_contrast(image, lower=0.4, upper=0.9)
        im = tf.image.random_brightness(image, max_delta=0.3)
        im = tf.image.random_saturation(image, lower=0.3, upper=1.5)

    else:
     
        im = tf.image.resize_image_with_crop_or_pad(im,
                                                       target_height=crop_sz,
                                                       target_width=crop_sz)

    return im


def wrapper_prepro(ims, tr):
  
    ims = tf.map_fn(lambda im: data_prepro(im, tr), ims)

    return ims


inflate_data = wrap_prepro(im1=input_im, tr=True)



def build_alexnet(ims, tr):
    
    x_pretty = pt.wrap(ims)

   
    if tr:
        state = pt.Phase.train
    else:
        state = pt.Phase.infer


    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=state):
        prediction, cost = x_pretty.\
            conv2d(kernel=5, depth=64, name='conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='conv2', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
	    conv2d(kernel=5, depth=64, name='conv3').\
            conv2d(kernel=5, depth=64, name='conv4').\
            conv2d(kernel=5, depth=64, name='conv5').\
            flatten().\
            fully_connected(size=256, name='fc1').\
            fully_connected(size=128, name='fc2').\
	    fully_connected(size=64, name='fc3').\
            softmax_classifier(num_classes=classes_of_image, labels=im_true_lb)

    return prediction, cost


def build_net(tr):
    
    with tf.variable_scope('net', reuse=not tr):
       
        ims = input_im

     
        ims = wrapper_prepro(ims=imas, tr=tr)

       
        prediction, cost = build_alexnet(ims=ims, tr=tr)

    return prediction, cost



global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

cost = build_net(tr=True)[1]

optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, global_step=global_step)

prediction = build_net(tr=False)[0]

prediction_cl = tf.argmax(prediction, axis=1)

accurate_prediction = tf.equal(prediction_cl, im_true_cl)

result_evaluation = tf.reduce_mean(tf.cast(accurate_prediction, tf.float32))

saver = tf.train.Saver()

   

session = tf.Session()


def batch_generate_rand():
    
    ims = len(im_tr)


    x = np.random.choice(num_images,
                           size=batch_training,
                           replace=False)

   
    batch_dim_x = im_tr[x, :, :, :]
    batch_dim_y = lb_tr[x, :]

    return batch_dim_x, batch_dim_y

def optimize_tensor(counts):
   
    begining = time.time()

    for c in range(counts):
      
    batch_dim_x, batch_dim_y_true = batch_generate_rand()

       
    feed_dict_tr = {input_im: batch_dim_x,
                           im_true_lb: batch_dim_y_true}

       
    global_sess = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)[0]

        if (global_sess % 100 == 0) or (c == counts - 1):
           
            batch_accuracy = session.run(accuracy,
                                    feed_dict=feed_dict_tr)

            
            info = "Global : {0:>6}, Batch Accuracy: {1:>6.1%}"
            print(info.format(global_sess, batch_accuracy))

       
    
    ending = time.time()

   
    diff = ending - begining


    print("Time : " + str(timedelta(seconds=int(round(diff)))))




def draw_confusion_matrix(pred_cl):
  
    matrix = confusion_matrix(y_true=cl_tt, 
                          y_pred=pred_cl)  

    
    for c in range(classes_of_image):
        
        name = "({}) {}".format(c, class_names[c])
        print(matrix[c, :], name)

   
    cl_counts = [" ({0})".format(c) for c in range(classes_of_image)]
    print("".join(cl_counts))




  





def predict_class(ims, lbs, true_cl):

    ims_len = len(ims)

   
    cl_pred = np.zeros(shape=ims_len, dtype=np.int)

   
    c = 0

    while c < ims_len:
       
        d = min(c + batches, ims_len)

      
        feed_dict = {input_im: ims[c:d, :],
                     im_true_lb: lbs[c:d, :]}

        cl_pred[c:d] = session.run(y_pred_cl, feed_dict=feed_dict)

    
        c = d

  
    accu = (cl_true == cl_pred)

    return accu, cl_pred


def wrapper_predict_class():
    return predict_class(ims = im_tt,
                       lbs = lb_tt,
                       true_cl = cl_tt)


def result(accu):
  
    return accu.mean(), accu.sum()


def print_result_test(confusion_matrix=False):

    correct, cl_pred = predict_cls_test()
    
    
    accurate, accu = classification_accuracy(correct)
   
    ims = len(accu)

 
    info = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(info.format(accurate, accu, ims))

   

   
    if confusion_matrix:
        print("Confusion Matrix:")
        draw_confusion_matrix(pred_cl=cl_pred)




print_result_test(confusion_matrix=True)


 session.close()



