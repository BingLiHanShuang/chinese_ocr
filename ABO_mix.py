# encoding: utf-8  
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
#导入必要的包

N_CLASSES = 3
#要分类的类别数，这里是5分类
IMG_W = 28
IMG_H = 28
#设置图片的size
BATCH_SIZE = 8
CAPACITY = 64
MAX_STEP = 1000
#迭代一千次，如果机器配置好的话，建议至少10000次以上
learning_rate = 0.0001
#学习率

def inference(images, batch_size, n_classes):#ABO卷积神经网络结构
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
    return softmax_linear

def get_one_image(image_input):#对单字图片进行预处理
     #image = Image.open(img_dir).convert('RGB')
     image = image_input.convert('RGB')

     plt.imshow(image)
     image = image.resize([28, 28])
     image_arr = np.array(image)
     return image_arr

def test(image_input):#将分割后的单字输入卷积神经网络进行ABO识别
    tf.reset_default_graph()
    log_dir = 'mnist/log'#训练好的模型储存在该目录下
    image_arr = get_one_image(image_input)#图像预处理

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,28, 28, 3])
        print(image.shape)
        p = inference(image,1,3)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32,shape = [28,28,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)#调用先前训练好的ABO识别模型
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #调用saver.restore()函数，加载训练好的网络模型
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})#识别
            max_index = np.argmax(prediction) 
            print('预测的标签为：')
            print(max_index)
            print('预测的结果为：')
            print(prediction)
            
            #print('This is a A with possibility %.6f' %prediction[:, 0])
            #print('This is a B with possibility %.6f' %prediction[:, 1])
            #print('This is a O with possibility %.6f' %prediction[:, 2])
            #返回识别结果
            if max_index == 0:
                print('This is a A with possibility %.6f' %prediction[:, 0])
                return 'A'
            elif max_index == 1:
                print('This is a B with possibility %.6f' %prediction[:, 1])
                return 'B'
            else :
                print('This is a O with possibility %.6f' %prediction[:, 2])
                return 'O'

#run_training()
def ABO_detection(image_input):#对chinese_out.py的接口
    result = test(image_input)
    return result
