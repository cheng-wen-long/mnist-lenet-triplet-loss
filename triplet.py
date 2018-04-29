#!/usr/bin/env mdl
# -*- coding:utf-8 -*-
# $Author: cwl<chengwenlong@bupt.com>
import numpy as np
import math
import tensorflow as tf
import collections
import struct
import cv2
model_dir = './model'
def scale_mean_norm(data, scale=0):
    mean = np.mean(data)
    data = (data - mean) * scale
    return data, mean

class data_generate:
    def __init__(self, data, labels, scale=True):
        scale_value = 0.0039625
        total_samples = data.shape[0]
        indexes = np.array(range(total_samples))
        np.random.shuffle(indexes)
        data = np.reshape(data, (data.shape[0], 28, 28, 1))
        self.total_labels = 10

        if scale:
            self.data, self.mean = scale_mean_norm(data, scale=scale_value)
        else:
            self.data = data
        self.labels = labels

    def get_batch(self, n_samples):
        
        data = self.data
        labels = self.labels
        indexes = np.array(range(data.shape[0]))
        np.random.shuffle(indexes)
        selected_data = data[indexes[0:n_samples], :, :, :]
        selected_labels = labels[indexes[0:n_samples]]

        return selected_data.astype('float32'), selected_labels
    
    def get_triplet(self, n_labels, n_triplet=1, is_target_set_train=True):
        
        def get_one_triplet(input_data, input_labels):
            
            #index = np.random.choice(n_labels, 2, p=[0.182, 0.163, 0.145, 0.127, 0.109, 0.091, 0.073, 0.055, 0.036, 0.019], replace=False)
            #index = np.random.choice(n_labels, 2, replace=False)
            #p_list = [5000 for i in range(6)]
            #for i in range(4):
            #    p_list.append(200)
            #p = []
            #for i in range(9):
            #    p.append(float(p_list[i])/sum(p_list))
            #p.append(1-sum(p))
            index = np.random.choice(n_labels, 2, replace=False)
            label_positive = index[0]
            label_negative = index[1]
            sum_label = 0
            indexes = np.where(input_labels == index[0])
            np.random.shuffle(indexes)
            #print('indexes:', indexes)
            #print('indexes[0]:', indexes[0])
            indexes = indexes[0]
            data_anchor = input_data[indexes[0], :, :, :]
            data_positive = input_data[indexes[1], :, :, :]
            indexes = np.where(input_labels == index[1])[0]
            np.random.shuffle(indexes)
            data_negative = input_data[indexes[0], :, :, :]

            return data_anchor, data_positive, data_negative, label_positive, label_positive, label_negative

        index_random = np.array(range(self.data.shape[0]))
        np.random.shuffle(index_random)
        target_data = self.data[index_random]
        target_labels = self.labels[index_random]
        c = target_data.shape[3]
        w = target_data.shape[2]
        h = target_data.shape[1]
#        data_a = np.zeros(shape=(n_triplet, w, h, c), dtype='float32')
#        data_p = np.zeros(shape=(n_triplet, w, h, c), dtype='float32')
#        data_n = np.zeros(shape=(n_triplet, w, h, c), dtype='float32')
#        labels_a = np.zeros(shape=n_triplet, dtype='float32')
#        labels_p = np.zeros(shape=n_triplet, dtype='float32')
#        labels_n = np.zeros(shape=n_triplet, dtype='float32')
        data = np.ones(shape=(n_triplet, w, h, c), dtype='float32')
        labels = np.ones(shape=n_triplet, dtype='float32')
#        for i in range(n_triplet):
#            data_a[i, :, :, :], data_p[i, :, :, :], data_n[i, :, :, :], \
#                    labels_a[i], labels_p[i], labels_n[i] = \
#                    get_one_triplet(target_data, target_labels)
#
        for i in range(n_triplet/3):
            data[3*i, :, :, :], data[3*i+1, :, :, :], data[3*i+2, :, :, :],\
                    labels[3*i], labels[3*i+1], labels[3*i+2] = \
                    get_one_triplet(target_data, target_labels)
        return data, labels
    def normal_data(self, batch_size):
        c = self.data.shape[3]
        w = self.data.shape[2]
        h = self.data.shape[1]
        #data = np.zeros(shape=(batch_size, w, h, c), dtype='float32')
        #labels = np.zeros(shape=batch_size, dtype='float32')
        indexes = np.array(range(self.data.shape[0]))
        np.random.shuffle(indexes)
        #print('self_data_shape:', self.data)
        #print(indexes[0])
        data, labels = self.data[indexes[0:210]], self.labels[indexes[0:210]]
        return data, labels



def create_weight_variables(shape, seed, name, use_gpu=False):
    if len(shape) == 4:
        in_out = shape[0] * shape[1] * shape[2] + shape[3]
    else:
        in_out = shape[0] + shape[1]

    stddev = math.sqrt(3.0 / in_out)
    initializer = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    if use_gpu:
        with tf.device('/gpu'):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device('/cpu'):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

def create_biase_variables(shape, name, use_gpu=False):
    """ create variables of biase"""
    initializer = tf.constant(0.0, shape=shape)
    if use_gpu:
        with tf.device('/gpu'):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device('/cpu'):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

def create_conv2d(x, W):
    """ create convoltuion layer """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def create_max_pool(x, kernel_size=2, padding='SAME'):
    """ create pooling layer """
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 2, 2, 1], padding=padding)

def create_PRelu(x, biase=0, name=None):
    """ create active function """
    with tf.variable_scope(name):
        _alaph = tf.get_variable('alaph', shape=x.get_shape(), initializer=tf.constant_initializer(0.1),
                                dtype=x.dtype)
    #x = tf.nn.bias_add(x, biase)
    #return tf.nn.elu(x)
    return tf.maximum(_alaph*x, x, name=name)

class lenet:
    """ lenet++ network"""
    def __init__(self, seed=10, use_gpu=True):
        self.w_conv1 = create_weight_variables([5, 5, 1, 32], seed=seed, name='w_conv1', use_gpu=use_gpu)
        self.b_conv1 = create_biase_variables([32], name='b_conv1', use_gpu=use_gpu)
        self.w_conv1_plus = create_weight_variables([5, 5, 32, 32], seed=seed, name='w_conv1_plus', use_gpu=use_gpu)
        self.b_conv1_plus = create_biase_variables([32], name='b_conv1_plus', use_gpu=use_gpu)
        self.w_conv2 = create_weight_variables([5, 5, 32, 64], seed=seed, name='w_conv2', use_gpu=use_gpu)
        self.b_conv2 = create_biase_variables([64], name='b_conv2', use_gpu=use_gpu)
        self.w_conv2_plus = create_weight_variables([5, 5, 64, 128], seed=seed, name='w_conv2_plus', use_gpu=use_gpu)
        self.b_conv2_plus = create_biase_variables([128], name='b_conv2_plus', use_gpu=use_gpu)
        self.w_conv3 = create_weight_variables([5, 5, 128, 128], seed=seed, name='w_conv3', use_gpu=use_gpu)
        self.b_conv3 = create_biase_variables([128], name='b_conv3', use_gpu=use_gpu)
        self.w_conv3_plus = create_weight_variables([5, 5, 128, 128], seed=seed, name='w_conv3_plus', use_gpu=use_gpu)
        self.b_conv3_plus = create_biase_variables([128], name='b_conv3_plus', use_gpu=use_gpu)
        self.ip1 = create_weight_variables([3*3*128, 2], seed=seed, name='ip1', use_gpu=use_gpu)
        self.ip1_bias = create_biase_variables([2], name='ip1_bias', use_gpu=use_gpu)
        self.ip2 = create_weight_variables([2, 10], seed=seed, name='ip2', use_gpu=use_gpu)
        self.ip2_bias = create_biase_variables([10], name='ip2_bias', use_gpu=use_gpu)

    def create_lenet(self, data, train=True):
        """
        create lenet++ network

        **parameters**
        data: input_data
        train:

        **returns**
        features of ip1
        logits

        """
        with tf.name_scope('conv1') as scope:
            conv1 = create_conv2d(data, self.w_conv1)

        with tf.name_scope('prelu1') as scope:
            prelu1 = create_PRelu(conv1, self.b_conv1, name='prelu1')

        with tf.name_scope('conv1_plus') as scope:
            conv1_plus = create_conv2d(prelu1, self.w_conv1_plus)

        with tf.name_scope('prelu1_plus') as scope:
            prelu1_plus = create_PRelu(conv1_plus, self.b_conv1_plus, name='prelu1_plus')
        
        with tf.name_scope('conv1_pooling') as scope:
            pool1 = create_max_pool(prelu1_plus)

        with tf.name_scope('conv2') as scope:
            conv2 = create_conv2d(pool1, self.w_conv2)

        with tf.name_scope('prelu2') as scope:
            prelu2 = create_PRelu(conv2, self.b_conv2, name='prelu2')

        with tf.name_scope('conv2_plus') as scope:
            conv2_plus = create_conv2d(prelu2, self.w_conv2_plus)

        with tf.name_scope('prelu2_plus') as scope:
            prelu2_plus = create_PRelu(conv2_plus, self.b_conv2_plus, name='prelu2_plus')
        
        with tf.name_scope('conv2_pooling') as scope:
            pool2 = create_max_pool(prelu2_plus)

        with tf.name_scope('conv3') as scope:
            conv3 = create_conv2d(pool2, self.w_conv3)

        with tf.name_scope('prelu3') as scope:
            prelu3 = create_PRelu(conv3, self.b_conv3, name='prelu3')

        with tf.name_scope('conv3_plus') as scope:
            conv3_plus = create_conv2d(prelu3, self.w_conv3_plus)

        with tf.name_scope('prelu3_plus') as scope:
            prelu3_plus = create_PRelu(conv3_plus, self.b_conv3_plus, name='prelu3_plus')
    
        with tf.name_scope('conv3_pooling') as scope:
            pool3 = create_max_pool(prelu3_plus, kernel_size=3, padding='VALID')
        
        with tf.name_scope('ip1') as scope:
            pool_shape = pool3.get_shape().as_list()
            reshape = tf.reshape(pool3, [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
            ip1 = tf.matmul(reshape, self.ip1)
        #ip1_bias = create_PRelu(ip1, self.ip1_bias, name='ip1')
        ip1_bias = tf.add(ip1, self.ip1_bias)
        with tf.name_scope('ip2') as scope:
            ip2 = tf.matmul(ip1_bias, self.ip2)

        return ip1, ip2

def compute_eulidean_distance(x, y):
    """
    computes the eulidean distance between two tensorflow variabls
    """
    d = tf.square(tf.subtract(x, y))
    d = tf.nn.l2_normalize(d, dim=1)
    d = tf.reduce_mean(d, axis=1)
    return d

def comput_triplet_loss(data, margin, batch_size):
    """
    compute the triplet loss
    """
    anchor_feature = tf.get_variable(shape=[batch_size/3, 2], name='anchor_feature')
    positive_feature = tf.get_variable(shape=(batch_size/3, 2), name='positive_feature')
    negative_feature = tf.get_variable(shape=(batch_size/3, 2), name='negative_feature')
    for i in range(batch_size/3):
       # anchor_feature[i] = data[i*3+0]
       # positive_feature[i] = data[i*3+1]
       # negative_feature[i] = data[i*3+2]
       # print('data_tensor:',data[i])
        tf.assign(anchor_feature[i], data[i*3+0])
        tf.assign(positive_feature[i], data[i*3+1])
        tf.assign(negative_feature[i], data[i*3+2])
    with tf.name_scope('triplet_loss') as scope:
        dp_squared = compute_eulidean_distance(anchor_feature, positive_feature)
        dn_squared = compute_eulidean_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0.0, dp_squared - dn_squared + margin)
        #print('loss_shape:', loss)
    return loss, tf.reduce_mean(dp_squared), tf.reduce_mean(dn_squared)

def data_read(file_data, file_label, average=True, train=True):
    """
    read data from mnist 
    """

    def load_image_set(filename):
        """
        load images
        """
        binfile = open(filename, 'rb')
        buffles = binfile.read()
        head = struct.unpack_from('>IIII', buffles, 0)
        offset = struct.calcsize('>IIII')
        imgNum = head[1]
        width = head[2]
        height = head[3]
        bits = imgNum*width*height
        bitstring = '>'+str(bits)+'B'
        imgs = struct.unpack_from(bitstring, buffles, offset)
        binfile.close()
        imgs = np.reshape(imgs, [imgNum,width,height])
        return imgs

    def load_label_set(filename):
        """
        load labels
        """
        binfile = open(filename)
        buffles = binfile.read()
        head = struct.unpack_from('>II', buffles, 0)
        #print(head)
        imgNum = head[1]
        offset = struct.calcsize('>II')
        numstring = '>'+str(imgNum)+'B'
        labels = struct.unpack_from(numstring, buffles, offset)
        binfile.close()
        labels = np.reshape(labels, [imgNum,1])
        return labels

    imgs = load_image_set(file_data)
    labels_all = load_label_set(file_label)
    label_dict = collections.defaultdict(int)
    if train is True:

        if average is True:
            label_count = [5000 for i in range(10)]
        else:
            label_count = [5000 for i in range(6)]
            for i in range(4):
                label_count.append(200)
        total_num = 0
        for num in label_count:
            total_num += num
        print('total_num:',total_num)
        data = np.ones(shape=(total_num,28,28))
        labels = np.ones(shape=(total_num,))
        index = 0
        for i in range(60000):
            label_dict[labels_all[i][0]] += 1
            if label_dict[labels_all[i][0]] < label_count[labels_all[i][0]]:
                data[index] = imgs[i]
                labels[index] = labels_all[i][0]
                index += 1
            else:
                continue
        return data, labels
    else:
        data = np.ones(shape=(10000,28,28))
        labels = np.ones(shape=(10000,))
        for i in range(10000):
            data[i] = imgs[i]
            labels[i] = labels_all[i][0]
        return data, labels

def main():
    BATCH_SIZE = 210
    ITERATIONS = 30000
    VALIDATION_TEST = 1000
    MARGIN = 0.2
    WEIGHT_DECAY = 2e-5 
    data_train, labels_train = data_read('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', average=True)
    data_test, labels_test = data_read('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', train=False)
    train_data_shuffle = data_generate(data_train, labels_train)
    test_data_shuffle = data_generate(data_test, labels_test)

    train_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='anchor_data')
    #train_positive_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='positive_data')
    #train_negative_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='negative_data')
    train_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
    test_data = tf.placeholder(tf.float32, shape=(10000, 28, 28, 1), name='test_data')
    test_labels = tf.placeholder(tf.int32, shape=(10000,), name='test_labels')
    #labels_positive = tf.placeholder(tf.int32, shape=BATCH_SIZE)
    #labels_negative = tf.placeholder(tf.int32, shape=BATCH_SIZE)
    lenet_architecture = lenet(seed=10)
    ip1, logits = lenet_architecture.create_lenet(train_data)
    #ip1_norm = tf.nn.l2_normalize(ip1, dim=1)
    #print(ip1[10])
    loss_triplet, positive_feature, negative_feature = comput_triplet_loss(ip1, MARGIN, BATCH_SIZE)
    batch = tf.Variable(0, trainable=False)
    loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(train_labels,depth=10), logits=logits))
    #print('loss_triplet:', loss_triplet)
    loss_new = loss_softmax #+ 0.05* tf.reduce_mean(loss_triplet)
    for v in tf.trainable_variables():
        print(v)
    #loss += WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    correct = tf.nn.in_top_k(logits, train_labels,1)
    #accuracy = tf.metrics.accuracy(train_labels, tf.argmax(logits,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    learning_rate_new = tf.train.exponential_decay(
            0.002,
            batch*BATCH_SIZE,
            train_data_shuffle.data.shape[0],
            0.95)
    def learning(step):
        if step < 2000:
            return 0.01
        elif step < 5000:
            return 0.001
        elif step < 15000:
            return 0.0001
        else:
            return 0.00001
    optimizer_new = tf.train.MomentumOptimizer(learning_rate_new,0.9).minimize(loss_new, global_step=batch)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        train_writer = tf.summary.FileWriter('./logs_tensorboard/triplet/train',
                                                sess.graph)
        test_writer = tf.summary.FileWriter('./logs_tensorboard/triplet/test',
                                                sess.graph)
        tf.summary.scalar('loss', loss_new)
        batch = tf.Variable(0, trainable=False)
        merged = tf.summary.merge_all()
        #tf.initilaizer_all_variables().run()
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        for step in range(ITERATIONS):
            batch_data, batch_labels = train_data_shuffle.get_triplet(n_labels=10, n_triplet=BATCH_SIZE)
            #print(batch_data)
            #batch_data, batch_labels = train_data_shuffle.normal_data(210)
            #batch_labels = tf.one_hot(batch_labels, depth=10)
            #batch_data = tf.convert_to_tensor(batch_data)
            #batch_labels = tf.convert_to_tensor(batch_labels)
            #print(batch_data)
            feed_dict = {train_data: batch_data,
                        train_labels: batch_labels}
            #print('batch_data:',batch_data)
            if step % 100 == 0:
                print('train_batch_data:',np.shape(batch_data))
                loss_train, positive_feature1, negative_feature1, acc1 = sess.run([loss_new, positive_feature, negative_feature, accuracy], feed_dict=feed_dict)
                print('loss_train:{}_step:{}'.format(loss_train, step))
                print('train_positvie:', positive_feature1)
                print('train_negative:', negative_feature1)
                #print('train_accuracy:{}'.format(acc1))
            _, summary = sess.run([optimizer_new, merged], feed_dict=feed_dict)
            train_writer.add_summary(summary,step)

            if step % 100 == 0:
                batch_data, batch_labels = test_data_shuffle.get_triplet(n_labels=10, n_triplet=BATCH_SIZE)
                #batch_data, batch_labels = test_data_shuffle.normal_data(210)
                #batch_data, batch_labels = test_data_shuffle.data, test_data_shuffle.labels
                #accuracy = np.zeros(shape=(48,))
                #loss_test_list = np.zeros(shape=(48,))
                #for i in range(48):
                feed_dict = {train_data: batch_data,
                            train_labels: batch_labels}
                loss_test,p_f, n_f, acc, lr, summary = sess.run([loss_new, positive_feature, negative_feature, accuracy, learning_rate_new, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                print('step:', step)
                print('batch:', batch)
                #print(batch_labels)
                print('accuracy:', acc)
                print('positive feature:', p_f)
                print('neagtive feature:', n_f)
                print('loss_test:', loss_test)
                print('learning_rate:', lr)
                saver.save(sess, './log/models', global_step=step)
            train_writer.close()
            test_writer.close()

    
if __name__ == '__main__':
    main()
