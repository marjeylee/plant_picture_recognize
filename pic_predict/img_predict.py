import inspect
import os
import numpy as np
import tensorflow as tf
import time

from build_model import ClassifyModel
from data.data_manage import DataManager

"""所有图片平均像素值"""
VGG_MEAN = [103.939, 116.779, 123.68]


def get_ys(train_batch):
    """
    获得y的值
    :param train_batch:
    :return:
    """
    y_array = []
    for train in train_batch:
        label = train['label']
        row = []
        for i in range(1000):
            row.append(0.0)
        row[label] = 1.0
        y_array.append(row)
    y_ndarray = np.array(y_array)
    return y_ndarray


def get_imgs(train_batch):
    """
    获得igamgs 的batch
    :param train_batch:
    :return:
    """
    imgs = []
    for train in train_batch:
        img = train['img']
        imgs.append(img)
    imgs = tuple(imgs)
    imgs = np.concatenate(imgs)
    return imgs


class Vgg16:
    """
    vgg16模型
    """

    def __init__(self, vgg16_npy_path=None):
        """
        指定读取模型存储位置，若没有，则为默认位置。
        获得模型初始值
        :param vgg16_npy_path:
        """
        if vgg16_npy_path is None:
            """
            初始化路径，若不存在，则加载默认路径。
            """
            self.images = tf.placeholder("float", [1, 224, 224, 3])
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        从npy中加载变量，构造vgg。
        :param rgb: 图片样式 [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
        """里面的每个值进行还原"""
        rgb_scaled = rgb * 255.0
        """red, green, blue 的shape(2,224,224,1)"""
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        tmp_a = red.get_shape().as_list()[1:]
        """
        断言：断定输入符合指定格式。
        """
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        """输入图片求平均值"""
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        #########################################################
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")
        #########################################################
        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def predict(self, sess):
        """
        进行一张图片预测
        :return:
        """
        # 定义输入图片
        pass

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        """
        定义卷积层
        :param bottom: 卷基层输入值
        :param name: 卷积层范围名称
        :return:
        """
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)  # 这是一个卷积层
            """
            bottom:地板图片
            filt:过滤器大小
            """
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')  # 进行卷积计算
            conv_biases = self.get_bias(name)  # 定义偏移量
            bias = tf.nn.bias_add(conv, conv_biases)  # 偏移量添加
            relu = tf.nn.relu(bias)  # relu输出
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        constant = tf.constant(self.data_dict[name][0], name="filter")
        return tf.Variable(constant)

    def get_bias(self, name):
        constant = tf.constant(self.data_dict[name][1], name="biases")
        return tf.Variable(constant)

    def get_fc_weight(self, name):
        constant = tf.constant(self.data_dict[name][0], name="weights")
        return tf.Variable(constant)


def init_model(self):
    """
    初始化模型
    :return:
    """
    # 定义输入图片
    self.build(self.images)
    saver = tf.train.Saver()
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7))))
    saver.restore(sess, "../Model/model")
    return sess


def build_model():
    """
    构建一个模型
    :return:
    """
    model = ClassifyModel(30)
    return model, init_model(model)


if __name__ == '__main__':
    vgg16 = Vgg16()
    vgg16.predict(vgg16)
