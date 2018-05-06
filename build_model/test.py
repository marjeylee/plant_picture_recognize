"""
图片分类模型
"""
import os

import tensorflow as tf

from build_model.model_shape import model_shape
from data.data_manage import DataManager


class ClassifyModel:
    """
    使用vgg16
    """

    def __init__(self, img_batch_size):
        """
        初始化模型
        :param img_batch_size:每批次传入的
        正规化，求过均值。
        """

        self.input_images = tf.placeholder("float", [img_batch_size, 224, 224, 3], name='input_images_batch')
        self.normal_image = self.normalize_image()
        ##############################
        self.convolution_layer1 = self.create_convolution_layer(self.normal_image, 'convolution_layer1')
        self.convolution_layer2 = self.create_convolution_layer(self.convolution_layer1, 'convolution_layer2')
        self.pool_layer3 = self.max_pool(self.convolution_layer2, 'pool_layer3')
        ###############################
        self.convolution_layer4 = self.create_convolution_layer(self.pool_layer3, 'convolution_layer4')
        self.convolution_layer5 = self.create_convolution_layer(self.convolution_layer4, 'convolution_layer5')
        self.pool_layer6 = self.max_pool(self.convolution_layer5, 'pool_layer6')
        ###############################
        self.convolution_layer7 = self.create_convolution_layer(self.pool_layer6, 'convolution_layer7')
        self.convolution_layer8 = self.create_convolution_layer(self.convolution_layer7, 'convolution_layer8')
        self.convolution_layer9 = self.create_convolution_layer(self.convolution_layer8, 'convolution_layer9')
        self.pool_layer10 = self.max_pool(self.convolution_layer9, 'pool_layer10')
        ###############################
        self.convolution_layer11 = self.create_convolution_layer(self.pool_layer10, 'convolution_layer11')
        self.convolution_layer12 = self.create_convolution_layer(self.convolution_layer11, 'convolution_layer12')
        self.convolution_layer13 = self.create_convolution_layer(self.convolution_layer12, 'convolution_layer13')
        self.pool_layer14 = self.max_pool(self.convolution_layer13, 'pool_layer14')
        ###############################
        self.convolution_layer15 = self.create_convolution_layer(self.pool_layer14, 'convolution_layer15')
        self.convolution_layer16 = self.create_convolution_layer(self.convolution_layer15, 'convolution_layer16')
        self.convolution_layer17 = self.create_convolution_layer(self.convolution_layer16, 'convolution_layer17')
        self.pool_layer18 = self.max_pool(self.convolution_layer17, 'pool_layer18')
        ###############################
        self.full_connection_layer19 = self.create_full_connection_layer(self.pool_layer18, "full_connection_layer19")
        self.relu20 = tf.nn.relu(self.full_connection_layer19, name='relu20')
        self.keep_prob20 = tf.placeholder(tf.float32, name='keep_prob20')
        self.dropout20 = tf.nn.dropout(self.relu20, keep_prob=self.keep_prob20, name='dropout20')
        ###############################
        self.full_connection_layer21 = self.create_full_connection_layer(self.dropout20, "full_connection_layer21")
        self.relu22 = tf.nn.relu(self.full_connection_layer21, name='relu22')
        self.keep_prob22 = tf.placeholder(tf.float32, name='keep_prob22')
        self.dropout22 = tf.nn.dropout(self.relu22, self.keep_prob22, name='dropout22')
        ###############################
        self.full_connection_layer23 = self.create_full_connection_layer(self.dropout22, "full_connection_layer23")
        # self.relu24 = tf.nn.relu(self.full_connection_layer23, name='relu24')
        # self.keep_prob24 = tf.placeholder(tf.float32, name='keep_prob24')
        # self.dropout24 = tf.nn.dropout(self.relu24, self.keep_prob24, name='dropout24')
        ###############################
        self.probability = tf.nn.softmax(self.full_connection_layer23, name="probability")

    def train(self):
        """
        进行模型训练
        :return:
        """
        # 定义输入图片

        y_ = tf.placeholder(tf.float32, shape=[None, 1000])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.full_connection_layer23, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.full_connection_layer23, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        data_manager = DataManager()
        saver = tf.train.Saver()
        with tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
            saver.restore(sess, "./Model/model")
            # sess.run(tf.global_variables_initializer())
            # train_batch = data_manager.get_batch_train_data(batch_size=30)
            for i in range(500000):
                try:
                    # if i % 10 == 0 and i != 0:
                    #     saver.save(sess, "./Model/model")
                    #     print(str(i) + '---保存模型')
                    train_batch = data_manager.get_batch_train_data(batch_size=30)
                    ys = data_manager.get_ys(train_batch)
                    imgs = data_manager.get_imgs(train_batch)
                    result = sess.run(self.full_connection_layer23, feed_dict={
                        self.input_images: imgs, y_: ys, self.keep_prob20: 0.70, self.keep_prob22: 0.85
                    })
                    print(result)
                    result = sess.run(self.probability, feed_dict={
                        self.input_images: imgs, y_: ys, self.keep_prob20: 0.70, self.keep_prob22: 0.85
                    })
                    print(result)


                except Exception as e:
                    print(e)

    def normalize_image(self):
        """
        对各种各样的图片进行正规化操作，
        使之成为[224,224,3]，已经经过mean的图片
        :return:
        """
        rgb = self.input_images
        """里面的每个值进行还原"""
        rgb_scaled = rgb * 255.0

        red, green, blue = tf.split(rgb_scaled, 3, 3)
        """
        断言：断定输入符合指定格式。
        """
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        """输入图片求平均值"""
        vgg_mean = [103.939, 116.779, 123.68]
        bgr = tf.concat([
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ], 3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        return bgr

    def create_convolution_layer(self, bottom, name):
        """
        创建卷积层
        :param bottom:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            filt = self.get_wight(name)  # 卷积核
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')  # 进行卷积计算
            conv_biases = self.get_bias(name)  # 定义偏移量
            bias = tf.nn.bias_add(conv, conv_biases)  # 偏移量添加
            name = name + '_relu'
            relu = tf.nn.relu(bias, name=name)  # relu输出
            return relu

    @staticmethod
    def get_wight(name):
        """
        获取权重w
        :param name:卷积层名称
        :return:
        """
        shape = model_shape[name][0]
        initial_value = tf.truncated_normal(shape, stddev=0.1)
        name = name + '_w'
        return tf.Variable(initial_value, name=name)

    @staticmethod
    def get_bias(name):
        """
        获取偏移量
        :param name:
        :return:
        """
        shape = model_shape[name][1]
        initial_value = tf.truncated_normal(shape, stddev=0.1)
        name = name + '_b'
        return tf.Variable(initial_value, name=name)

    @staticmethod
    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def create_full_connection_layer(self, bottom, name):
        """
        创建全连接层
        :param bottom: 上一层输出
        :param name:名称
        :return:
        """
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.get_wight(name)
            biases = self.get_bias(name)
            name = name + '_fc'
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases, name=name)
            return fc

    def get_pb(self):
        """
        获取pb文件
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
            saver.restore(sess, "./Model/model")
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['probability'])
            pb_file_path = os.getcwd()
            with tf.gfile.FastGFile(pb_file_path + '/pb/model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    model = ClassifyModel(30)
    model.train()
