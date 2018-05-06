"""
加载pb模型
"""
import os

import tensorflow as tf

from data.data_manage import DataManager

pb_file_path = os.getcwd()
sess = tf.Session()
with tf.gfile.FastGFile(pb_file_path + '/pb/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())
    scope = sess.graph.get_name_scope()
    keys = sess.graph.get_all_collection_keys()
    ops = sess.graph.get_operations()

    input_images = sess.graph.get_tensor_by_name('input_images_batch:0')
    keep_prob22 = sess.graph.get_tensor_by_name('keep_prob22:0')
    keep_prob20 = sess.graph.get_tensor_by_name('keep_prob20:0')
    probability = sess.graph.get_tensor_by_name('probability:0')
    data_manager = DataManager()
    train_batch = data_manager.get_batch_train_data(batch_size=1)
    imgs = data_manager.get_imgs(train_batch)
    full_connection_layer23 = sess.graph.get_tensor_by_name('full_connection_layer23/full_connection_layer23_fc:0')
    ret = sess.run(probability, feed_dict={input_images: imgs, keep_prob22: 0.99, keep_prob20: 0.99})
    full_connection_layer23_values = sess.run(full_connection_layer23,
                                              feed_dict={input_images: imgs, keep_prob22: 0.99, keep_prob20: 0.99})
    print(ret)
    print(full_connection_layer23_values)
