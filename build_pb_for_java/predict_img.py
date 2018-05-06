import os
import tensorflow as tf

from tensorflow.python.platform import gfile

import utils

pb_file_path = os.getcwd()

sess = tf.Session()
with gfile.FastGFile(pb_file_path + '/pb/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())

image_path = "F:/224.jpg"
img = utils.load_image(image_path)
img = img.reshape((1, 224, 224, 3))
# 输入
input_x = sess.graph.get_tensor_by_name('input_images_batch:0')
output_y = sess.graph.get_tensor_by_name('probability:0')
ret = sess.run(output_y, feed_dict={input_x: img})
print(ret)
