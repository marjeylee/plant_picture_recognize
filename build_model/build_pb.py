"""
生成pb模型
"""
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

from build_model import ClassifyModel

pb_file_path = os.getcwd()

model = ClassifyModel(1)
model.get_pb()
