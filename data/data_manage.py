"""
数据管理，用于获取测试集，验证集，训练机的图片数据以及label
"""
import inspect
import os
import random

import numpy as np

import utils


class DataManager:
    def __init__(self):
        path = inspect.getfile(DataManager)
        path = os.path.abspath(os.path.join(path, os.pardir))
        self.train_text_path = os.path.join(path, 'train.txt')
        self.valid_text_path = os.path.join(path, 'valid.txt')
        self.test_text_path = os.path.join(path, 'test.txt')
        self.dic = {}
        self.load_dictionary()

    def load_dictionary(self):
        """
        加载字典表
        :return:
        """
        self.get_dict('train', self.train_text_path)
        # self.get_dict('test', self.test_text_path)
        # self.get_dict('valid', self.valid_text_path)

    def get_dict(self, dic_name, path):
        """
        获取字典
        :param dic_name:自定名称
        :param path: 字典文件路径
        """
        with open(path, encoding='utf8') as file:
            lines = file.readlines()
            train = []
            for line in lines:
                content = line.split(' ')
                train.append({'img_path': content[0].strip(), 'label': int(content[1])})
            self.dic[dic_name] = train

    def get_batch_train_data(self, batch_size=1):
        """
        每批次要获取的数据
        :param batch_size: 数据多少
        :return:元组list
        """
        return_list = []
        train_dic = self.dic['train']
        size = len(train_dic)
        for num in range(batch_size):
            location = random.randint(0, size - 1)
            image_path = train_dic[location]['img_path']
            img = utils.load_image(image_path)
            img = img.reshape((1, 224, 224, 3))
            return_list.append({'img': img, 'label': train_dic[location]['label']})
        return return_list

    def get_batch_test_data(self, batch_size):
        """
        获取test中的测试集
        :param batch_size:
        :return:
        """
        return_list = []
        train_dic = self.dic['test']
        size = len(train_dic)
        for num in range(batch_size):
            location = random.randint(0, size - 1)
            image_path = train_dic[location]['img_path']
            img = utils.load_image(image_path)
            img = img.reshape((1, 224, 224, 3))
            return_list.append({'img': img, 'label': train_dic[location]['label']})
        return return_list

    @staticmethod
    def get_ys(train_batchs):
        """
        获得y的值
        :param train_batchs:
        :return:
        """
        y_array = []
        for train in train_batchs:
            label = train['label']
            row = []
            for i in range(1000):
                row.append(0.0)
            row[label] = 1.0
            y_array.append(row)
        y_ndarray = np.array(y_array)
        return y_ndarray

    @staticmethod
    def get_imgs(train_batchs):
        """
        获得igamgs 的batch
        :param train_batchs:
        :return:
        """
        imgs = []
        for train in train_batchs:
            img = train['img']
            imgs.append(img)
        imgs = tuple(imgs)
        imgs = np.concatenate(imgs)
        return imgs


if __name__ == '__main__':
    dataManager = DataManager()
    # test_batch = dataManager.get_batch_test_data(batch_size=100)
    train_batch = dataManager.get_batch_train_data(batch_size=20)
    print(train_batch)
