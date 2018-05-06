"""
用于图片预测模块
"""
import utils
from pic_predict.img_predict import *

num_name_mapping = {0: '石莲花', 1: '木贼', 2: '白果', 3: '侧柏叶', 4: '玉兰花', 5: '荷叶', 6: '枇杷', 7: '鸡冠花', 8: '紫薇花', 9: '金丝桃'}
name_num_mapping = {'石莲花': 0, '木贼': 1, '白果': 2, '侧柏叶': 3, '玉兰花': 4, '荷叶': 5, '枇杷': 6, '鸡冠花': 7, '紫薇花': 8, '金丝桃': 9}


def find_most_possibility_top_three(result):
    """
    找到最可能前三个结果
    :param result:
    :return:
    """
    result_list = result.tolist()[0]

    top = [-1, -1.1]
    second = [-1, -1.1]
    third = [-1, -1.1]
    for i in range(len(result_list)):
        value = result_list[i]
        if value < third[1]:
            continue
        if second[1] > value >= third[1]:
            third[0] = i
            third[1] = value
            continue
        if top[1] > value >= second[1]:
            third[0] = second[0]
            third[1] = second[1]
            second[0] = i
            second[1] = value
            continue
        if value >= top[1]:
            third[0] = second[0]
            third[1] = second[1]
            second[0] = top[0]
            second[1] = top[1]
            top[0] = i
            top[1] = value
            continue
    if top[0] < 10:
        top[0] = num_name_mapping[top[0]]
    if second[0] < 10:
        second[0] = num_name_mapping[second[0]]
    if third[0] < 10:
        third[0] = num_name_mapping[third[0]]
    return top, second, third


while True:
    model, sess = build_model()
    img_path = './img/'
    img_path = img_path + input("输入图片名称: ")
    img = utils.load_image(img_path)
    img = img.reshape((1, 224, 224, 3))
    result = sess.run(model.prob, feed_dict={model.images: img})
    top_three = find_most_possibility_top_three(result)
    print(top_three)
