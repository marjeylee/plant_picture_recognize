import skimage
import skimage.io
import skimage.transform
import numpy as np


def load_image(path):
    """
    输入待检查的图片路径，返回经过标准化的图片（224，224，3）（高度，宽度，深度）
    图片已经经过正规化
    :param path:文件路径
    :return:
    """
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    resized_img = resized_img * 255.0
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    load_image("./test_data/plant.jpg")
    # img = skimage.io.imread("./test_data/plant.jpg")
    # ny = 300
    # nx = img.shape[1] * ny / img.shape[0]
    # img = skimage.transform.resize(img, (ny, nx))
    # skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
