import imutils
import cv2
import numpy as np


if __name__ == '__main__':
    filename = 'F:\pythonProject\Data\\20220522\\21-300-2.tif'
    img1 = cv2.imread(filename, 1)
    model = ROISelection(img1)
    img1 = model.selection_roi()
    white_paper = np.zeros([img1.shape[0], img1.shape[1], img1.shape[2]], np.uint8)  # 创建全新图像
    selection_roi(white_paper)