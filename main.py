import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import Methods
import matplotlib.pyplot as plt
import ImageDeal
import PrivateTools
import LeastsqEllipse
#TODo
#在提取单个圆之后，进行边缘平滑处理

if __name__ == '__main__':
    # 1.读取图片,初始化工具类
    # filename = 'F:\pythonProject\Data\\20220525\\12-3.tif'
    filename = '65-1.tif'
    # filename = 'F:\cv-for-hwglue\Data\\0-2.tif'
    # filename = 'F:\cv-for-hwglue\Data\\20220712\P12\\21 - 1 - spd5.tif'

    img = cv2.imread(filename, 1)
    # img = cv2.resize(img,(8192, 7102))
    Tools = PrivateTools.Tools()
    # 2.初始化阈值选择类，选择二值化阈值
    img_getvalue = cv2.resize(img, (800, 800))
    PickThresholdModel = PrivateTools.Pick_threshold(img_getvalue)
    PickThresholdModel.getModeltypeAndValue()
    print("a=%s\nb = %s\nd = %s\n" % (PickThresholdModel.binarization_type, PickThresholdModel.binarization_value, PickThresholdModel.filter_value))
    # 3.进行初始化处理，灰度，二值化，滤波等
    ImageDealmodel = ImageDeal.ImageDeal(img)
    ImageDealmodel.gray_procession(ImageDealmodel.img)
    ImageDealmodel.threshold_procession(ImageDealmodel.img_gray,PickThresholdModel.binarization_type,PickThresholdModel.binarization_value)
    Tools.img_show(ImageDealmodel.img_thresholds)
    # 4.粗识别轮廓
    EdgeContoursModel = Methods.findEdgeContours(ImageDealmodel.img_thresholds)
    EdgeContoursModel.contours_filter()
    # 5.选择ROI
    ImageDealmodel.onlyCountersImage(EdgeContoursModel.contours)
    ROISelectionModel = PrivateTools.ROISelection(ImageDealmodel.img_onlyCounters)
    ROISelectionModel.selection_roi(EdgeContoursModel.contours)
    Tools.scatter_img(np.array(ROISelectionModel.coor))
    print("aaa")
    # 6.边缘处理和霍夫圆粗识别圆心
    # img_gray = cv2.cvtColor(ImageDealmodel.img_onlyCounters, cv2.IMREAD_GRAYSCALE)
    # canny = Methods.Canny(img_gray)
    # canny.canny_algorithm()
    # Hough = Methods.Hough_transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)
    # circles = Hough.Calculate()
    # for circle in circles:
    #     cv2.circle(img, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 2)
    # Tools.img_show(img)
    # print('Finished!')
    # imgray = cv2.Canny(ImageDealmodel.img_onlyCounters, 30, 100)  # Canny算子边缘检测
    # img_circles = hough.hough_circle(imgray)

    # img_onlyleftCounters,img_onlyright = ImageDealmodel.onlySplitCountersImage(ROISelectionModel.leftcontour,ROISelectionModel.rightcontour)
    # gray_img = cv2.cvtColor(ImageDealmodel.img_onlyleftCounters, cv2.COLOR_BGR2GRAY)
    # _, threshold_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    # Tools.img_show(threshold_img)
    # ImageHufuModel = Methods.HufuMethod(threshold_img)
    # ImageHufuModel.houghCircle(threshold_img)
    # leftCOC = ImageHufuModel.COC[0]
    # gray_img = cv2.cvtColor(ImageDealmodel.img_onlyrightCounters, cv2.COLOR_BGR2GRAY)
    # _, threshold_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    # ImageHufuModel = Methods.HufuMethod(threshold_img)
    # ImageHufuModel.houghCircle(threshold_img)
    # rightCOC = ImageHufuModel.COC[0]
    leftcontour = []
    for i in range(len(ROISelectionModel.leftcontour)):
        leftcontour.append(ROISelectionModel.leftcontour[i])
    px, py = PrivateTools.Tools.splitCoorsToPxPy(leftcontour)
    Simple_LeastSmodel = Methods.LeastSquares(px,py)
    CircleCenterX, CircleCenterY, CircleCenterR2 =  Simple_LeastSmodel.getCircleCenter(px=px,py=py)
    leftCOC = [CircleCenterX, CircleCenterY]

    rightcontour = []
    for i in range(len(ROISelectionModel.rightcontour)):
        rightcontour.append(ROISelectionModel.rightcontour[i])
    px, py = PrivateTools.Tools.splitCoorsToPxPy(rightcontour)
    Simple_LeastSmodel = Methods.LeastSquares(px,py)
    CircleCenterX, CircleCenterY, CircleCenterR2 =  Simple_LeastSmodel.getCircleCenter(px=px,py=py)
    rightCOC = [CircleCenterX, CircleCenterY]


    ImageDealmodel.selectTwoRound(ROISelectionModel.leftcontour,ROISelectionModel.rightcontour,leftCOC,rightCOC,ROISelectionModel.pts)


    # 7.进入Rasacn算法圆识别
    ImageLeastSquares=Methods.LeastSquares()
    px, py = PrivateTools.Tools.splitCoorsToPxPy(leftcontour)
    ImageRascanModel = Methods.Ransac()
    leftfit = ImageRascanModel.fitting(ImageLeastSquares, px, py)
    px, py = PrivateTools.Tools.splitCoorsToPxPy(rightcontour)
    ImageRascanModel = Methods.Ransac()
    rightfit = ImageRascanModel.fitting(ImageLeastSquares, px, py)

    # 7.进入Rasacn算法椭圆识别
    px, py = PrivateTools.Tools.splitCoorsToPxPy(leftcontour)
    # ImageLeastSquares=LeastsqEllipse.func(px,py)
    ImageLeastSquares = LeastsqEllipse.RascanEllipse()
    # left_Xc, left_Yc, left_MA, left_SMA, left_Theta,left_x1,left_y1 = ImageLeastSquares.getEllipseParam()
    left_Xc, left_Yc, left_MA, left_SMA, left_Theta = ImageLeastSquares.fitting(ImageLeastSquares, px, py)
    px, py = PrivateTools.Tools.splitCoorsToPxPy(rightcontour)
    # ImageLeastSquares = LeastsqEllipse.func(px, py)
    # right_Xc, right_Yc, right_MA, right_SMA, right_Theta,right_x1,right_y1 = ImageLeastSquares.getEllipseParam()
    ImageLeastSquares = LeastsqEllipse.RascanEllipse()
    right_Xc, right_Yc, right_MA, right_SMA, right_Theta = ImageLeastSquares.fitting(ImageLeastSquares, px, py)

    # px,py = PrivateTools.Tools.splitCoorsToPxPy(ImageDealmodel.rightinnercontour)
    # ImageRascanModel = Methods.RansacEllipse()
    # rightfit = ImageRascanModel.fitting(ImageLeastSquares, px, py)


    print("the center of left circle is",leftfit[0],leftfit[1])
    print("the center of left circle is",rightfit[0],rightfit[1])

    print("the center of left ellipse is", left_Xc, left_Yc)
    print("the center of left ellipse is", right_Xc, right_Yc)



    # # 8. 解析图片,获得坐标值
    # scalefactor = PrivateTools.Tools.parseImage(img)
    # Distance    =PrivateTools.Tools.calDis(leftfit,rightfit,scalefactor)
