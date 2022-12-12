import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
class Tools:
    def __init__(self):
        pass
    def scatter_img(self,data):
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y)
        plt.show()
    # 展示图像
    def img_show(self,img, imgname='result'):
        cv2.namedWindow(imgname, 0)
        cv2.imshow(imgname, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    def dealInSignerCiecle(self,img):
        # kernel = np.ones((5, 5), np.uint8)  # 创建全一矩阵，数值类型设置为uint8
        # erosion = cv2.erode(witerpaper2, kernel, iterations=1)  # 腐蚀处理
        # dilation = cv2.dilate(erosion, kernel, iterations=1)  # 膨胀处理
        pass

        return circles
    def splitCoorsToPxPy(contours):
        contours = np.array(contours)

        px = contours[:,0]
        py = contours[:,1]
        return px,py
    def parseImage(self,img):
        pass
    def calDis(self,center1,center2,scalefactor):
        pass



class Pick_threshold:
    def __init__(self,img):
        self.img = img
        self.binarization_type = 0
        self.binarization_value = 115
        self.filter_value = 50
        self.filtercontours = []
    def on_type(self,x):
        model_type = cv2.getTrackbarPos("type", "image")
        value = cv2.getTrackbarPos("value", "image")
        ret, dst = cv2.threshold(self.img, self.binarization_value, 255, self.binarization_type)
        cv2.imshow("image", dst)
    def on_value(self,x):
        self.binarization_type = cv2.getTrackbarPos("type", "image")
        self.binarization_value = cv2.getTrackbarPos("value", "image")
        ret, dst = cv2.threshold(self.img, self.binarization_value, 255, self.binarization_type)
        # ret, dst = cv2.threshold(imgforgetvalue, 100, 255, 0)#for experment
        cv2.imshow("image", dst)

    def on_filter(self,x):
        self.filter_value = cv2.getTrackbarPos("filter", "filter")
        tempcontours = []  # tuple转换为list
        imgforselectfilter = self.img
        for i in range(len(self.filtercontours)):
            if len(self.filtercontours[i]) > self.filter_value:
                tempcontours.append(self.filtercontours[i])
        filtercontours = tuple(tempcontours)
        cv2.drawContours(imgforselectfilter, filtercontours, -1, (0, 0, 255), 3)
        cv2.imshow("filter", imgforselectfilter)
    def getModeltypeAndValue(self):
        # 2.显示图片 创建图片窗体
        cv2.namedWindow("image")
        cv2.imshow("image",self.img)
        # 3.创建滑动条
        cv2.createTrackbar("type", "image", 0, 4, self.on_type)
        cv2.createTrackbar("value", "image", 0, 255, self.on_value)
        cv2.waitKey()
        self.binarization_type = cv2.getTrackbarPos("type", "image")
        self.binarization_value = cv2.getTrackbarPos("value", "image")
        cv2.destroyAllWindows()
        return self.binarization_type, self.binarization_value

    def getfilterfactor(self,filtercontours):
        # 2.显示图片 创建图片窗体
        self.filtercontours = filtercontours
        cv2.namedWindow("select filter factor")
        cv2.imshow("filter", self.img)
        # 3.创建滑动条
        cv2.createTrackbar("filter", "filter", 0, 500, self.on_filter)
        cv2.waitKey()
        self.filter_value = cv2.getTrackbarPos("filter", "filter")
        cv2.destroyAllWindows()
        return self.filter_value
def draw_roi(event, x, y, flags, param):
    img = param[0]
    pts = param[1]
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
        pts.pop()
    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img2, pts[-1], 1, (0, 0, 255), -1)
    if len(pts) > 1:
        # 画线
        if len(pts) == 2:
            # cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            # cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
            cv2.rectangle(img2, pt1=pts[0], pt2=pts[1], color=(255, 0, 0), thickness=2)
        if len(pts) == 4:
            # cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            # cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)$
            cv2.rectangle(img2, pt1=pts[2], pt2=pts[3], color=(255, 0, 0), thickness=2)
    cv2.imshow('image', img2)
    return pts
class ROISelection:
    def __init__(self,img):
        self.img = img
        self.pts =[]
        self.param = [img,self.pts]
        self.coor = []
        self.coutours = []
        self.leftcontour = []
        self.rightcontour = []

    def ROI_select(self):
        value = 2
        if value == 1:
            roi0 = cv2.selectROI(windowName="roi", img=self.img, showCrosshair=True, fromCenter=False)
            cv2.imshow("roi", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            time.sleep(1)
            roi1 = cv2.selectROI(windowName="roi", img=self.img, showCrosshair=True, fromCenter=False)
            cv2.imshow("roi", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif value == 2:
            # 创建图像与窗口并将窗口与回调函数绑定
            cv2.namedWindow('image',0)
            cv2.setMouseCallback('image', draw_roi, param=self.param)
            print("[INFO] 单击左键：选择点，单击右键：删除上一次选择的点，单击中键：确定ROI区域")
            print("[INFO] 按 ESC 退出")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            cv2.destroyAllWindows()

    # 统一的：mouse callback function
    # 作用是事件捕获， draw——roi 就是 onmouse
    # 是将ROI内部的坐标点筛选出来
    def Pick_Contours(self):
        # data = np.array([[0, 0]])
        # for i in range(len(self.contours)):
        #     aa = np.array(self.contours[i])
        #     cc = aa[:, 0]
        #     data = np.append(data, cc, axis=0)
        x0 = np.array(self.pts[0])
        x1 = np.array(self.pts[1])
        x2 = np.array(self.pts[2])
        x3 = np.array(self.pts[3])
        data1 = []
        data = self.contours
        for j in range(len(data)):
            aa = data[j]
            if ((aa<x1).all() and (aa > x0).all()) or ((aa>x2).all() and (aa<x3).all()):
                data1.append(aa)
            if ((aa < x1).all() and (aa > x0).all()):
                self.leftcontour.append(aa)
            if ((aa > x2).all() and (aa < x3).all()):
                self.rightcontour.append(aa)
        data1 = np.array(data1)
        return data1
    def selection_roi(self,contours):
        self.pts = []
        self.img = imutils.resize(self.img)
        self.param = [self.img,self.pts]
        self.contours = contours
        param = self.ROI_select()
        Picked_data = self.Pick_Contours()
        print('selected coors:',self.param[1])
        for i in Picked_data:
            self.coor.append(i.tolist())



if __name__ == '__main__':
    Debugmodel = 2
    if Debugmodel ==1:
        # Debug Pick_value
        filename = 'F:\cv-for-hwglue\Data\\0-2.tif'
        img = cv2.imread(filename, 1)
        model = Pick_threshold(img)
        model.getModeltypeAndValue()
        # model.getfilterfactor()
        print("a=%s\nb = %s\nd = %s\n" % (model.binarization_type,model.binarization_value,model.filter_value))
    elif Debugmodel ==2:
        # Debug ROI_selection
        filename = 'F:\cv-for-hwglue\Data\\0-2.tif'
        img1 = cv2.imread(filename, 1)
        model = ROISelection(img1)
        img1 = model.selection_roi()
        white_paper = np.zeros([img1.shape[0], img1.shape[1], img1.shape[2]], np.uint8)  # 创建全新图像
        selection_roi(white_paper)



