import cv2
import random
import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize
class findEdgeContours:
    def __init__(self,img):
        self.img = img
        self.contours = []
        self.contours_filter()
    def contours_filter(self, filterthresh=50):
        # 返回带轮廓的图像
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);
        tempcontours = []
        # tuple转换为list
        for i in range(len(contours)):
            if len(contours[i]) > filterthresh:
                temp = contours[i]
                temp = temp.reshape(-1, 2)
                for j in range(temp.__len__()):
                    # temp = temp[j].tolist()
                    tempcontours.append(temp[j].tolist())
        self.contours = tempcontours
class HufuMethod:
    def __init__(self,img):
        self.img = img
        self.COC = []
    def houghCircle(self,img):
        # circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=100, minRadius=60,maxRadius=150)
        # TODO 建立一个循环，动态调节参数，直到圆心输出符合要求
        # 把半径范围缩小点，检测内圆，瞳孔
        circle1 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=100)

        # 把半径范围缩小点，检测内圆，瞳孔 @param dp = 1
        circles = circle1[0, :, :]  # 提取为二维
        circles = np.uint16(np.around(circles))  # 四舍五入，取整
        self.COC = []
        for i in circles[:]:
            # cv2.circle(whitepaper1, (i[0], i[1]), i[2], (255, 0, 0), 1)  # 画圆
            # cv2.circle(whitepaper1, (i[0], i[1]), 3, (0, 255, 0), -1)  # 画圆心
            print("the center of Concentric circles is : %f     %f " % (i[0], i[1]))
            self.COC.append([i[0], i[1]])
        print('aa')
class Canny:
    def __init__(self,img):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Guassian_kernal_size = 3  # 获取高斯核尺寸
        self.img = img  # 获取图片
        self.y, self.x = img.shape[0:2]  # 获取图片的维度
        self.angle = np.zeros([self.y, self.x])  # 定义一个新的矩阵，用来存储梯度方向
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])  # x方向偏导卷积核
        self.y_kernal = np.array([[-1], [1]])  # y方向偏导卷积核
        self.HT_high_threshold = 45  # 高门限
        self.HT_low_threshold = 25  # 低门限

    def Get_gradient_img(self):
        '''
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print('Get_gradient_img')

        new_img_x = np.zeros([self.y, self.x], dtype=np.float)  # 定义x偏导存储矩阵
        new_img_y = np.zeros([self.y, self.x], dtype=np.float)  # 定义y偏导存储矩阵
        for i in range(0, self.x):
            for j in range(0, self.y):
                if j == 0:
                    new_img_y[j][i] = 1
                else:
                    new_img_y[j][i] = np.sum(np.array([[self.img[j - 1][i]], [self.img[j][i]]]) * self.y_kernal)
                    # y方向卷积后的灰度图，dy
                if i == 0:
                    new_img_x[j][i] = 1
                else:
                    new_img_x[j][i] = np.sum(np.array([self.img[j][i - 1], self.img[j][i]]) * self.x_kernal)
                    # x方向卷积后的灰度图，dx

        gradient_img, self.angle = cv2.cartToPolar(new_img_x, new_img_y)  # 通过dx，dy求取梯度强度和梯度方向即角度
        self.angle = np.tan(self.angle)
        self.img = gradient_img.astype(np.uint8)  # 将获得梯度强度转换成无符号八位整形
        return self.img

    def Non_maximum_suppression(self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print('Non_maximum_suppression')

        result = np.zeros([self.y, self.x])  # 定义一个新矩阵，用来存储非极大化抑制结果图
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if abs(self.img[i][j]) <= 4:
                    result[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:  # dy>dx
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    # g1 g2
                    #    C
                    #    g4 g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #    g2 g1
                    #    C
                    # g3 g4
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                else:
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    # g1
                    # g2 C g4
                    #      g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #      g3
                    # g2 C g4
                    # g1
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]

                temp1 = abs(self.angle[i][j]) * gradient1 + (1 - abs(self.angle[i][j])) * gradient2
                temp2 = abs(self.angle[i][j]) * gradient3 + (1 - abs(self.angle[i][j])) * gradient4
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
        self.img = result
        return self.img

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print('Hysteresis_thresholding')

        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        # g1 g2
                        #    C
                        #    g4 g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #    g2 g1
                        #    C
                        # g3 g4
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        # g1
                        # g2 C g4
                        #      g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #      g3
                        # g2 C g4
                        # g1
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
        return self.img

    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img



class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''
        img: 输入的边缘图像
        angle: 输入的梯度方向矩阵
        step: Hough 变换步长大小
        threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        # 投票矩阵，将原图的宽和高，半径分别除以步长，向上取整，
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单元进行投票。每个点投出来结果为一折线。
        return:  投票矩阵
        '''
        print('Hough_transform_algorithm')

        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0:
                    # 沿梯度正方向投票
                    y = i
                    x = j
                    r = 0
                    while y < self.y and x < self.x and y >= 0 and x >= 0: # 保证圆心在图像内
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        # 为了避免 y / self.step 向上取整会超出矩阵索引的情况，这里将该值向下取整
                        y = y + self.step * self.angle[i][j]
                        x = x + self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    # 沿梯度反方向投票
                    y = i - self.step * self.angle[i][j]
                    x = j - self.step
                    r = math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y - self.step * self.angle[i][j]
                        x = x - self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
        return self.vote_matrix   # 返回投票矩阵

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制，这里的非极大化抑制我采
        用的是邻近点结果取平均值的方法，而非单纯的取极大值。
        return: None
        '''
        print('Select_Circle')
		# 挑选投票数大于阈值的圆
        houxuanyuan = []
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i * self.step + self.step / 2   # 通过投票矩阵中的点，恢复到原图中的点，self.step / 2为补偿值
                        x = j * self.step + self.step / 2
                        r = r * self.step + self.step / 2
                        houxuanyuan.append((math.ceil(x), math.ceil(y), math.ceil(r)))
        if len(houxuanyuan) == 0:
            print("No Circle in this threshold.")
            return
        x, y, r = houxuanyuan[0]
        possible = []
        middle = []
        for circle in houxuanyuan:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                # 设定一个误差范围（这里设定方圆20个像素以内，属于误差范围），在这个范围内的到圆心视为同一个圆心
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)  # 对同一范围内的圆心，半径取均值
                middle.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)  # 将最后一组同一范围内的圆心，半径取均值
        middle.append((result[0], result[1], result[2]))  # 误差范围内的圆取均值后，放入其中

        def takeFirst(elem):
            return elem[0]

        middle.sort(key=takeFirst)  # 排序
        # 重复类似上述取均值的操作，并将圆逐个输出
        x, y, r = middle[0]
        possible = []
        for circle in middle:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)
                print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)
        print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
        self.circles.append((result[0], result[1], result[2]))

    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles

class LeastSquares:
    def __init__(self,px=[],py=[]):
        self.px = px
        self.py = py
    def getCircleCenter(self,px,py):
        method_2 = "Fitting Circle"
        # Coordinates of the 2D points
        basename = 'arc'
        # 质心坐标
        self.px = px
        self.py = py
        x_m = np.mean(self.px)
        y_m = np.mean(self.py)
        # print(x_m, y_m)
        # 圆心估计
        center_estimate = x_m, y_m
        center_2, _ = optimize.leastsq(self.f_2, center_estimate)
        xc_2, yc_2 = center_2
        Ri = self.calc_R(xc_2, yc_2)
        # 拟合圆的半径
        R_average = Ri.mean()
        residu_2 = sum((Ri - R_average) ** 2)
        residu2_2 = sum((Ri ** 2 - R_average ** 2) ** 2)
        R3 = R_average ** 2
        # ncalls_2   = f_2.ncalls
        # print(xc_2, yc_2,R3)
        return [xc_2, yc_2, R3]
    # @countcalls
    def f_2(self,c):
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()
    def calc_R(self,xc, yc):
        return np.sqrt((self.px - xc) ** 2 + (self.py - yc) ** 2)
    def getEllipseParam(self,px,py):
        x2 = np.array([0.01] * 6, dtype='float64')
        # res1 = minimize(self.f_ellipse, x2, method='Powell')
        self.px = px
        self.py = py
        x_m = np.mean(self.px)
        y_m = np.mean(self.py)
        # print(x_m, y_m)
        # 圆心估计
        center_estimate = x_m, y_m
        res1 = optimize.leastsq(self.f_ellipse, center_estimate)
        return res1.x[0], res1.x[1], res1.x[2], res1.x[3], res1.x[4], res1.x[5]
    def f_ellipse(self,c):
        Ri = self.calc_Ellipse(c)
    def calc_Ellipse(self,x1):
        """Evaluate cost function fitting ellipse"""
        x = self.px
        y = self.py
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        x1 = x1[np.newaxis, :]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        # S = np.dot(D.T,D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2;
        C[1, 1] = -1
        aa = np.dot(x1, D.T)
        aa = np.dot(aa, D)
        aa = np.dot(aa, x1.T)
        bb = np.dot(x1, C)
        bb = np.dot(bb, x1.T)
        d = aa - bb
        return np.sum(d)
class Debug:
    def __init__(self):
        # Coordinates of the 2D points
        self.true_x = 5
        self.true_y = 5
        self.true_r = 10
        self.points_num = 150
        self.inline_points = 130
        self.points_x = []
        self.points_y = []
    def generateTestData(self):
        '''生成随机点， 作为数据源'''
        for i in range(1, self.points_num):
            # print(random.randint(-5, 15))
            a = np.random.uniform(self.true_x - self.true_r, self.true_x + self.true_r)
            x = round(a, 2)
            up_down = random.randrange(-1, 2, 2)
            y = self.true_y + up_down * math.sqrt(self.true_r ** 2 - (x - self.true_x) ** 2)
            b1 = np.random.uniform(-0.05, 0.05)
            b2 = np.random.uniform(-0.05, 0.05)
            c1 = np.random.uniform(0.3, 0.8)
            c2 = np.random.uniform(0.3, 0.8)
            error_b1 = round(b1, 2)
            error_b2 = round(b1, 2)
            error_c1 = round(c1, 2)
            error_c2 = round(c2, 2)
            if i <= self.inline_points:
                self.points_x.append(x + error_b1)
                self.points_y.append(y + error_b2)
            else:
                up_down1 = random.randrange(-1, 2, 2)
                self.points_x.append(x + error_c1)
                self.points_y.append(y + error_c2)
    def plotFig(self):
        plt.plot(self.points_x, self.points_y, "ro", label="data points")
        plt.axis('equal')
        plt.show()
class Ransac:
    def __init__(self):
        # run RANSAC 算法
        # data - 样本点
        # model - 假设模型: 事先自己确定
        # n - 生成模型所需的最少样本点
        # k - 最大迭代次数
        # t - 阈值: 作为判断点满足模型的条件
        # d - 拟合较好时, 需要的样本点最少的个数, 当做阈值看待
        self.k = 5000
        self.n = 100
        self.d = 100
        self.t = 500000
        self.iterations = 0
        self.bestfit = None
        self.besterr = np.inf  # 设置默认值
        self.best_inlier_idxs = None
        self.choice = 1
    def fitting(self, LeastSmodel, points_x, points_y, debug=False, return_all=False):
        """
        参考:http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
        伪代码:http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
        输入:
            data - 样本点
            model - 假设模型:事先自己确定
            n - 生成模型所需的最少样本点
            k - 最大迭代次数
            t - 阈值:作为判断点满足模型的条件
            d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        输出:
            bestfit - 最优拟合解（返回nil,如果未找到）
        iterations = 0
        bestfit = nil #后面更新
        besterr = something really large #后期更新besterr = thiserr
        while iterations < k
        {
            maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
            maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
            alsoinliers = emptyset #满足误差要求的样本点,开始置空
            for (每一个不是maybeinliers的样本点)
            {
                if 满足maybemodel即error < t
                    将点加入alsoinliers
            }
            if (alsoinliers样本点数目 > d)
            {
                %有了较好的模型,测试模型符合度
                bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
                thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
                if thiserr < besterr
                {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            iterations++
        }
        return bestfit
        """
        print("running....")
        data = np.vstack([points_x, points_y]).T
        while self.iterations < self.k:
            maybe_idxs, test_idxs = self.random_partition(self.n, data.shape[0])
            maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
            test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
            # 使用maybe_inliers进行圆的拟合

            CircleCenterX, CircleCenterY, CircleCenterR2 = LeastSmodel.getCircleCenter(px=maybe_inliers[:, 0],
                                                                                         py=maybe_inliers[:, 1])  # 拟合模型
            # 对拟合出来的圆进行误差的计算
            test_err = self.get_error(CircleCenterX=CircleCenterX, CircleCenterY=CircleCenterY,
                                 CircleCenterR2=CircleCenterR2, data=test_points)  # 计算误差:平方和最小
            also_idxs = test_idxs[test_err < self.t]
            also_inliers = data[also_idxs, :]
            if debug:
                print('test_err.min()', test_err.min())
                print('test_err.max()', test_err.max())
                print('numpy.mean(test_err)', np.mean(test_err))
                print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
            if len(also_inliers) > self.d:
                betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
                CircleCenterX, CircleCenterY, CircleCenterR2 = LeastSmodel.getCircleCenter(px=betterdata[:, 0],
                                                                                             py=betterdata[:, 1])
                better_errs = self.get_error(CircleCenterX=CircleCenterX, CircleCenterY=CircleCenterY,
                                        CircleCenterR2=CircleCenterR2, data=betterdata)
                thiserr = np.mean(better_errs)  # 平均误差作为新的误差
                if thiserr < self.besterr:
                    self.bestfit = [CircleCenterX, CircleCenterY, CircleCenterR2]
                    self.besterr = thiserr
                    self.best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
            self.iterations += 1
            if self.bestfit is None:
                raise ValueError("did't meet fit acceptance criteria")
            if return_all:
                return self.bestfit, {'inliers': best_inlier_idxs}
            else:
                return self.bestfit

    def random_partition(self,n, n_data):
        """return n random rows of data and the other len(data) - n rows"""
        all_idxs = np.arange(n_data)  # 获取n_data下标索引
        np.random.shuffle(all_idxs)  # 打乱下标索引
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

    '''最小二乘拟合直线模型'''
    def get_error(self,CircleCenterX, CircleCenterY, CircleCenterR2, data):
        err_per_point = []
        for d in data:
            B = (d[0] - CircleCenterX) ** 2 + (d[1] - CircleCenterY) ** 2 - CircleCenterR2
            err_per_point.append((B) ** 2)  # sum squared error per row
        return np.array(err_per_point)

class RansacEllipse:
    def __init__(self):
        # run RANSAC 算法
        # data - 样本点
        # model - 假设模型: 事先自己确定
        # n - 生成模型所需的最少样本点
        # k - 最大迭代次数
        # t - 阈值: 作为判断点满足模型的条件
        # d - 拟合较好时, 需要的样本点最少的个数, 当做阈值看待
        self.k = 5000
        self.n = 60
        self.d = 30
        self.t = 50000
        self.iterations = 0
        self.bestfit = None
        self.besterr = np.inf  # 设置默认值
        self.best_inlier_idxs = None
        self.choice = 1
    def fitting(self, LeastSmodel, points_x, points_y, debug=False, return_all=False):
        """
        参考:http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
        伪代码:http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
        输入:
            data - 样本点
            model - 假设模型:事先自己确定
            n - 生成模型所需的最少样本点
            k - 最大迭代次数
            t - 阈值:作为判断点满足模型的条件
            d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        输出:
            bestfit - 最优拟合解（返回nil,如果未找到）
        iterations = 0
        bestfit = nil #后面更新
        besterr = something really large #后期更新besterr = thiserr
        while iterations < k
        {
            maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
            maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
            alsoinliers = emptyset #满足误差要求的样本点,开始置空
            for (每一个不是maybeinliers的样本点)
            {
                if 满足maybemodel即error < t
                    将点加入alsoinliers
            }
            if (alsoinliers样本点数目 > d)
            {
                %有了较好的模型,测试模型符合度
                bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
                thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
                if thiserr < besterr
                {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            iterations++
        }
        return bestfit
        """
        print("running....")
        data = np.vstack([points_x, points_y]).T
        while self.iterations < self.k:
            maybe_idxs, test_idxs = self.random_partition(self.n, data.shape[0])
            maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
            test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
            # 使用maybe_inliers进行圆的拟合
            A,B,C,D,E = LeastSmodel.getEllipseParam(px=maybe_inliers[:, 0],py=maybe_inliers[:, 1])  # 拟合模型
            # 对拟合出来的椭圆进行误差的计算
            test_err = self.get_error(A=A,B=B,C=C,D=D,E=E, data=test_points)  # 计算误差:平方和最小
            also_idxs = test_idxs[test_err < self.t]
            also_inliers = data[also_idxs, :]
            if debug:
                print('test_err.min()', test_err.min())
                print('test_err.max()', test_err.max())
                print('numpy.mean(test_err)', np.mean(test_err))
                print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
            if len(also_inliers) > self.d:
                betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
                A,B,C,D,E = LeastSmodel.getEllipseParam(px=maybe_inliers[:, 0],py=maybe_inliers[:, 1])  # 拟合模型
                better_errs = self.get_error(A=A,B=B,C=C,D=D,E=E,data=betterdata)
                thiserr = np.mean(better_errs)  # 平均误差作为新的误差
                if thiserr < self.besterr:
                    self.bestfit = [A,B,C,D,E]
                    self.besterr = thiserr
                    self.best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
            self.iterations += 1
            if self.bestfit is None:
                raise ValueError("did't meet fit acceptance criteria")
            if return_all:
                return self.bestfit, {'inliers': best_inlier_idxs}
            else:
                return self.bestfit

    def random_partition(self,n, n_data):
        """return n random rows of data and the other len(data) - n rows"""
        all_idxs = np.arange(n_data)  # 获取n_data下标索引
        np.random.shuffle(all_idxs)  # 打乱下标索引
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

    '''最小二乘拟合直线模型'''
    def get_error(self,A,B,C,D,E,data):
        err_per_point = []
        for d in data:
            B = (d[0])**2 + A*d[1]*d[2] + B*d[2]**2 + C*d[1] + D*d[2] +E
            err_per_point.append((B) ** 2)  # sum squared error per row
        return np.array(err_per_point)

if __name__ == '__main__':
    Datamodel = Debug()
    Datamodel.generateTestData()
    # Datamodel.plotFig()
    Model = LeastSquares()
    Model.getCircleCenter(Datamodel.points_x,Datamodel.points_y)
    RsnsacModel = Ransac()
    RsnsacModel.fitting(Model, Datamodel.points_x, Datamodel.points_y)



