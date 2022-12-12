# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.patches import Ellipse

# shapes = np.genfromtxt("_data.txt")


class CostFunction_circle3:  # from tim
    """Cost function for circle fit (x=[cx,cy,r]. Initialised with points."""

    def __init__(self, pts):
        self.pts = pts

    # def f(self, x):
    #     """Evaluate cost function fitting circle centre (x[0],x[1]) radius x[2]"""
    #     r2 = np.square(self.pts[0, :] - x[0]) + np.square(self.pts[1, :] - x[1])
    #     d = np.square(np.sqrt(r2) - x[2])
    #     return np.sum(d)

    def f(self, x):
        Ri = self.calc_R(*x)
        return Ri - Ri.mean()

    def calc_R(self,xc,yc):
        # return np.sqrt((self.pts[0,:]-xc) ** 2 + (self.pts[1,:]-yc)**2)
        return np.sqrt((self.pts[0]-xc) ** 2 + (self.pts[1]-yc)**2)
class CostFunction_ellipse3:
    """Cost function for ellipse fit (x=[A B C D E]. Initialised with points."""
    def __init__(self, pts):
        self.pts = pts

    # def f_ellipse(self,x):
    #     Ri = self.calc_Ellipse(x)
    #     return Ri

    def calc_Ellipse(self, x1):
        """Evaluate cost function fitting ellipse"""
        # x = self.pts[0, :]
        # y = self.pts[1, :]
        x = self.pts[0]
        y = self.pts[1]
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        x1 = x1[np.newaxis, :]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        # S = np.dot(D.T,D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        aa = np.dot(x1, D.T)
        aa = np.dot(aa, D)
        aa = np.dot(aa, x1.T)
        bb = np.dot(x1, C)
        bb = np.dot(bb, x1.T)
        d = aa - bb
        return np.sum(d)

    # def calc_Ellipse(self, A ,B ,C ,D ,E, F):
    #     """Evaluate cost function fitting ellipse"""
    #     x = self.pts[0, :]
    #     y = self.pts[1, :]
    #     x = x[:, np.newaxis]
    #     y = y[:, np.newaxis]
    #     x1 = np.array([A, B, C, D, E, F])
    #     x1 = x1[np.newaxis, :]
    #     D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    #     # S = np.dot(D.T,D)
    #     C = np.zeros([6, 6])
    #     C[0, 2] = C[2, 0] = 2
    #     C[1, 1] = -1
    #     aa = np.dot(x1, D.T)
    #     aa = np.dot(aa, D)
    #     aa = np.dot(aa, x1.T)
    #     bb = np.dot(x1, C)
    #     bb = np.dot(bb, x1.T)
    #     d = aa - bb
    #     return np.sum(d)

def solve_ellipse(A, B, C, D, E, F):
    Xc = (B * E - 2 * C * D) / (4 * A * C - B ** 2)
    Yc = (B * D - 2 * A * E) / (4 * A * C - B ** 2)

    FA1 = 2 * (A * Xc ** 2 + C * Yc ** 2 + B * Xc * Yc - F)
    FA2 = np.sqrt((A - C) ** 2 + B ** 2)

    MA = np.sqrt(FA1 / (A + C + FA2))
    SMA = np.sqrt(FA1 / (A + C - FA2)) if A + C - FA2 != 0 else 0

    if B == 0 and F * A < F * C:
        Theta = 0
    elif B == 0 and F * A >= F * C:
        Theta = 90
    elif B != 0 and F * A < F * C:
        alpha = np.arctan((A - C) / B) * 180 / np.pi
        Theta = 0.5 * (-90 - alpha) if alpha < 0 else 0.5 * (90 - alpha)
    else:
        alpha = np.arctan((A - C) / B) * 180 / np.pi
        Theta = 90 + 0.5 * (-90 - alpha) if alpha < 0 else 90 + 0.5 * (90 - alpha)

    if MA < SMA:
        MA, SMA = SMA, MA
    return [Xc, Yc, MA, SMA, Theta]

def Cons(x1):
    x1 = x1[np.newaxis, :]
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    bb = np.dot(x1, C)
    bb = np.dot(bb, x1.T)
    return np.sum(bb)-1

# zd 2020/4/12  83-93 from tim
# Set up a cost function to evaluate how well a circle fits to pts2
class func:
    def __init__(self,px,py):
        self.data = np.vstack([px, py])
    def getEllipseParam(self):
        c3 = CostFunction_circle3(self.data)
        px = self.data[0,:]
        py = self.data[1,:]
        x_m = np.mean(px)
        y_m = np.mean(py)
        # center_estimate = np.zeros(2)
        # center_estimate[0] = x_m
        # center_estimate[1] = y_m
        center_estimate = x_m, y_m
        center_2, _ = optimize.leastsq(c3.f,center_estimate)
        xc_2, yc_2 = center_2
        Ri = c3.calc_R(xc_2,yc_2)
        # Initialise a 3 element vector to zero (as a start)
        # x0 = np.zeros(3)
        # x0[0] = np.mean(self.data[0,:])
        # x0[1] = np.mean(self.data[1,:])
        # Use Powell's method to fit, starting at x
        # res = minimize(c3.f, x0, method='Powell')
        # print("Best fit has centre (", res.x[0], ",", res.x[1], ") radius ", res.x[2])
        # plot a circle
        # coc = res
        theta = np.arange(0, 2 * np.pi, 0.01)
        # x1 = res.x[0] + res.x[2] * np.cos(theta)
        # y1 = res.x[1] + res.x[2] * np.sin(theta)
        R_average = Ri.mean()
        x1 = xc_2 + R_average * np.cos(theta)
        y1 = yc_2 + R_average * np.sin(theta)

        plt.title("First shape")
        plt.plot(self.data[0, :], self.data[1, :], "o")
        plt.plot(x1, y1)
        plt.show()
        plt.savefig("circles.png")

        # Ax^2+Bxy+Cy^2+Dx+Ey+F=0
        e3 = CostFunction_ellipse3(self.data)
        px = self.data[0,:]
        py = self.data[1,:]
        x_m = np.mean(px)
        y_m = np.mean(py)
        x2 = np.array([0.01] * 6, dtype='float64')
        center_estimate = x_m, y_m
        # res1,_ = optimize.leastsq(e3.calc_Ellipse, x2)

        cons = (
            {'type':'eq','fun':Cons(x2)}
        )

        res1 = optimize.minimize(e3.calc_Ellipse, x2, method='Powell', constraints= cons)

        [Xc, Yc, MA, SMA, Theta] = solve_ellipse(res1.x[0], res1.x[1], res1.x[2], res1.x[3], res1.x[4], res1.x[5])

        ell = Ellipse([Xc, Yc], MA * 2, SMA * 2, Theta)

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.plot(self.data[0, :],self.data[1, :], "o")
        ax.add_artist(ell)
        ell.set_facecolor("white")
        ell.set_edgecolor("black")
        plt.plot(self.data[0, :], self.data[1, :], "o")
        plt.show()
        plt.savefig("Ellipse.png")
        return Xc, Yc, MA, SMA, Theta,xc_2,yc_2

class RascanEllipse:
    def __init__(self):
        # self.data = np.vstack([px, py])
        # run RASCAN 算法
        # data - 样本点
        # model - 假设模型: 事先自己确定
        # n - 生成模型所需的最少样本点
        # k - 最大迭代次数
        # t - 阈值: 作为判断点满足模型的条件
        # d - 拟合较好时, 需要的样本点最少的个数, 当做阈值看待
        self.k = 500
        self.n = 60
        self.d = 30
        self.t = 50000
        self.iterations = 0
        self.bestfit = None
        self.besterr = np.inf  # 设置默认值
        self.best_inlier_idxs = None
        self.choice = 1
    def fitting(self,LeastSmodel,points_x,points_y,debug=False,return_all=False):
        print("running....")
        data = np.vstack([points_x, points_y]).T
        while self.iterations < self.k:
            # maybe_idxs, test_idxs = self.random_partition(self.n, data.shape[0])
            # maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
            # test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
            # 使用maybe_inliers进行圆的拟合
            # Xc, Yc, MA, SMA, Theta ,A,B,C,D,E = LeastSmodel.getEllipseParam(px=maybe_inliers[:, 0],py=maybe_inliers[:, 1])
            Xc, Yc, MA, SMA, Theta ,A,B,C,D,E = LeastSmodel.getEllipseParam(px=data[:, 0],py=data[:, 1]) # 拟合模型
            # 对拟合出来的椭圆进行误差的计算
            # test_err = self.get_error(A=A,B=B,C=C,D=D,E=E, data=test_points)  # 计算误差:平方和最小
            test_err = self.get_error(A=A,B=B,C=C,D=D,E=E, data=data)  # 计算误差:平方和最小
            # also_idxs = test_idxs[test_err < self.t]
            # also_inliers = data[also_idxs, :]
            thiserr = np.mean(test_err)
            if thiserr < self.besterr:
                self.bestfit = [Xc, Yc, MA, SMA, Theta]
                self.besterr = thiserr
            # if debug:
            #     print('test_err.min()', test_err.min())
            #     print('test_err.max()', test_err.max())
            #     print('numpy.mean(test_err)', np.mean(test_err))
            #     print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
            # if len(also_inliers) > self.d:
            #     betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            #     Xc, Yc, MA, SMA, Theta , A,B,C,D,E = LeastSmodel.getEllipseParam(px=maybe_inliers[:, 0],py=maybe_inliers[:, 1])  # 拟合模型
            #     better_errs = self.get_error(A=A,B=B,C=C,D=D,E=E,data=betterdata)
            #     thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            #     if thiserr < self.besterr:
            #         self.bestfit = [Xc, Yc, MA, SMA, Theta]
            #         self.besterr = thiserr
            #         self.best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
            self.iterations += 1
        if self.bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return self.bestfit, {'inliers': best_inlier_idxs}
        else:
            return self.bestfit

    def getEllipseParam(self,px,py):
        c3 = CostFunction_circle3([px,py])
        # px = self.data[0,:]
        # py = self.data[1,:]
        x_m = np.mean(px)
        y_m = np.mean(py)
        # center_estimate = np.zeros(2)
        # center_estimate[0] = x_m
        # center_estimate[1] = y_m
        center_estimate = x_m, y_m
        center_2, _ = optimize.leastsq(c3.f,center_estimate)
        xc_2, yc_2 = center_2
        Ri = c3.calc_R(xc_2,yc_2)
        # Initialise a 3 element vector to zero (as a start)
        # x0 = np.zeros(3)
        # x0[0] = np.mean(self.data[0,:])
        # x0[1] = np.mean(self.data[1,:])
        # Use Powell's method to fit, starting at x
        # res = minimize(c3.f, x0, method='Powell')
        # print("Best fit has centre (", res.x[0], ",", res.x[1], ") radius ", res.x[2])
        # plot a circle
        # coc = res
        theta = np.arange(0, 2 * np.pi, 0.01)
        # x1 = res.x[0] + res.x[2] * np.cos(theta)
        # y1 = res.x[1] + res.x[2] * np.sin(theta)
        R_average = Ri.mean()
        x1 = xc_2 + R_average * np.cos(theta)
        y1 = yc_2 + R_average * np.sin(theta)

        # plt.title("First shape")
        # # plt.plot(self.data[0, :], self.data[1, :], "o")
        # plt.plot(px, py, "o")
        # plt.plot(x1, y1)
        # plt.show()
        # plt.savefig("circles.png")

        # Ax^2+Bxy+Cy^2+Dx+Ey+F=0
        e3 = CostFunction_ellipse3([px,py])
        # px = self.data[0,:]
        # py = self.data[1,:]
        x_m = np.mean(px)
        y_m = np.mean(py)
        x2 = np.array([0.01] * 6, dtype='float64')
        center_estimate = x_m, y_m
        # res1,_ = optimize.leastsq(e3.calc_Ellipse, x2)

        cons = (
            {'type':'eq','fun':Cons(x2)}
        )

        res1 = optimize.minimize(e3.calc_Ellipse, x2, method='Powell', constraints= cons)

        [Xc, Yc, MA, SMA, Theta] = solve_ellipse(res1.x[0], res1.x[1], res1.x[2], res1.x[3], res1.x[4], res1.x[5])

        # ell = Ellipse([Xc, Yc], MA * 2, SMA * 2, Theta)
        #
        # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        # # plt.plot(self.data[0, :],self.data[1, :], "o")
        # plt.plot(px,py, "o")
        # ax.add_artist(ell)
        # ell.set_facecolor("white")
        # ell.set_edgecolor("black")
        # # plt.plot(self.data[0, :], self.data[1, :], "o")
        # plt.plot(px, py, "o")
        # plt.show()
        # plt.savefig("Ellipse.png")
        return Xc, Yc, MA, SMA, Theta , res1.x[0], res1.x[1], res1.x[2], res1.x[3], res1.x[4]

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
            # B = (d[0])**2 + A*d[1]*d[2] + B*d[2]**2 + C*d[1] + D*d[2] +E
            F = A*d[0]**2 + B*d[0]*d[1] + C*d[1]**2 + D*d[0] + E*d[1]
            err_per_point.append((F) ** 2)  # sum squared error per row
        return np.array(err_per_point)
