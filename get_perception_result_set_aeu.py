#!/usr/bin/env python
from __future__ import print_function
#########################
import sys
from typing import DefaultDict
import rospy
import cv2
import numpy
from std_msgs.msg import String, Float32MultiArray,Float32
from sensor_msgs.msg import Image
from beginner_tutorials.msg import egoinfo
from beginner_tutorials.msg import action_expect_utility_msg
from opencv_apps.msg import RectArrayStamped
from cv_bridge import CvBridge, CvBridgeError
import math
import numpy as np

offset=0.0
xe=0.0
ye=0.0
yawe=0.0
max_distance=0
pub=rospy.Publisher("/aeu", action_expect_utility_msg, queue_size=10)
def callback(data):
    global max_distance
    
    probability_of_each_lane_for_all_car=[]
    pro_dangerous_all_obstacle=[]
    min_distance_of_lanes=[1000,1000,1000,1000]
    for i in range(9, len(data.data),14):  # 遍历每一个障碍物，这是对一个障碍物的处理。
        # xj yj ， 现在得到的xj和yj就是全局坐标系下的
        xj=data.data[i]
        yj=-data.data[i+1]
        lj=yj/3.5   # which lane?
        dj=data.data[i-6]**2/3

        if True: # 得到 不同车道上，距离 自车 最近的车 的距离
            idx=int(yj // 3.5)
            if idx<0 :   idx=0
            if idx>3:  idx=3
            if xj < min_distance_of_lanes[idx]:
                min_distance_of_lanes[idx]=xj
            

        if True:  # 计算 一个车子 在 不同车道上的 概率
            probability_of_each_lane_for_one_car=[]  # 计算车在不同车道上的概率。
            """
            为什么 计算 K， 因为 为了满足 似然函数的 概率之和 为1，所以 这个高斯分布不能是原来的高斯分布，要进行放缩才可以使用
            """
            k=1/(1+math.sqrt(2*math.pi*dj) )  # 这个k是高斯分布的 按比例缩减之后的 系数
            for lanei in range(0,4): 
                """
                计算在  每个 lane的似然函数下的 概率
                """
                if lj<= lanei:
                    probability_of_each_lane_for_one_car.append(math.exp(-(lj-lanei)**2/2/dj)*k)  # 在左侧的高斯部分的概率
                elif lj>=lanei+1:
                    probability_of_each_lane_for_one_car.append( math.exp(-(lj-lanei-1)**2/2/dj)*k) # 在右侧高斯部分的概率
                else :
                    probability_of_each_lane_for_one_car.append(k)       # 中间均匀分布的概率

            total=sum(probability_of_each_lane_for_one_car)  
            # 计算他们的和，  每个lane的似然函数下有一个值，计算他们的和
            probability_of_each_lane_for_one_car=numpy.array(probability_of_each_lane_for_one_car)
            probability_of_each_lane_for_one_car=probability_of_each_lane_for_one_car/total  
            # 这才是自车在不同车道上的概率，是一个后验概率
            # 得到自车在不同车道上的后验概率！！！ [p1, p2, p3, p4] --------------------------------------------------------------------------  计算 得到 车 在 不同车道上的 概率

        probability_of_each_lane_for_all_car.append(probability_of_each_lane_for_one_car)  #[[. . . .], [. . . .], [. . . .], ...]
        # 汇总
        
        if True : #  根据相对距离 计算 安全 和 危险的 概率   ， 这里计算的是一个 车子 相对 自车 的 安全性
            distance=xj-xe
            if distance>max_distance:
                max_distance=distance
                ##print("this is max distance !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print(max_distance)
            else:
                pass
                #print("max distance %f" % max_distance)
            sigma_distance=data.data[i-7]*10   # uncertainty larger
            distance_yu1=1.0
            distance_yu2=10.0
            k1=1/(distance_yu1+0.5*math.sqrt(2*math.pi) * sigma_distance)
            k2=1/(15-distance_yu2+0.5*math.sqrt(2*math.pi) * sigma_distance)
            dan=0
            freee=0
            if distance<=5:
                dan=k1
                dan=dan*50
            else :
                dan=math.exp(-(distance-distance_yu1)**2/2/sigma_distance**2) *k1
                dan=dan*50
            if distance>=10:
                freee=k2
            else :
                freee=math.exp(-(distance-distance_yu2)**2/2/sigma_distance**2) *k2
        
        pro_dangerous_all_obstacle.append(distance)
        pro_dangerous_all_obstacle.append(dan/(dan+freee))   
        # 汇总
    
    probability_of_each_lane_for_all_car=list(probability_of_each_lane_for_all_car)

    for i in range(len(min_distance_of_lanes)):
        min_distance_of_lanes[i]-=xe 
    # 得到 相对 距离

    if True:  # 根据 之前 计算 的 所有 车子 在 不同车道上的 概率  和 安全的概率 ，计算 车道 的 安全情况  
        # 最后 得到 的 数据 的 个数  就是 车道 的 个数
        lane_dan_pro=[]
        lane_free_pro=[]
        for i_lane in range(4):
            pro_i_dan=0
            liancheng_pro=1
            for j in range(len(probability_of_each_lane_for_all_car)):
                liancheng_pro=liancheng_pro*(1-probability_of_each_lane_for_all_car[j][i_lane] * pro_dangerous_all_obstacle[2*j+1])
                #pro_i_dan=1-liancheng_pro
                pass
            pro_i_dan=1-liancheng_pro
            lane_dan_pro.append(pro_i_dan)
            lane_free_pro.append(1-pro_i_dan)
    
    #这里计算四个车道的安全状态。
    """
    最后得到的数据量 就是  车道的个数，但是在实际上使用的时候，只使用三个车道，自车道和左右车道
    三个车道，是确定的。 
    车道状态 可以是多个  x个     这样总的状态数量 就是 power(x，3)
    """

    utility_value=[[0.0,0.0,0.0,0.0,30.0,30.0,30.0,30.0] , [20.0,20.0,50.0,50.0,20.0,20.0,50.0,50.0], [0.0,30.0,0.0,30.0,0.0,30.0,0.0,30.0] ] # left, keep ,right 三个路，每个路有两种情况，所以一共是八种情况。
    utility_value=np.array(utility_value)
    action_expect_utility=[0.0, 0.0, 0.0]
    env_pro=[]
    #  ego car on which lane?
    if ye<=3.5:  # 1 lane
        DDD_pro=lane_dan_pro[0]*lane_dan_pro[1]
        DDF_pro=lane_dan_pro[0]*lane_free_pro[1]
        DFD_pro=lane_free_pro[0]*lane_dan_pro[1]
        DFF_pro=lane_free_pro[0]*lane_free_pro[1]
        FDD_pro=lane_free_pro[0]*lane_dan_pro[1]*0.0
        FDF_pro=lane_free_pro[0]*lane_dan_pro[1]*0.0
        FFD_pro=lane_free_pro[0]*lane_free_pro[1]*lane_dan_pro[2]*0.0
        FFF_pro=lane_free_pro[0]*lane_free_pro[1]*lane_free_pro[2]*0.0

        utility_value[1,:]+=min_distance_of_lanes[0]
        utility_value[2,:]+=min_distance_of_lanes[1]
        
        pass
    elif ye<=7.0:  # 2lane
        DDD_pro=lane_dan_pro[0]*lane_dan_pro[1]*lane_dan_pro[2]
        DDF_pro=lane_dan_pro[0]*lane_dan_pro[1]*(1-lane_dan_pro[2])
        DFD_pro=lane_dan_pro[0]*(1-lane_dan_pro[1])*lane_dan_pro[2]
        DFF_pro=lane_dan_pro[0]*lane_free_pro[1]*lane_free_pro[2]
        FDD_pro=lane_free_pro[0]*lane_dan_pro[1]*lane_dan_pro[2]
        FDF_pro=lane_free_pro[0]*lane_dan_pro[1]*lane_free_pro[2]
        FFD_pro=lane_free_pro[0]*lane_free_pro[1]*lane_dan_pro[2]
        FFF_pro=lane_free_pro[0]*lane_free_pro[1]*lane_free_pro[2]
        
        utility_value[0,:]+=min_distance_of_lanes[0]
        utility_value[1,:]+=min_distance_of_lanes[1]
        utility_value[2,:]+=min_distance_of_lanes[2]
    elif ye <=10.5 :
        DDD_pro=lane_dan_pro[1]*lane_dan_pro[2]*lane_dan_pro[3]
        DDF_pro=lane_dan_pro[1]*lane_dan_pro[2]*(1-lane_dan_pro[3])
        DFD_pro=lane_dan_pro[1]*(1-lane_dan_pro[2])*lane_dan_pro[3]
        DFF_pro=lane_dan_pro[1]*lane_free_pro[2]*lane_free_pro[3]
        FDD_pro=lane_free_pro[1]*lane_dan_pro[2]*lane_dan_pro[3]
        FDF_pro=lane_free_pro[1]*lane_dan_pro[2]*lane_free_pro[3]
        FFD_pro=lane_free_pro[1]*lane_free_pro[2]*lane_dan_pro[3]
        FFF_pro=lane_free_pro[1]*lane_free_pro[2]*lane_free_pro[3]
        
        utility_value[0,:]+=min_distance_of_lanes[1]
        utility_value[1,:]+=min_distance_of_lanes[2]
        utility_value[2,:]+=min_distance_of_lanes[3]
    else:  # 4 lane
        DDD_pro=lane_dan_pro[2]*lane_dan_pro[3]
        DDF_pro=lane_dan_pro[2]*(1-lane_dan_pro[3])*0.0
        DFD_pro=lane_dan_pro[2]*lane_free_pro[3]
        DFF_pro=lane_free_pro[2]*lane_free_pro[3]*0.0
        FDD_pro=lane_free_pro[2]*lane_dan_pro[3]
        FDF_pro=0.0*lane_free_pro[1]*lane_dan_pro[2]*lane_free_pro[3]
        FFD_pro=lane_free_pro[2]*lane_free_pro[3]
        FFF_pro=0.0*lane_free_pro[1]*lane_free_pro[2]*lane_free_pro[3]
        
        utility_value[0,:]+=min_distance_of_lanes[2]
        utility_value[1,:]+=min_distance_of_lanes[3]

    env_pro.append(DDD_pro)
    env_pro.append(DDF_pro)
    env_pro.append(DFD_pro)
    env_pro.append(DFF_pro)
    env_pro.append(FDD_pro)
    env_pro.append(FDF_pro)
    env_pro.append(FFD_pro)
    env_pro.append(FFF_pro)
    # 这里得到每一个环境的概率。

    for i in range(len(action_expect_utility)):  #  某一个 action
        for j in range(len(env_pro)):  # 遍历 所有的 状态 
            action_expect_utility[i]=action_expect_utility[i]+utility_value[i][j]*env_pro[j]
    
    # 根据每一个动作，和每一个环境，计算期望效用
    action_expect_utility_for_publish=action_expect_utility_msg()
    action_expect_utility_for_publish.left=action_expect_utility[0]
    action_expect_utility_for_publish.keep=action_expect_utility[1]
    action_expect_utility_for_publish.right=action_expect_utility[2]
    pub.publish(action_expect_utility_for_publish)

    
def callback2(data):
    #print(data)
    #print(data.x)
    #print("this is ego x and y :(%f  %f )" % (data.x, data.y))
    global xe
    global ye
    global yawe
    xe=data.x
    ye=data.y
    yawe=data.yaw
    global offset
    if yawe==0:
        if ye-10.5>=0:
            offset=ye-10.5
        elif ye-7>=0:
            offset=ye-7
        elif ye-3.5>=0:
            offset=ye-3.5
        else:
            offset=ye
    else:    # yawe!=0  
        if ye>=0 and ye<=3.5:
            y_edge=0.0
            x_edge=xe+(ye-y_edge)/(math.tan(yawe/180*math.pi))
            offset=math.sqrt((xe-x_edge)**2+(y_edge-ye)**2)
        elif ye<=7.0:
            y_edge=3.5
            x_edge=xe+(ye-y_edge)/(math.tan(yawe/180*math.pi))
            offset=math.sqrt((xe-x_edge)**2+(y_edge-ye)**2)
        elif ye<=10.5:
            y_edge=7.0
            x_edge=xe+(ye-y_edge)/(math.tan(yawe/180*math.pi))
            offset=math.sqrt((xe-x_edge)**2+(y_edge-ye)**2)
        else:
            y_edge=10.5
            x_edge=xe+(ye-y_edge)/(math.tan(yawe/180*math.pi))
            offset=math.sqrt((xe-x_edge)**2+(y_edge-ye)**2)


#!！!！!！!！!
# # 计算 位置，角度 和 offset ，offset是计算 他车的 全局 坐标 时使用的，现在的检测网络直接给出的就是全局坐标，所以这个好像目前用不到


def listener():

    rospy.init_node('get_info', anonymous=True)
    print("spining ...")
    rospy.Subscriber("/perception_node/un_array", Float32MultiArray, callback)
  #  rospy.Subscriber("/egoyaw", Float32,callback1)
    rospy.Subscriber("/egoinfo",egoinfo,callback2)
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()
    rospy.spin()
