#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """

import time
from time import gmtime,strftime
import rospy
import copy
import numpy as np
import cv2
import sys
import math
import os
import logging
import socket
import matplotlib.pyplot as plt
import tf
import carla
import random
from mpi4py import MPI
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque
from planner.map import CarlaMap
import torch
import torch.nn as nn
import scipy.misc as m


# from network.bottom_network import MobileNetv2
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf2

from planner_hxq.planner import Planner

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, Image, PointCloud2
from carla_msgs.msg import CarlaCollisionEvent, CarlaEgoVehicleInfo, CarlaLaneInvasionEvent
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
from std_msgs.msg import Int32
import sensor_msgs.point_cloud2 as pc2

bridge = CvBridge()
road = [[15, 14, 13, 9, 8, 7, 1, 0], [12, 10, 11], [4, 5, 6, 18, 19]]


class CarlaEnv1():
    def __init__(self, index, bottom_policy):
        self.bottom_points = None

        self.bottom_policy = bottom_policy
	self.beam_mum = 512
        self.img = None
        self.img2 = None
        self.img3 = None
	self.img11 = None
	self.img21 = None
	self.img31 = None
        self.map = cv2.imread('02.png')
        self.carla_map = CarlaMap('Town01', 0.1653, 50)
        self.is_crashed = None
        self.crossed_lane_markings = None
        self.speed = None
        self.index = index
        self.car_id = None
        self.lane = None
        self.count = 0
        self.goal_point = None
        # self.off_road_time = deque([0,0,0])
        self.last_road_id = 0
        self.last_right_flag = 0
        self.last_direction = 0
        self.host = rospy.get_param('/carla/host', '127.0.0.1')
        self.port = rospy.get_param('/carla/port', '2000')
        client = carla.Client(self.host, self.port)
        client.set_timeout(2000.0)
        self.world = client.get_world()
        self.target_transform = None
        self.planner= None
        self.vehicle = None
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.last_dis = 0
        self.replanning = False
        self.Flag = False
        self.recount=0
        self.steering = 0
        self.velocity = 0
        self.com = 0

        node_name = 'CarlaEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)
        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = '/carla/ego' + str(index) + '/twist_cmd'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        self.vs = rospy.Subscriber(cmd_vel_topic, Twist, self.Callback)
            
        #frontbc_topic = '/frontbc'
        #self.bottom_pub = rospy.Publisher(frontbc_topic, Image, queue_size=10)

        image_state_topic = '/carla/ego' + str(index) + '/camera/rgb/front/image_color'  # front
        self.image_state_sub = rospy.Subscriber(image_state_topic, Image, self.image_callback)
        
        #imagebc_state_topic = '/carla/ego' + str(index) + '/camera/rgb/frontbc/image_color'  # front_bottom
        #self.image_state_sub = rospy.Subscriber(image_state_topic, Image, self.image_callbackbc)

        image_state_topic2 = '/carla/ego' + str(index) + '/camera/rgb/left/image_color'  # left
        self.image_state_sub2 = rospy.Subscriber(image_state_topic2, Image, self.image_callback2)

        image_state_topic3 = '/carla/ego' + str(index) + '/camera/rgb/right/image_color'  # right
        self.image_state_sub3 = rospy.Subscriber(image_state_topic3, Image, self.image_callback3)

        odom_topic = '/carla/ego' + str(index) + '/odometry'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = '/carla/ego' + str(index) + '/collision'
        self.check_crash = rospy.Subscriber(crash_topic, CarlaCollisionEvent, self.crash_callback)

        lane_topic = '/carla/ego' + str(index) + '/lane_invasion'
        self.lane_crash = rospy.Subscriber(lane_topic, CarlaLaneInvasionEvent, self.lane_callback, queue_size=1)

        info_topic = '/carla/ego' + str(index) + '/vehicle_info'
        self.check_crash = rospy.Subscriber(info_topic, CarlaEgoVehicleInfo, self.vehicle_info_callback)

	flag_topic = '/carla/ego' + str(index)+'/vehicle_control_manual'
	self.Flag = rospy.Subscriber(flag_topic, Bool, self.flag_callback)
 
        laser_topic = '/carla/ego' + str(index) + '/lidar/lidar1/point_cloud'
        self.laser_sub = rospy.Subscriber(laser_topic, PointCloud2, self.laser_scan_callback)


        while self.car_id is None or self.speed is None or self.odom_location is None:
            pass

        rospy.sleep(1)

  
    def image_callback(self, image_data):

        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img = np.asarray(image_data, dtype=np.float32)
	#self.img3 = self.img3
	self.img11 = m.imresize(self.img[115:500,:],[80,200])
    def image_callback2(self, image_data):

        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img2 = np.asarray(image_data, dtype=np.float32)
	#self.img2 = self.img2
	self.img21 = m.imresize(self.img2[115:500,:],[80,200])

    def image_callback3(self, image_data):
        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img3 = np.asarray(image_data, dtype=np.float32)
	#self.img3 = self.img3
	self.img31 = m.imresize(self.img3[115:500,:],[80,200])

    def odometry_callback(self, odometry):
        Quaternious = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        lox = math.cos(-Euler[2])
        loy = math.sin(-Euler[2])
        self.odom_location = [odometry.pose.pose.position.x, -odometry.pose.pose.position.y, 0.22, lox, loy, 1e-6]
        self.odom_location1 = [odometry.pose.pose.position.x, -odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def flag_callback(self, f):
	self.Flag = f.data


    def bottom_observation(self):
        cimg = copy.deepcopy(self.img)
        cimg = cimg[:, :, 0:3]
        cimg = np.array(cimg, dtype=np.uint8)
        cimg = m.imresize(cimg, (128, 160))
        cimg = cimg.astype(np.float64)
        cimg = cimg.astype(float) / 255.0

#        img = copy.deepcopy(self.img)
        img = cimg[:, :, 0:3]
        #cv2.imshow('221111',img)
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, (128, 160 ))
       # img = img.astype(np.float64)
        #img = img.astype(float) / 255.0
        #img = np.array([img])
	#print(img.shape)
       # print(type(img))   
        outc = self.bottom_policy.predict([cimg[None]])
        bottom_c = np.argmax(outc[0][0], axis=1)
        bottom_c = bottom_c.tolist()

        img_bg = np.ones([128, 160, 3], np.uint8)
        img_bg = img_bg * 255       

        for i in range(len(bottom_c)):
            ptStart = (i, 240)
            ptEnd = (i, int(bottom_c[i]))
            green = (0, 255, 0)
            cv2.line(img_bg, ptStart, ptEnd, green, 1)

        img = cv2.addWeighted(img, 0.8, img_bg, 0.2, 0)
        print(type(img))
        try:
            img = bridge.cv2_to_imgmsg(img, 'bgr8')
        except CvBridgeError, e:
            print e
        
        print('success....')
        
        self.bottom_pub.publish(img)


    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.other_actor_id

    def lane_callback(self, data):
        self.crossed_lane_markings = data.crossed_lane_markings

    def vehicle_info_callback(self, info):
        self.car_id = info.id


    def get_bottom_pixel(self):  # seg = visual_policy(rgb_list)
        cimg = copy.deepcopy(self.img)
        limg = copy.deepcopy(self.img2)
        rimg = copy.deepcopy(self.img3)

        cimg = cimg[:, :, 0:3]
        cimg = np.array(cimg, dtype=np.uint8)
        cimg = m.imresize(cimg, (128, 160))
        cimg = cimg.astype(np.float64)
        cimg = cimg.astype(float) / 255.0

        limg = limg[:, :, 0:3]
        limg = np.array(limg, dtype=np.uint8)
        limg = m.imresize(limg, (128, 160))
        limg = limg.astype(np.float64)
        limg = limg.astype(float) / 255.0

        rimg = rimg[:, :, 0:3]
        rimg = np.array(rimg, dtype=np.uint8)
        rimg = m.imresize(rimg, (128, 160))
        rimg = rimg.astype(np.float64)
        rimg = rimg.astype(float) / 255.0

        outc = self.bottom_policy.predict([cimg[None]])
        outl = self.bottom_policy.predict([limg[None]])
        outr = self.bottom_policy.predict([rimg[None]])

        bottom_c = np.argmax(outc[0][0], axis=1).astype(float)
        bottom_l = np.argmax(outl[0][0], axis=1).astype(float)
        bottom_r = np.argmax(outr[0][0], axis=1).astype(float)


        # bottom = bottom_l + bottom_c + bottom_r
        
        # print(len(bottom))
        bottom_l = (bottom_l/127).reshape(1,-1)
        bottom_c = (bottom_c/127).reshape(1,-1)
        bottom_r = (bottom_r/127).reshape(1,-1)

        return bottom_l, bottom_c, bottom_r

    def get_self_speed(self):
        return self.speed

    def get_self_flag(self):
        return self.Flag

    def get_self_odom_location(self):
        return self.odom_location

    def get_self_odom_location1(self):
        return self.odom_location1

    def get_crash_state(self):
        if self.is_crashed != None:
            print('self.is_crashed:', self.is_crashed)
            self.is_crashed = None
            return True
        else:
            return False

    def get_lane_state(self):

        # print(self.crossed_lane_markings) #(1,1,10,1)

        if self.crossed_lane_markings != None:
            if self.crossed_lane_markings[0] is CarlaLaneInvasionEvent.LANE_MARKING_OTHER or self.crossed_lane_markings[
                0] is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                self.crossed_lane_markings = None

                return 1
            elif self.crossed_lane_markings[0] is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                self.crossed_lane_markings = None

                return 2

            else:
                self.crossed_lane_markings = None
                return 3

        else:
            return 1


    def control_vel(self, action):
        self.move_cmd = Twist()
        self.move_cmd.linear.x = action[0] * 2  # 20/3.6   # action[0] * 2
        self.move_cmd.linear.y = 0.
        self.move_cmd.linear.z = 0.
        self.move_cmd.angular.x = 0.
        self.move_cmd.angular.y = 0.
        self.move_cmd.angular.z = -action[1]
        self.cmd_vel.publish(self.move_cmd)

    def get_reward_and_terminate(self, t):

        done = False
        done1 = False
        collsion = False
        collsion = self.get_crash_state()
        lane = 1
        lane = self.get_lane_state()
        # print('collision:',collsion)
        x, y, theta = self.get_self_odom_location1()

        [v, w] = self.get_self_speed()
        self.distant = self.planner.distance(self.vehicle.get_transform().location,self.spawn_points[self.goal_point].location)
        self.distant1 = np.sqrt((self.vehicle.get_transform().location.x- self.target_transform[0] )** 2 + (self.vehicle.get_transform().location.y- self.target_transform[1] ) ** 2)   
        if (self.distant1 - self.last_distant1) >0:
            self.count+=1
        else : 
            self.count =0
        reward_dis = self.last_distant - self.distant
        if reward_dis<-20:
            reward_dis = -1
            self.replanning =True
        if reward_dis>20:
            reward_dis = 1
            self.replanning =True
        if reward_dis == 0:
            reward_dis = self.last_dis
        self.last_distant1 = self.distant1
        self.last_dis = reward_dis

        self.last_distant = self.distant
        reward_time = 0#-0.1  # t * 0.001
        reward_c = 0
        reward_c1 = 0
        reward_c2 = 0
        reward_v = 0
        reward_w = 0
        reward_direction = 0
        result = 0
        right_flag = 0
        reward_terminal = 0

        if lane == 3 and t < 8: print('reset is on Sidewalk line:', lane)
        if lane == 2:
            reward_c1 = 0
            result = 'Cross center line'
            # print(result)
        elif lane == 3:
            done = True
            reward_c1 = -200
            result = 'Cross Sidewalk line'

        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())

        road_id = waypoint.road_id
        if road_id in road[0]:
            if theta > -np.pi and theta < 0:
                right_flag = 1
            else:
                right_flag = -1
        if road_id in road[1]:
            if theta > -0.5 * np.pi and theta < 0.5 * np.pi:
                right_flag = 1
            else:
                right_flag = -1
        if road_id in road[2]:
            if theta > -0.5 * np.pi and theta < 0.5 * np.pi:
                right_flag = -1
            else:
                right_flag = 1

        if waypoint.lane_id == right_flag or right_flag == 0:
            # print ("right")
            pass
        else:
            reward_c2 = -0.5   ##
            # print ("left")

        if collsion:
            done = True
            reward_c = -200
            result = 'Crashed'

        reward_v = v
        reward_w = -0.5 * w ** 2

        # print(w)
        # reward for steer
        if self.last_direction == 2:  # go  straight
            if abs(w) > 0.2: reward_direction = -0.5

        elif self.last_direction == 3:  # turn  left ,steer should be positive
            if w < -0.05: reward_direction = -1.5

        elif self.last_direction == 4:  # turn  right
            if w > 0.05: reward_direction = -1.5

        if self.distant<3:
            done1 = True
            reward_terminal = 400
            result = 'Reach goal'
            print(" ")
            print(11111111111111111111111111111111)
            print(" ")
        if self.count>6 and self.count<12:
            
            reward_terminal = -2
            result = 'fail1'
	if self.count>112:	
            done = True
            reward_terminal = -200
            result = 'fail'

        if t > 2000:
            done = True
            result = 'Time out'

        if self.last_road_id == road_id and self.last_right_flag != right_flag and self.last_right_flag != 0 and t > 3:
            done = True
            reward_terminal = -200
            result = 'reverse'
        self.last_right_flag = right_flag
        self.last_road_id = road_id
        
        reward = reward_c + reward_c1 + reward_c2 + reward_v*0.1 + reward_time + reward_direction + reward_terminal + reward_dis*0.3
        reward = reward/220
        #print(reward_direction)
       # print(reward_c , reward_c1 , reward_c2 , reward_v , reward_time , reward_direction ,reward_terminal , reward_dis)
        return reward, done1, result

    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)

    def get_directions(self):
        if self.planner.flag:
            self.recount+=1
            #sself.planner.set_route(self.vehicle.get_transform().location,self.spawn_points[self.goal_point].location)
            #print('Replanning')
        else:
            self.recount = 0
        direction = self.planner.get_direction()
        if direction != self.last_direction:
            if direction == 1:print("follow  lane")
            elif direction == 3:print("turn  left ")
            elif direction == 4:print("turn  right ")
            elif direction == 2:print("go  straight ")
            else :print("achieve")
        self.last_direction = direction

        if direction == 1:
            return 0
        elif direction == 2:
            return 3
        elif direction == 3:
            return 1
        elif direction == 4:
            return 2
        else:
            return 0


    def reset_pose(self, start_point, goal_point ,i):
        path = '/home/hxq/MVE2E/town1/'+str(i)+'.png'
        print(path)
        self.routed_map = cv2.imread(path)
        self.goal_point = goal_point
        self.replanning = False
        self.count = 0
        self.last_direction = 0
        self.control_vel([0, 0])
        rospy.sleep(0.15)
        self.vehicle = self.world.get_actor(self.car_id)

        self.planner= Planner(self.vehicle)
        self.planner.set_route(self.spawn_points[start_point].location,self.spawn_points[goal_point].location)
	#print(self.spawn_points[start_point].location.x, self.spawn_points[start_point].location.y)
        spawn_point = self.spawn_points[start_point]  
        spawn_point.location.z = 0.5
        self.vehicle.set_transform(spawn_point)
        rospy.sleep(0.05)
        b = self.vehicle.get_transform()
        # print('---start reset---')
        start = time.time()
        while np.abs(b.location.x - spawn_point.location.x) > 0.2 or np.abs(
                b.location.y - spawn_point.location.y) > 0.2 or b.location.z > 0.3:
            stop = time.time()
            if stop - start > 3:
                self.reset_pose(start_point, goal_point,i)
                print('break while')
                break
            b = self.vehicle.get_transform()
        rospy.sleep(0.5)
      
        for i in range(10): collsion = self.get_crash_state()
        for i in range(10): lane = self.get_lane_state()

        print('---achieve reset start: ', start_point)
        print('---achieve reset goal: ', goal_point)
        rospy.sleep(0.5)
        self.target_transform = [self.spawn_points[goal_point].location.x, self.spawn_points[goal_point].location.y]
        self.last_distant = self.planner.distance(self.vehicle.get_transform().location,self.spawn_points[goal_point].location)
        self.last_distant1 = np.sqrt((self.vehicle.get_transform().location.x- self.target_transform[0] )** 2 + (self.vehicle.get_transform().location.y- self.target_transform[1] ) ** 2)
        # print(self.get_self_odom_location())


    def laser_scan_callback(self, scan):
        # print(scan.width,scan.height)
        
        lidar = pc2.read_points(scan, field_names=("x", "y", "z"))  # ,field_names=("x", "y", "z"),skip_nans=False
        self.lidar_points = np.array(list(lidar))

    def get_map(self):
        loc = self.vehicle.get_transform()
        
        x, y, theta = self.get_self_odom_location1()
        pixel = self.carla_map.convert_to_pixel([loc.location.x,loc.location.y,loc.location.z])
        #print(pixel)
        M= cv2.getRotationMatrix2D(tuple(pixel),-theta*180/np.pi +90,1.0)
        roted = cv2.warpAffine(self.map,M,(2600,2200)) 
        a = cv2.resize(roted[int(pixel[1]-80):
,int(pixel[0]-80):int(pixel[0]+80)],(50,50))
        roted1 = cv2.warpAffine(self.routed_map,M,(2600,2200))
        b = cv2.resize(roted1[int(pixel[1]-80):int(pixel[1]+80),int(pixel[0]-80):int(pixel[0]+80)],(50,50))
        return a[:,:,0] ,b 

    def get_laser_observation(self):
        scan = copy.deepcopy(self.lidar_points)
        scan[np.isnan(scan)] = 20.0
        scan[np.isinf(scan)] = 20.0
        scan1 = 20 * np.ones((16000), dtype=np.float32)
        scan2 = np.zeros((8001), dtype=np.float32)
        theta = np.zeros((512), dtype=np.float32)
        for i in range(scan.shape[0]):
            m = math.sqrt(math.pow(scan[i][0], 2) + math.pow(scan[i][1], 2))
            n = math.atan(scan[i][1] / (scan[i][0] + 1e-10))
            n = n * 180 / math.pi
            if (scan[i][0] < 0) and (scan[i][1] > 0):
                n = n + 180
            if (scan[i][0] < 0) and (scan[i][1] < 0):
                n = n + 180
            if (scan[i][0] > 0) and (scan[i][1] < 0):
                n = n + 360
            p = int(round(n / 0.0225))
            if p == 16000:
                p = 0
            scan1[p] = m
        for i in range(8001):
            scan2[i] = scan1[i - 4000]

        for j in range(512):
            theta[j] = np.pi - np.pi / 512 * j
        r = scan2

        scan2[np.isnan(scan2)] = 6.0
        scan2[np.isinf(scan2)] = 6.0
        scan2 = np.array(scan2)
        scan2[scan2 > 6] = 6.0

        raw_beam_num = len(scan2)  # 8001
        sparse_beam_num = self.beam_mum  # 512
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan2[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan2[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        '''
        ax = plt.subplot(111, projection='polar')
        c = ax.scatter(theta, scan_sparse)
        plt.show()'''

        return scan_sparse /6.0 -0.5


    def comCallback(self, data):
        self.com = data.data

    def Callback(self, data):
        self.steering = data.angular.z
	self.velocity = data.linear.x





if __name__ == '__main__':
    def relu6(x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def jsonToModel(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            # print(loaded_model_json)
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf2, 'relu6': relu6})
        return model


    model_pixel = '/home/hxq/cw/bottom_pixel/model/model_struct.json'
    pixel_weights = '/home/hxq/cw/bottom_pixel/model/weights_009.h5'
    bottom_policy = jsonToModel(model_pixel)
    bottom_policy.load_weights(pixel_weights)

    env = CarlaEnv1(0, bottom_policy)
    env.get_bottom_pixel()
    # env.reset_pose()
    # env.get_image_observation_rgb2seg()
    print("start")
#    env.reset_pose()
#    cur = env.get_directions()
#    print(cur)
   



