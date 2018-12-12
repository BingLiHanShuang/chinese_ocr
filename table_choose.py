#!/usr/bin/env python  
# encoding: utf-8  

import os 
import cv2 as cv
import numpy as np
import math

import num_out
import mnist_recognize
import chinese_out
import chinese_out_jiguan
import time_out

import csv

##### 初始化CSV文件 #####
csvfile = open("try1010.csv", 'w')
csvwrite = csv.writer(csvfile)
fileHeader = ["登记表编号","性别","民族","体重","血型","籍贯","高中学校名称","高中专业","高中学位","高中起止时间","高中是否毕业","大专学校名称","大专专业","大专学位","大专起止时间","大专是否毕业","本科学校名称","本科专业","本科学位","本科起止时间","本科是否毕业","研究生学校名称","研究生专业","研究生学位","研究生起止时间","研究生是否毕业","学校名称（其它）","专业（其它）","学位（其它）","起止时间（其它）","是否毕业（其它）"]
csvwrite.writerow(fileHeader)

path = 'test_data'
for (path,dirs,files) in os.walk(path):
	for filename in files:

		#docu_num = '20110158' #测试单张登记表

                ##### 按文件名读取登记表 #####
		(docu_num,extension) = os.path.splitext(filename)
		print "filename:", filename

		##### 根据文件名年份的不同区分不同类型的登记表 #####
		print docu_num[3]
		if docu_num[3] == '1':
			if len(docu_num) == 9 or (docu_num[5] == '3' and docu_num[6] == '1'):
				table_style = 2
			else:
				table_style = 1
		if docu_num[3] == '3':
			table_style = 3
		if docu_num[3] == '4' or docu_num[3] == '5':
			table_style = 4
		jiaozheng = 0
		image = cv.imread('test_data/' + docu_num + '.jpg')
		rows, cols, channels = image.shape
		print rows, cols
		image_copy = image.copy()
		
		##### 旋转校正Rotation #####
		#统计图中长横线的斜率来判断整体需要旋转矫正的角度
		gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
		if table_style == 1 or table_style == 3 or table_style == 2:
			edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
			cv.imwrite('edges_whole.jpg', edges)
			lines = cv.HoughLinesP(edges, 1, np.pi / 180, 500, 0, minLineLength=50, maxLineGap=50)#650,50,20
		if table_style == 4:
			edges_gray = cv.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
			edges = edges_gray[400:1000, 0:1000]
			cv.imwrite('edges_whole.jpg', edges)
			lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, 0, minLineLength=50, maxLineGap=35)#650,50,20
		pi = 3.1415
		theta_total = 0
		theta_count = 0
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if table_style == 4:
				y1 = y1 + 400
				y2 = y2 + 400
			rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			theta = math.atan(float(y2 - y1)/float(x2 - x1 + 0.001))
			print(rho, theta, x1, y1, x2, y2)
			if theta < pi/4 and theta > -pi/4:
				theta_total = theta_total + theta
				theta_count+=1
				cv.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
				#cv.line(edges, (x1, y1), (x2, y2), (0, 0, 0), 2)
		theta_average = theta_total/theta_count
		print theta_average, theta_average*180/pi
		cv.imwrite('line_detect4rotation.jpg', image_copy)
		#cv.imwrite('line_detect4rotation.jpg', ~edges)
		#affineShrinkTranslationRotation = cv.getRotationMatrix2D((cols/2, rows/2), theta_average*180/pi, 1)
		affineShrinkTranslationRotation = cv.getRotationMatrix2D((0, rows), theta_average*180/pi, 1)
		ShrinkTranslationRotation = cv.warpAffine(image, affineShrinkTranslationRotation, (cols, rows))
		image_copy = cv.warpAffine(image_copy, affineShrinkTranslationRotation, (cols, rows))
		cv.imwrite('image_Rotation.jpg',ShrinkTranslationRotation)

		##### 平移校正Move #####
		#通过对表格左下角直角进行识别，将其顶点统一平移矫正至(78,1581)
		#print "rows: ",rows
		roi = image_copy[1450:rows, 0:150]#180
		gray = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
		edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
		roi_mean_set = cv.mean(~edges[0:int((rows-1450)/2), 85:150])#通过区域灰度值特征排除文字对直线识别的干扰
		roi_mean = roi_mean_set[0]
		#cv.imwrite('edges_sample.jpg', edges)
		cv.imwrite('edges_sample.jpg', ~edges[0:int((rows-1450)/2), 75:150])
		lines = cv.HoughLinesP(edges, 1.0, np.pi / 180, 35, 0, minLineLength=10,maxLineGap=20)#50,10,20
		lines_message_set = []
		for line in lines: 
			x1, y1, x2, y2 = line[0]
			y1 = y1 + 1450
			y2 = y2 + 1450
			rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			xielv = (y2 - y1)/(x2 - x1 + 0.001)
			theta = math.atan(float(y2 - y1)/float(x2 - x1 + 0.001))
			print(rho, theta, x1, y1, x2, y2, xielv)
			lines_message = (rho, theta, x1, y1, x2, y2, xielv)
			#cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
			lines_message_set.append(lines_message)
		#print len(lines_message_set)
		
                #求点到直线的距离
		def point2line_distance(x1,y1,x2,y2,pointPx,pointPy):
			A = y1 - y2
			B = x2 - x1
			C = x1*y2 - y1*x2
			distance = abs(A*pointPx + B*pointPy + C)/((A*A + B*B)**0.5)
			return distance

		lines_cluster_set = []
		repeat_num_set = []	
		for j in range(0,len(lines_message_set)):
			for i in range(0,len(lines_message_set)):
				if not(j in repeat_num_set):
						lines_cluster_set.append(lines_message_set[j])
						repeat_num_set.append(j)
				print point2line_distance(lines_message_set[i][2], lines_message_set[i][3], lines_message_set[i][4], lines_message_set[i][5], (lines_message_set[j][2]+lines_message_set[j][4])/2, (lines_message_set[j][3]+lines_message_set[j][5])/2)
				if i!=j and abs(lines_message_set[j][6] - lines_message_set[i][6]) < 0.1 and point2line_distance(lines_message_set[i][2], lines_message_set[i][3], lines_message_set[i][4], lines_message_set[i][5], (lines_message_set[j][2]+lines_message_set[j][4])/2, (lines_message_set[j][3]+lines_message_set[j][5])/2) <= 10:
					repeat_num_set.append(i)
		print lines_cluster_set
                #对直角的横线、竖线进行分析，缺省时根据表格类型进行矫正
		Point_heng = []
		Point_shu = []
		MiddlePoint_heng = (112,1450+(rows-1450)/2)
		MiddlePoint_shu = (75,1450+(rows-1450)/4)
		distance2point = rows-1450
		distance2point2 = rows-1450
		for j in range(0,len(lines_cluster_set)):
			if abs(lines_cluster_set[j][6]) < 1:
				if ((MiddlePoint_heng[0]-(lines_cluster_set[j][2]+lines_cluster_set[j][4])/2)**2 + (MiddlePoint_heng[1]-(lines_cluster_set[j][3]+lines_cluster_set[j][5])/2)**2)**0.5 < distance2point:
					distance2point = ((MiddlePoint_heng[0]-(lines_cluster_set[j][2]+lines_cluster_set[j][4])/2)**2 + (MiddlePoint_heng[1]-(lines_cluster_set[j][3]+lines_cluster_set[j][5])/2)**2)**0.5
					Point_heng = lines_cluster_set[j]
			else:
				if ((MiddlePoint_shu[0]-(lines_cluster_set[j][2]+lines_cluster_set[j][4])/2)**2 + (MiddlePoint_shu[1]-(lines_cluster_set[j][3]+lines_cluster_set[j][5])/2)**2)**0.5 < distance2point2:
					distance2point2 = ((MiddlePoint_shu[0]-(lines_cluster_set[j][2]+lines_cluster_set[j][4])/2)**2 + (MiddlePoint_shu[1]-(lines_cluster_set[j][3]+lines_cluster_set[j][5])/2)**2)**0.5
					Point_shu = lines_cluster_set[j]
		need_stronger = 0
		something_missing = 0
		#缺省矫正
		if Point_shu != []:
			if Point_heng == []:
				cross_x = 78
				cross_y = 1616
				something_missing = 1
				if len(docu_num) == 9:
					something_missing = 0
					cross_x = 93
					cross_y = 1666
				if docu_num[3] == '4':
					something_missing = 0
					cross_x = 78
					cross_y = 1655
					if docu_num[3] == '4' and docu_num[5] == '6':
						cross_x = 78
						cross_y = 1665
			else:
				cross_x = (Point_shu[3]-Point_shu[6]*Point_shu[2]-Point_heng[3]+Point_heng[6]*Point_heng[2])/(Point_heng[6]-Point_shu[6])
				cross_y = Point_heng[6]*cross_x + Point_heng[3] - Point_heng[6]*Point_heng[2]
					
				cv.line(image_copy, (Point_heng[2], Point_heng[3]), (Point_heng[4], Point_heng[5]), (0, 0, 255), 2)
				cv.line(image_copy, (Point_shu[2], Point_shu[3]), (Point_shu[4], Point_shu[5]), (0, 0, 255), 2)
		else:
			cross_x = Point_heng[2]
			cross_y = Point_heng[3]
			need_stronger = 1
		if Point_heng != [] and cross_x > Point_heng[2] and len(docu_num) == 9 and docu_num[6] == '0' and docu_num[7] == '9':
			cross_x = 50
			cross_y = 1586
		if Point_heng != [] and cross_x > Point_heng[2] and len(docu_num) == 9 and docu_num[6] == '1' and docu_num[7] == '8':
			cross_x = 50
			cross_y = 1616
		print 'roi_mean:',roi_mean
		if len(docu_num) == 9 and docu_num[7] == '8' and roi_mean < 230:
			cross_x = 78
			cross_y = 1631
		if cross_y - 1648 < 3:
			if len(docu_num) == 8 and docu_num[5] == '2' and docu_num[6] == '4':
				cross_x = 78
				cross_y = 1601
			if len(docu_num) == 9 and docu_num[6] == '1' and docu_num[7] == '4':
				cross_x = 78
				cross_y = 1621
		if table_style == 3 and cross_y < 1485:
			cross_x = 78
			cross_y = 1641
			
		print cross_x,cross_y#当下直角顶点位置，标准位置为78,1581

		cv.circle(image_copy, (int(cross_x), int(cross_y)), 3,(255,0,0),3)
		cv.rectangle(image_copy,(0,1450),(180,rows),(255,0,0),3)
		cv.imwrite('line_detect_possible_demo.jpg', image_copy)

		rows, cols, channels = ShrinkTranslationRotation.shape
		print rows, cols
		affineShrinkTranslation = np.array([[1, 0, int(78 - cross_x)], [0, 1, int(1581 - cross_y)]], np.float32)
		#affineShrinkTranslation = np.array([[1, 0, int(78 - 78)], [0, 1, int(1581 - 1581)]], np.float32)
		shrinkTwoTimesTranslation = cv.warpAffine(ShrinkTranslationRotation, affineShrinkTranslation, (cols, rows))
		image_copy = cv.warpAffine(image_copy, affineShrinkTranslation, (cols, rows))

		##### 对201100XXX中左下角表格内有额外竖线的表格进行检测Detect_not_shu_line_in_201100XXX #####
		if table_style == 2:
			shrinkTwoTimesTranslation_copy_copy = shrinkTwoTimesTranslation.copy()
			roi = shrinkTwoTimesTranslation[1350:1548, 125:300]#180
			cv.rectangle(image_copy,(125,1350),(300,1548),(255,0,0),3)
			gray = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
			edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
			lines = []
			lines = cv.HoughLinesP(edges, 1.0, np.pi / 180, 100, 0, minLineLength=60,maxLineGap=20)#50,10,20
			print lines
			go_through = 0
			try:
				if lines == None:
					go_through = 0
			except:
				go_through = 1
			
			if go_through == 1:
				pi = 3.1415
				theta_total = 0
				theta_count = 0
				for line in lines:
					x1, y1, x2, y2 = line[0]
					x1 = x1 + 125
					x2 = x2 + 125
					y1 = y1 + 1350
					y2 = y2 + 1350
					rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
					theta = math.atan(float(y2 - y1)/float(x2 - x1 + 0.001))
					print(rho, theta, x1, y1, x2, y2)
					if theta > pi/3 or theta < -pi/3:
						table_style = 1
						if len(docu_num) == 9 and docu_num[6] == '1' and docu_num[7] == '1':
							jiaozheng = 1
						cv.line(shrinkTwoTimesTranslation_copy_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
				cv.imwrite('shrinkTwoTimesTranslation_copy_copy.jpg', shrinkTwoTimesTranslation_copy_copy)

		##### 提取表格Table_Out #####
		#分别通过对二值化后的表格用长横条、长竖条内核进行开操作，将表格分别化为全横线与全竖线，叠加后提取交点，即可得到表格中每个矩形的四个顶点
		shrinkTwoTimesTranslation_gray = cv.cvtColor(shrinkTwoTimesTranslation, cv.COLOR_RGB2GRAY)
		th2 = cv.adaptiveThreshold(~shrinkTwoTimesTranslation_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,-2)
		cv.imwrite('th2.jpg', th2)
                #长横条内核处理
		shrinkTwoTimesTranslation_copy = shrinkTwoTimesTranslation.copy()
		th2_copy = th2.copy()
		scale = 44;#20,50,45,40,44
		rows,cols = shrinkTwoTimesTranslation_gray.shape
		horizontalsize = cols / scale
		horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalsize, 1))
		erosion = cv.erode(th2,horizontalStructure,iterations = 1)
		dilation = cv.dilate(erosion,horizontalStructure,iterations = 1)
		#长竖条内核处理
		scale = 39;#20,50,45,40,39
		horizontalsize2 = rows / scale
		horizontalStructure2 = cv.getStructuringElement(cv.MORPH_RECT, (1,horizontalsize2))
		erosion2 = cv.erode(th2_copy,horizontalStructure2,iterations = 1)
		dilation2 = cv.dilate(erosion2,horizontalStructure2,iterations = 1)
                #全横线图与全竖线图叠加，并提取交点
		mask = dilation + dilation2
		cv.imwrite('mask.jpg', mask)
		joints = cv.bitwise_and(dilation, dilation2)
		cv.imwrite('joints.jpg', joints)
                #根据矩形大小筛选矩形框，并画在矫正后的表格上
		#cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE
		mask, contours, hierarchy = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
		length = len(contours)
		print length
		small_rects = []
		big_rects = []
		for i in range(length):
			cnt = contours[i]
			area = cv.contourArea(cnt)
			if area < 10:
				continue
			approx = cv.approxPolyDP(cnt, 3, True)#3
			x, y, w, h = cv.boundingRect(approx)
			rect = (x, y, w, h)
			#cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
			roi = joints[y:y+h, x:x+w]
			roi, joints_contours, joints_hierarchy = cv.findContours(roi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
			#print len(joints_contours)
			#if h < 80 and h > 20 and w > 10 and len(joints_contours)<=4:
			if h < 80 and h > 20 and w > 10 and len(joints_contours)<=6:#important
				cv.rectangle(image_copy, (x, y), (x+w, y+h), (255-h*3, h*3, 0), 3)
				small_rects.append(rect)
		cv.imwrite('table_out.jpg', image_copy)

                ##### 在不同类型的表格中根据信息所在的大致位置获取相应矩形框的坐标 #####
		#矫正后的表格中信息的大致位置各在一定范围内，根据大致位置的坐标点筛选出该表中该信息对应的矩形框具体坐标
		request_info_set = []#存储筛选出的所需矩形框坐标
		#pure_for_message = shrinkTwoTimesTranslation.copy()
		print "table_style:", table_style

		if table_style == 1:
			request_info = (770, 150, 1150-770, 300-150)
			request_info_set.append(request_info)
			cv.rectangle(shrinkTwoTimesTranslation,(770,150),(1150,300),(255,0,0),1)#登记表编号0

			for j in range(0,len(small_rects)):
				(x, y, w, h) = small_rects[j]
				if x < 1060 and y < 1130 and x+w > 1060 and y+h > 1130 and something_missing == 1:#特殊情况下表格矩形的提取
					x_rem = x
					y_rem = y
					w_rem = w
					h_rem = h

			for j in range(0,len(small_rects)):
				(x, y, w, h) = small_rects[j]
				if x < 700 and y < 370 and x+w > 700 and y+h > 370 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#性别1
				if x < 880 and y < 370 and x+w > 880 and y+h > 370 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#民族2
				if x < 685 and y < 412 and x+w > 685 and y+h > 412 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#体重3
				if x < 880 and y < 412 and x+w > 880 and y+h > 412 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(0,0,255),1)#血型4
				if x < 890 and y < 452 and x+w > 890 and y+h > 452 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#籍贯5

				if x < 486 and y < 1090 and x+w > 486 and y+h > 1090 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中学校名称6
				if x < 696 and y < 1090 and x+w > 696 and y+h > 1090 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中专业7
				if x < 808 and y < 1090 and x+w > 808 and y+h > 1090 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中学位8
				if x < 931 and y < 1090 and x+w > 931 and y+h > 1090 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中起止时间9
				if x < 1060 and y < 1087 and x+w > 1060 and y+h > 1087 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中是否毕业10
					#提取是、否选项旁的正方形框
					if jiaozheng == 1 or (len(docu_num) == 9 and (docu_num[6] == '2' or docu_num[6] == '3')) or (len(docu_num) == 8 and ((docu_num[5] == '3' and (docu_num[6] == '3' or (docu_num[6] == '3' and (docu_num[7] == '5' or docu_num[7] == '6' or docu_num[7] == '7' or docu_num[7] == '8' or docu_num[7] == '9')) or docu_num[6] == '4' or docu_num[6] == '5' or docu_num[6] == '7' or docu_num[6] == '8' or docu_num[6] == '9')) or docu_num[5] == '4')):
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.18*w+x):int(0.28*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.18*w+x),int(y+h/3.5)),(int(0.28*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				if x < 486 and y < 1130 and x+w > 486 and y+h > 1130 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专学校名称11
				if x < 696 and y < 1130 and x+w > 696 and y+h > 1130 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专专业12
				if x < 808 and y < 1130 and x+w > 808 and y+h > 1130 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专学位13
				if x < 931 and y < 1130 and x+w > 931 and y+h > 1130 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专起止时间14
				if x < 1060 and y < 1127 and x+w > 1060 and y+h > 1127 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专是否毕业15
					#提取是、否选项旁的正方形框
					if jiaozheng == 1 or (len(docu_num) == 9 and (docu_num[6] == '2' or docu_num[6] == '3')) or (len(docu_num) == 8 and ((docu_num[5] == '3' and (docu_num[6] == '3' or (docu_num[6] == '3' and (docu_num[7] == '5' or docu_num[7] == '6' or docu_num[7] == '7' or docu_num[7] == '8' or docu_num[7] == '9')) or docu_num[6] == '4' or docu_num[6] == '5' or docu_num[6] == '7' or docu_num[6] == '8' or docu_num[6] == '9')) or docu_num[5] == '4')):
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.18*w+x):int(0.28*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.18*w+x),int(y+h/3.5)),(int(0.28*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
						
				if x < 486 and y < 1170 and x+w > 486 and y+h > 1170 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科学校名称16
				if x < 696 and y < 1170 and x+w > 696 and y+h > 1170 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科专业17
				if x < 808 and y < 1170 and x+w > 808 and y+h > 1170 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科学位18
				if x < 931 and y < 1170 and x+w > 931 and y+h > 1170 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科起止时间19
				if x < 1060 and y < 1167 and x+w > 1060 and y+h > 1167 or something_missing == 1:
					if something_missing == 1:#特殊情况下表格矩形的提取
						x = x_rem
						y = y_rem + 40
						w = w_rem
						h = h_rem
					else:
						request_info = (x, y, w, h)
						request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科是否毕业20
					#提取是、否选项旁的正方形框
					if jiaozheng == 1 or (len(docu_num) == 9 and (docu_num[6] == '2' or docu_num[6] == '3')) or (len(docu_num) == 8 and ((docu_num[5] == '3' and (docu_num[6] == '3' or (docu_num[6] == '3' and (docu_num[7] == '5' or docu_num[7] == '6' or docu_num[7] == '7' or docu_num[7] == '8' or docu_num[7] == '9')) or docu_num[6] == '4' or docu_num[6] == '5' or docu_num[6] == '7' or docu_num[6] == '8' or docu_num[6] == '9')) or docu_num[5] == '4')):
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.18*w+x):int(0.28*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.18*w+x),int(y+h/3.5)),(int(0.28*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				if x < 486 and y < 1210 and x+w > 486 and y+h > 1210 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生学校名称21
				if x < 696 and y < 1210 and x+w > 696 and y+h > 1210 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生专业22
				if x < 808 and y < 1210 and x+w > 808 and y+h > 1210 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生学位23
				if x < 931 and y < 1210 and x+w > 931 and y+h > 1210 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生起止时间24
				if x < 1060 and y < 1207 and x+w > 1060 and y+h > 1207 or something_missing == 1:
					if something_missing == 1:#特殊情况下表格矩形的提取
						x = x_rem
						y = y_rem + 80
						w = w_rem
						h = h_rem
					else:
						request_info = (x, y, w, h)
						request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生是否毕业25
					#提取是、否选项旁的正方形框
					if jiaozheng == 1 or (len(docu_num) == 9 and (docu_num[6] == '2' or docu_num[6] == '3')) or (len(docu_num) == 8 and ((docu_num[5] == '3' and (docu_num[6] == '3' or (docu_num[6] == '3' and (docu_num[7] == '5' or docu_num[7] == '6' or docu_num[7] == '7' or docu_num[7] == '8' or docu_num[7] == '9')) or docu_num[6] == '4' or docu_num[6] == '5' or docu_num[6] == '7' or docu_num[6] == '8' or docu_num[6] == '9')) or docu_num[5] == '4')):
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.18*w+x):int(0.28*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.18*w+x),int(y+h/3.5)),(int(0.28*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
			#特殊情况下表格矩形的提取
			if need_stronger == 1:
				cv.rectangle(shrinkTwoTimesTranslation,(673,385),(740,420),(0,0,255),1)#性别
				cv.rectangle(shrinkTwoTimesTranslation,(845,385),(938,420),(0,0,255),1)#民族
				cv.rectangle(shrinkTwoTimesTranslation,(673,425),(720,460),(0,0,255),1)#体重
				cv.rectangle(shrinkTwoTimesTranslation,(845,425),(938,465),(0,0,255),1)#血型
				cv.rectangle(shrinkTwoTimesTranslation,(845,470),(938,510),(0,0,255),1)#籍贯
				cv.rectangle(shrinkTwoTimesTranslation,(973,1095),(1153,1135),(0,0,255),1)#高中是否毕业
				x = 973
				y = 1095
				w = 1153 - 973
				h = 1135 - 1095
				gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
				gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
				cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
				cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
				
				cv.rectangle(shrinkTwoTimesTranslation,(973,1135),(1153,1175),(0,0,255),1)
				x = 973
				y = 1135
				w = 1153 - 973
				h = 1175 - 1135
				dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
				dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
				cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
				cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				cv.rectangle(shrinkTwoTimesTranslation,(973,1175),(1153,1215),(0,0,255),1)
				x = 973
				y = 1175
				w = 1153 - 973
				h = 1215 - 1175
				benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
				benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
				cv.imwrite('benke_roi_left.jpg', benke_roi_left)
				cv.imwrite('benke_roi_right.jpg', benke_roi_right)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				cv.rectangle(shrinkTwoTimesTranslation,(973,1220),(1153,1260),(0,0,255),1)
				x = 973
				y = 1220
				w = 1153 - 973
				h = 1260 - 1220
				yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
				yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
				cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
				cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
				cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
		#测试各信息的大致位置
		#cv.circle(shrinkTwoTimesTranslation, (700, 335), 3,(255,0,0),3)
		#cv.circle(shrinkTwoTimesTranslation, (75, 1581), 3,(255,0,0),3)
		#cv.rectangle(shrinkTwoTimesTranslation,(663,360),(740,400),(255,0,0),1)#性别
		#cv.rectangle(shrinkTwoTimesTranslation,(835,360),(928,400),(255,0,0),1)#民族
		#cv.rectangle(shrinkTwoTimesTranslation,(663,402),(710,442),(255,0,0),1)#体重
		#cv.rectangle(shrinkTwoTimesTranslation,(835,402),(928,442),(255,0,0),1)#血型
		#cv.rectangle(shrinkTwoTimesTranslation,(835,443),(948,482),(255,0,0),1)#籍贯

		#cv.rectangle(shrinkTwoTimesTranslation,(345,1075),(627,1115),(255,0,0),1)#高中学校名称
		#cv.rectangle(shrinkTwoTimesTranslation,(628,1075),(765,1115),(255,0,0),1)#高中专业
		#cv.rectangle(shrinkTwoTimesTranslation,(766,1075),(850,1115),(255,0,0),1)#高中学位
		#cv.rectangle(shrinkTwoTimesTranslation,(851,1075),(1011,1115),(255,0,0),1)#高中起止时间
		#cv.rectangle(shrinkTwoTimesTranslation,(971,1070),(1150,1110),(255,0,0),1)#高中是否毕业

		if table_style == 2:
			request_info = (770, 150, 1150-770, 300-150)
			request_info_set.append(request_info)
			cv.rectangle(shrinkTwoTimesTranslation,(770,150),(1150,300),(255,0,0),1)#登记表编号0

			for j in range(0,len(small_rects)):
				(x, y, w, h) = small_rects[j]
				if x < 700 and y < 335 and x+w > 700 and y+h > 335 :#350-20
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#性别1
				if x < 885 and y < 335 and x+w > 885 and y+h > 335 :#x+5
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#民族2
				if x < 685 and y < 377 and x+w > 685 and y+h > 377 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#体重3
				if x < 885 and y < 377 and x+w > 885 and y+h > 377 :#x+5
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(0,0,255),1)#血型4
				if x < 890 and y < 417 and x+w > 890 and y+h > 417 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#籍贯5

				if x < 486 and y < 1045 and x+w > 486 and y+h > 1045 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中学校名称6
				if x < 696 and y < 1045 and x+w > 696 and y+h > 1045 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中专业7
				if x < 808 and y < 1045 and x+w > 808 and y+h > 1045 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中学位8
				
				if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
					y_xiuzheng = 1090
				else:
					y_xiuzheng = 1055
				if x < 931 and y < y_xiuzheng and x+w > 931 and y+h > y_xiuzheng :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中起止时间9
				
				if x < 1060 and y < y_xiuzheng and x+w > 1060 and y+h > y_xiuzheng :#y+10
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中是否毕业10
					
					if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')) or (docu_num[6] == '3' and docu_num[8] == '9'):
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.22*w+x):int(0.32*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.55*w+x):int(0.65*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.22*w+x),int(y+h/3.5)),(int(0.32*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.55*w+x),int(y+h/3.5)),(int(0.65*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
				
				if x < 486 and y < 1085 and x+w > 486 and y+h > 1085 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专学校名称11
				if x < 696 and y < 1085 and x+w > 696 and y+h > 1085 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专专业12
				if x < 808 and y < 1085 and x+w > 808 and y+h > 1085 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专学位13
				
				if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
					y_xiuzheng = 1140
				else:
					y_xiuzheng = 1095
				if x < 931 and y < y_xiuzheng and x+w > 931 and y+h > y_xiuzheng :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专起止时间14
				
				if x < 1060 and y < y_xiuzheng and x+w > 1060 and y+h > y_xiuzheng :#y+10
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专是否毕业15
					
					if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')) or (docu_num[6] == '3' and docu_num[8] == '9'):
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.22*w+x):int(0.32*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.55*w+x):int(0.65*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.22*w+x),int(y+h/3.5)),(int(0.32*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.55*w+x),int(y+h/3.5)),(int(0.65*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				if x < 486 and y < 1125 and x+w > 486 and y+h > 1125 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科学校名称16
				if x < 696 and y < 1125 and x+w > 696 and y+h > 1125 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科专业17
				if x < 808 and y < 1125 and x+w > 808 and y+h > 1125 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科学位18
				
				if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
					y_xiuzheng = 1190
				else:
					y_xiuzheng = 1135
				if x < 931 and y < y_xiuzheng and x+w > 931 and y+h > y_xiuzheng :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科起止时间19
				
				if x < 1060 and y < y_xiuzheng and x+w > 1060 and y+h > y_xiuzheng :#y+10
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科是否毕业20
					
					if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')) or (docu_num[6] == '3' and docu_num[8] == '9'):
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.22*w+x):int(0.32*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.55*w+x):int(0.65*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.22*w+x),int(y+h/3.5)),(int(0.32*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.55*w+x),int(y+h/3.5)),(int(0.65*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)
					
				if x < 486 and y < 1165 and x+w > 486 and y+h > 1165 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生学校名称21
				if x < 696 and y < 1165 and x+w > 696 and y+h > 1165 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生专业22
				if x < 808 and y < 1165 and x+w > 808 and y+h > 1165 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生学位23
				
				if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
					y_xiuzheng = 1230
				else:
					y_xiuzheng = 1175
				if x < 931 and y < y_xiuzheng and x+w > 931 and y+h > y_xiuzheng :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生起止时间24
				
				if x < 1060 and y < y_xiuzheng and x+w > 1060 and y+h > y_xiuzheng :#y+10
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生是否毕业25
					
					if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')) or (docu_num[6] == '3' and docu_num[8] == '9'):
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.22*w+x):int(0.32*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.55*w+x):int(0.65*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.22*w+x),int(y+h/3.5)),(int(0.32*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.55*w+x),int(y+h/3.5)),(int(0.65*w+x),int(y+2.2*h/3)),(255,0,0),1)
					else:
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.28*w+x):int(0.38*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.2*h/3), int(0.51*w+x):int(0.61*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.28*w+x),int(y+h/3.5)),(int(0.38*w+x),int(y+2.2*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.51*w+x),int(y+h/3.5)),(int(0.61*w+x),int(y+2.2*h/3)),(255,0,0),1)

		if table_style == 3:
			request_info = (970, 50, 1200-970, 200-50)
			request_info_set.append(request_info)
			cv.rectangle(shrinkTwoTimesTranslation,(970,50),(1200,200),(255,0,0),1)#登记表编号0

			for j in range(0,len(small_rects)):
				(x, y, w, h) = small_rects[j]
				if x < 724 and y < 316 and x+w > 724 and y+h > 316 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#性别1
				if x < 960 and y < 316 and x+w > 960 and y+h > 316 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#民族2
				if x < 724 and y < 363 and x+w > 724 and y+h > 363 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#体重3
				if x < 960 and y < 363 and x+w > 960 and y+h > 363 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#血型4
				if x < 960 and y < 410 and x+w > 960 and y+h > 410 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#籍贯5
					
				if x < 981 and y < 1089 and x+w > 981 and y+h > 1089 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中起止时间9
					
				if x < 1105 and y < 1089 and x+w > 1105 and y+h > 1089 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中是否毕业10
					
					if docu_num[5] == '3' and docu_num[6] == '8':
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.19*w+x):int(0.29*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.19*w+x),int(y+h/3.5)),(int(0.29*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.1*h/3)),(255,0,0),1)
					else:
						gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
						gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
						cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
						cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1137 and x+w > 981 and y+h > 1137 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专起止时间14
				
				if x < 1105 and y < 1137 and x+w > 1105 and y+h > 1137 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专是否毕业15
					
					if docu_num[5] == '3' and docu_num[6] == '8':
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.19*w+x):int(0.29*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.19*w+x),int(y+h/3.5)),(int(0.29*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.1*h/3)),(255,0,0),1)
					else:
						dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
						dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
						cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
						cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1185 and x+w > 981 and y+h > 1185 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科起止时间19
				
				if x < 1105 and y < 1185 and x+w > 1105 and y+h > 1185 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科是否毕业20
					
					if docu_num[5] == '3' and docu_num[6] == '8':
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.19*w+x):int(0.29*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.19*w+x),int(y+h/3.5)),(int(0.29*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.1*h/3)),(255,0,0),1)
					else:
						benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
						benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
						cv.imwrite('benke_roi_left.jpg', benke_roi_left)
						cv.imwrite('benke_roi_right.jpg', benke_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1233 and x+w > 981 and y+h > 1233 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生起止时间24
				
				if x < 1105 and y < 1233 and x+w > 1105 and y+h > 1233 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生是否毕业25
					
					if docu_num[5] == '3' and docu_num[6] == '8':
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.19*w+x):int(0.29*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.59*w+x):int(0.69*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.19*w+x),int(y+h/3.5)),(int(0.29*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.59*w+x),int(y+h/3.5)),(int(0.69*w+x),int(y+2.1*h/3)),(255,0,0),1)
					else:
						yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
						yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
						cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
						cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
						cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
			
		if table_style == 4:
			request_info = (970, 50, 1200-970, 200-50)
			request_info_set.append(request_info)
			cv.rectangle(shrinkTwoTimesTranslation,(970,50),(1200,200),(255,0,0),1)#登记表编号0

			for j in range(0,len(small_rects)):
				(x, y, w, h) = small_rects[j]
				if x < 724 and y < 292 and x+w > 724 and y+h > 292 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#性别1
				if x < 960 and y < 292 and x+w > 960 and y+h > 292 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#民族2
				if x < 724 and y < 339 and x+w > 724 and y+h > 339 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#体重3
				if x < 960 and y < 339 and x+w > 960 and y+h > 339 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#血型4
				if x < 960 and y < 386 and x+w > 960 and y+h > 386 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#籍贯5
				
				if x < 981 and y < 1094 and x+w > 981 and y+h > 1094 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中起止时间9
					
				if x < 1105 and y < 1094 and x+w > 1105 and y+h > 1094 :#y-15
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#高中是否毕业10
					
					gaozhong_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
					gaozhong_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
					cv.imwrite('gaozhong_roi_left.jpg', gaozhong_roi_left)
					cv.imwrite('gaozhong_roi_right.jpg', gaozhong_roi_right)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1142 and x+w > 981 and y+h > 1142 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专起止时间14
				
				if x < 1105 and y < 1142 and x+w > 1105 and y+h > 1142 :#y-15
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#大专是否毕业15
					
					dazhuan_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
					dazhuan_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
					cv.imwrite('dazhuan_roi_left.jpg', dazhuan_roi_left)
					cv.imwrite('dazhuan_roi_right.jpg', dazhuan_roi_right)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1190 and x+w > 981 and y+h > 1190 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科起止时间19
				
				if x < 1105 and y < 1190 and x+w > 1105 and y+h > 1190 :#y-15
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#本科是否毕业20
					
					benke_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
					benke_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
					cv.imwrite('benke_roi_left.jpg', benke_roi_left)
					cv.imwrite('benke_roi_right.jpg', benke_roi_right)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
				if x < 981 and y < 1238 and x+w > 981 and y+h > 1238 :
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生起止时间24
				
				if x < 1105 and y < 1238 and x+w > 1105 and y+h > 1238 :#y-15
					request_info = (x, y, w, h)
					request_info_set.append(request_info)
					cv.rectangle(shrinkTwoTimesTranslation,(x,y),(x+w,y+h),(255,0,0),1)#研究生是否毕业25
					
					yanjiu_roi_left = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.23*w+x):int(0.33*w+x)]
					yanjiu_roi_right = shrinkTwoTimesTranslation[int(y+h/3.5):int(y+2.1*h/3), int(0.56*w+x):int(0.66*w+x)]
					cv.imwrite('yanjiu_roi_left.jpg', yanjiu_roi_left)
					cv.imwrite('yanjiu_roi_right.jpg', yanjiu_roi_right)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.23*w+x),int(y+h/3.5)),(int(0.33*w+x),int(y+2.1*h/3)),(255,0,0),1)
					cv.rectangle(shrinkTwoTimesTranslation,(int(0.56*w+x),int(y+h/3.5)),(int(0.66*w+x),int(y+2.1*h/3)),(255,0,0),1)
				
		#cv.rectangle(shrinkTwoTimesTranslation,(663,269),(786,315),(255,0,0),1)#性别
		#cv.rectangle(shrinkTwoTimesTranslation,(900,269),(1020,315),(255,0,0),1)#民族
		#cv.rectangle(shrinkTwoTimesTranslation,(663,316),(786,362),(255,0,0),1)#体重
		#cv.rectangle(shrinkTwoTimesTranslation,(900,316),(1020,362),(255,0,0),1)#血型
		#cv.rectangle(shrinkTwoTimesTranslation,(900,363),(1020,409),(255,0,0),1)#籍贯
		
		#cv.rectangle(shrinkTwoTimesTranslation,(1016,1065),(1195,1113),(255,0,0),1)#高中是否毕业

		#登记表编号,性别,民族,体重,血型,籍贯,高中学校名称,高中专业,高中学位,高中起止时间,高中是否毕业,大专学校名称,大专专业,大专学位,大专起止时间,大专是否毕业,本科学校名称,本科专业,本科学位,本科起止时间,本科是否毕业,研究生学校名称,研究生专业,研究生学位,研究生起止时间,研究生是否毕业,学校名称（其它）,专业（其它）,学位（其它）,起止时间（其它）,是否毕业（其它）
		cv.imwrite('image_correct.jpg', shrinkTwoTimesTranslation)#关键信息框重点标注的矫正图像

		########### 去除噪点Filter ###########
		th2 = cv.medianBlur(th2,5)#5
		horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))#2,2
		th2 = cv.erode(th2,horizontalStructure,iterations = 1)
		#horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))#2,2
		#th2 = cv.dilate(th2,horizontalStructure,iterations = 1)

		########### 登记表编号0 ###########
		#(x, y, w, h) = request_info_set[0]
		#num_num = 8
		#num_result = num_out.num_o(th2, x, y, w, h, num_num)
		#print "登记表编号: ",num_result

		########### 识别 性别1 ###########
		if table_style == 1:
			x_com = 700#信息所在的大致位置
			y_com = 370
		if table_style == 2:
			x_com = 700
			y_com = 335
		if table_style == 3:
			x_com = 724
			y_com = 316
		if table_style == 4:
			x_com = 724
			y_com = 292
		if need_stronger == 1:#特殊情况处理
			x = 673
			y = 385
			w = 740 - 673
			h = 420 - 385
			get_all = 1
			cha_num = 3
			is_ABO = 0
			xingbie_result = ''
			return_change, xingbie_result_set = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
			print xingbie_result_set
			if xingbie_result_set[0] == 2206 or xingbie_result_set[1] == 2206 or xingbie_result_set[2] == 2206:
				xingbie_result = '男'
			if xingbie_result_set[0] == 775 or xingbie_result_set[1] == 775 or xingbie_result_set[2] == 775:
				xingbie_result = '女'
			if xingbie_result == '':
				xingbie_result = '男'
			print "性别: ",xingbie_result
		else:
			for j in range(0,len(request_info_set)):
				(x, y, w, h) = request_info_set[j]
				if x < x_com and y < y_com and x+w > x_com and y+h > y_com :#提取信息所在的矩形框
					get_all = 1#对手写汉字模型的三个识别结果都进行分析
					cha_num = 3#字段预计最长长度
					is_ABO = 0#不是血型数据
					xingbie_result = ''#初始化
					#调用chinese_out.py中的chinese_o()函数进行手写汉字识别
					return_change, xingbie_result_set = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
					print xingbie_result_set
					#如果三个识别结果中有一个'男'，则确定为男性
					if xingbie_result_set[0] == 2206 or xingbie_result_set[1] == 2206 or xingbie_result_set[2] == 2206:
						xingbie_result = '男'
					#如果三个识别结果中有一个'女'，则确定为女性
					if xingbie_result_set[0] == 775 or xingbie_result_set[1] == 775 or xingbie_result_set[2] == 775:
						xingbie_result = '女'
					#特殊情况处理
					if xingbie_result == '':
						xingbie_result = '男'
					print "性别: ",xingbie_result
		
		########### 识别 民族2 ###########
		if table_style == 1:
			x_com = 880
			y_com = 370
		if table_style == 2:
			x_com = 885
			y_com = 335
		if table_style == 3:
			x_com = 960
			y_com = 316
		if table_style == 4:
			x_com = 960
			y_com = 292
		minzu_result = '汉'
		if need_stronger == 1:
			x = 845
			y = 385
			w = 938 - 845
			h = 420 - 385
			get_all = 2
			cha_num = 1
			is_ABO = 0
			return_change, minzu_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
			if return_change != 0:
				if minzu_result[0] == 1809 or minzu_result[1] == 1809 or minzu_result[2] == 1809:
					minzu_result = '汉'
				else:
					minzu_result = return_change
			print "民族: ",minzu_result
		else:
			for j in range(0,len(request_info_set)):
				(x, y, w, h) = request_info_set[j]
				if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
					get_all = 2#对手写汉字模型的三个识别结果都进行分析，且标志当前为民族数据提取
					cha_num = 1#字段预计最长长度
					is_ABO = 0#不是血型数据
					#调用chinese_out.py中的chinese_o()函数进行手写汉字识别
					return_change, minzu_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
					if return_change != 0:#特殊情况处理
                                                #如果三个识别结果中有一个'汉'，则确定为汉，否则输出单字
						if minzu_result[0] == 1809 or minzu_result[1] == 1809 or minzu_result[2] == 1809:
							minzu_result = '汉'
						else:
							minzu_result = return_change
					print "民族: ",minzu_result
		print "民族: ",minzu_result

		########### 识别 体重3 ###########
		if table_style == 1:
			x_com = 685
			y_com = 412
		if table_style == 2:
			x_com = 685
			y_com = 377
		if table_style == 3:
			x_com = 724
			y_com = 363
		if table_style == 4:
			x_com = 724
			y_com = 339
		if need_stronger == 1:
			x = 673
			y = 425
			w = 720 - 673
			h = 460 - 425
			num_num = 3
			weight_result = num_out.num_o(th2_copy, x, y, int(w*0.6), h, num_num)
			if weight_result == '':
				weight_result = '无'
			print "体重: ",weight_result
		else:
			for j in range(0,len(request_info_set)):
				(x, y, w, h) = request_info_set[j]
				if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
					num_num = 3#字段预计最长长度
					#调用num_out.py中的num_o()函数进行手写数字识别，体重框只取前60%部分（避免kg干扰）
					weight_result = num_out.num_o(th2_copy, x, y, int(w*0.6), h, num_num)#0.6
					#特殊情况处理
					if weight_result == '':
						weight_result = '无'
					if len(weight_result) > 3:
						weight_result = weight_result[0:2]
					#if weight_result[0] != '1':
						#weight_result = weight_result[0:1]
					print "体重: ",weight_result
		
		########### 识别 血型4 ###########
		if table_style == 1:
			x_com = 880
			y_com = 412
		if table_style == 2:
			x_com = 885
			y_com = 377
		if table_style == 3:
			x_com = 960
			y_com = 363
		if table_style == 4:
			x_com = 960
			y_com = 339
		xuexing_result = 'O'
		if need_stronger == 1:
			x = 845
			y = 425
			w = 938 - 845
			h = 465 - 425
			get_all = 0
			cha_num = 2
			is_ABO = 1
			return_change, xuexing_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
			print xuexing_result, len(xuexing_result)
			if xuexing_result == '':
				xuexing_result = '无'
			if len(xuexing_result) >= 3 and (xuexing_result[2] == 'A' or xuexing_result[2] == 'B' or xuexing_result[2] == 'O'):
				if return_change != 2:
					xuexing_result = xuexing_result[0]
			if len(xuexing_result) == 2 and return_change != 2:
				xuexing_result = 'AB'
			print "血型: ",xuexing_result
		else:
			for j in range(0,len(request_info_set)):
				(x, y, w, h) = request_info_set[j]
				if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
					get_all = 0#只分析模型的最佳识别结果
					cha_num = 2#字段预计最长长度
					is_ABO = 1#是血型数据
					#调用chinese_out.py中的chinese_o()函数进行手写血型识别
					return_change, xuexing_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
					print xuexing_result, len(xuexing_result)
					#特殊情况处理
					if xuexing_result == '':
						xuexing_result = '无'
					if len(xuexing_result) >= 3 and (xuexing_result[2] == 'A' or xuexing_result[2] == 'B' or xuexing_result[2] == 'O'):
						if return_change != 2:
							xuexing_result = xuexing_result[0]
					if len(xuexing_result) == 2 and return_change != 2:
						xuexing_result = 'AB'
					print "血型: ",xuexing_result
		print "血型: ",xuexing_result
		
		########### 识别 籍贯5 ###########
		jiguan_result = '无'
		
		if table_style == 1:
			x_com = 890
			y_com = 452
		if table_style == 2:
			x_com = 890
			y_com = 417
		if table_style == 3:
			x_com = 960
			y_com = 410
		if table_style == 4:
			x_com = 960
			y_com = 386
		jiguan_result = ''
		try:
			if need_stronger == 1:
				x = 845
				y = 465
				w = 938 - 845
				h = 465 - 425
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 3
					is_ABO = 0
					return_change, jiguan_result = chinese_out_jiguan.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					jiguan_result = '无'
				print "籍贯: ",jiguan_result
			else:
				for j in range(0,len(request_info_set)):
					(x, y, w, h) = request_info_set[j]
					if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
						pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
						print "pic_mean[0]:", pic_mean[0]
						if pic_mean[0] > 30:#排除空框
							get_all = 0
							cha_num = 6#字段预计最长长度
							is_ABO = 0
							#调用chinese_out_jiguan.py中的chinese_o()函数进行手写汉字识别
							return_change, jiguan_result = chinese_out_jiguan.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
						else:
							jiguan_result = '无'
						print "籍贯: ",jiguan_result
		except:
			jiguan_result = '无'
			print "籍贯: ",jiguan_result
		
		########### 识别 高中学校名称6 ###########
		gaozhong_name_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 486 and y < 1090 and x+w > 486 and y+h > 1090 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 12
					is_ABO = 0
					gaozhong_name_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					gaozhong_name_result = '无'
				print "高中学校名称: ",gaozhong_name_result
'''
		########### 识别 高中专业7 ###########
		gaozhong_master_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 696 and y < 1090 and x+w > 696 and y+h > 1090 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 6
					is_ABO = 0
					gaozhong_master_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					gaozhong_master_result = '无'
				print "高中专业: ",gaozhong_master_result
'''
		########### 识别 高中学位8 ###########
		gaozhong_degree_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 808 and y < 1090 and x+w > 808 and y+h > 1090 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 3
					is_ABO = 0
					gaozhong_degree_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					gaozhong_degree_result = '无'
				print "高中学位: ",gaozhong_degree_result
'''
		########### 识别 高中起止时间9 ###########
		gaozhong_time_result = '无'
		if table_style == 1:
			x_com = 931
			y_com = 1090
		if table_style == 2:#特殊情况处理
			if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
				y_com = 1090
			else:
				y_com = 1055
			x_com = 931
		if table_style == 3:
			x_com = 981
			y_com = 1089
		if table_style == 4:
			x_com = 981
			y_com = 1094
		try:
			if need_stronger == 1:
				gaozhong_time_result = '无'
			else:
				for j in range(0,len(request_info_set)):
					(x, y, w, h) = request_info_set[j]
					if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
						pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
						print "pic_mean[0]:", pic_mean[0]
						if pic_mean[0] > 30:#排除空框
							num_num = 10#字段预计最长长度
							#调用time_out.py中的num_o()函数进行手写数字识别
							gaozhong_time_result = time_out.num_o(th2_copy, x, y, w, h, num_num)
						else:
							gaozhong_time_result = '无'
						print "高中起止时间: ",gaozhong_time_result
		except:
			gaozhong_time_result = '无'
			print "高中起止时间: ",gaozhong_time_result
		
		########### 识别 高中是否毕业10 ###########
		left_mean = cv.mean(gaozhong_roi_left)#求提取出的正方形区域均值
		right_mean = cv.mean(gaozhong_roi_right)
		print left_mean[0],right_mean[0]
		mean_diff = 14#两者间的差值超过该值则判断为是、否，否则判断为两个空框
		if left_mean[0] < right_mean[0] - mean_diff:
			gaozhong_biye_result = '是'
		else:
			if right_mean[0] < left_mean[0] - mean_diff:
				gaozhong_biye_result = '否'
			else:
				gaozhong_biye_result = '无'
		print "高中是否毕业: ",gaozhong_biye_result

		########### 识别 大专学校名称11 ###########
		dazhuan_name_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 486 and y < 1130 and x+w > 486 and y+h > 1130 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 12
					is_ABO = 0
					dazhuan_name_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					dazhuan_name_result = '无'
				print "大专学校名称: ",dazhuan_name_result
'''
		########### 识别 大专专业12 ###########
		dazhuan_master_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 696 and y < 1130 and x+w > 696 and y+h > 1130 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 6
					is_ABO = 0
					dazhuan_master_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					dazhuan_master_result = '无'
				print "大专专业: ",dazhuan_master_result
'''
		########### 识别 大专学位13 ###########
		dazhuan_degree_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 808 and y < 1130 and x+w > 808 and y+h > 1130 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 3
					is_ABO = 0
					dazhuan_degree_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					dazhuan_degree_result = '无'
				print "大专学位: ",dazhuan_degree_result
'''
		########### 识别 大专起止时间14 ###########
		dazhuan_time_result = '无'
		if table_style == 1:
			x_com = 931
			y_com = 1130
		if table_style == 2:
			if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
				y_com = 1140
			else:
				y_com = 1095
			x_com = 931
		if table_style == 3:
			x_com = 981
			y_com = 1137
		if table_style == 4:
			x_com = 981
			y_com = 1142
		try:
			if need_stronger == 1:
				dazhuan_time_result = '无'
			else:
				for j in range(0,len(request_info_set)):
					(x, y, w, h) = request_info_set[j]
					if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
						pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
						print "pic_mean[0]:", pic_mean[0]
						if pic_mean[0] > 30:
							num_num = 10
							#调用time_out.py中的num_o()函数进行手写数字识别
							dazhuan_time_result = time_out.num_o(th2_copy, x, y, w, h, num_num)
						else:
							dazhuan_time_result = '无'
						print "大专起止时间: ",dazhuan_time_result
		except:
			dazhuan_time_result = '无'
			print "大专起止时间: ",dazhuan_time_result

		########### 识别 大专是否毕业15 ###########
		left_mean = cv.mean(dazhuan_roi_left)
		right_mean = cv.mean(dazhuan_roi_right)
		print left_mean[0],right_mean[0]
		mean_diff = 14#两者间的差值超过该值则判断为是、否，否则判断为两个空框
		if left_mean[0] < right_mean[0] - mean_diff:
			dazhuan_biye_result = '是'
		else:
			if right_mean[0] < left_mean[0] - mean_diff:
				dazhuan_biye_result = '否'
			else:
				dazhuan_biye_result = '无'
		print "大专是否毕业: ",dazhuan_biye_result

		########### 识别 本科学校名称16 ###########
		benke_name_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 486 and y < 1170 and x+w > 486 and y+h > 1170 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 12
					is_ABO = 0
					benke_name_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					benke_name_result = '无'
				print "本科学校名称: ",benke_name_result
'''
		########### 识别 本科专业17 ###########
		benke_master_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 696 and y < 1170 and x+w > 696 and y+h > 1170 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 6
					is_ABO = 0
					benke_master_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					benke_master_result = '无'
				print "本科专业: ",benke_master_result
'''
		########### 识别 本科学位18 ###########
		benke_degree_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 808 and y < 1170 and x+w > 808 and y+h > 1170 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 3
					is_ABO = 0
					benke_degree_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					benke_degree_result = '无'
				print "本科学位: ",benke_degree_result
'''
		########### 识别 本科起止时间19 ###########
		benke_time_result = '无'
		if table_style == 1:
			x_com = 931
			y_com = 1170
		if table_style == 2:
			if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
				y_com = 1190
			else:
				y_com = 1135
			x_com = 931
		if table_style == 3:
			x_com = 981
			y_com = 1185
		if table_style == 4:
			x_com = 981
			y_com = 1190
		try:
			if need_stronger == 1:
				benke_time_result = '无'
			else:
				for j in range(0,len(request_info_set)):
					(x, y, w, h) = request_info_set[j]
					if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
						pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
						print "pic_mean[0]:", pic_mean[0]
						if pic_mean[0] > 30:
							num_num = 10
							#调用time_out.py中的num_o()函数进行手写数字识别
							benke_time_result = time_out.num_o(th2_copy, x, y, w, h, num_num)
						else:
							benke_time_result = '无'
						print "本科起止时间: ",benke_time_result
		except:
			benke_time_result = '无'
			print "本科起止时间: ",benke_time_result
		
		########### 识别 本科是否毕业20 ###########
		left_mean = cv.mean(benke_roi_left)
		right_mean = cv.mean(benke_roi_right)
		print left_mean[0],right_mean[0]
		mean_diff = 14#两者间的差值超过该值则判断为是、否，否则判断为两个空框
		if left_mean[0] < right_mean[0] - mean_diff:
			benke_biye_result = '是'
		else:
			if right_mean[0] < left_mean[0] - mean_diff:
				benke_biye_result = '否'
			else:
				benke_biye_result = '无'
		print "本科是否毕业: ",benke_biye_result

		########### 识别 研究生学校名称21 ###########
		yanjiu_name_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 486 and y < 1210 and x+w > 486 and y+h > 1210 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 12
					is_ABO = 0
					yanjiu_name_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					yanjiu_name_result = '无'
				print "研究生学校名称: ",yanjiu_name_result
'''
		########### 识别 研究生专业22 ###########
		yanjiu_master_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 696 and y < 1210 and x+w > 696 and y+h > 1210 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 6
					is_ABO = 0
					yanjiu_master_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					yanjiu_master_result = '无'
				print "研究生专业: ",yanjiu_master_result
'''
		########### 识别 研究生学位23 ###########
		yanjiu_degree_result = '无'#功能已实现，因效果一般，暂时注释
		'''
		for j in range(0,len(request_info_set)):
			(x, y, w, h) = request_info_set[j]
			if x < 808 and y < 1210 and x+w > 808 and y+h > 1210 :
				pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
				print "pic_mean[0]:", pic_mean[0]
				if pic_mean[0] > 30:
					get_all = 0
					cha_num = 3
					is_ABO = 0
					yanjiu_degree_result = chinese_out.chinese_o(th2_copy, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO)
				else:
					yanjiu_degree_result = '无'
				print "研究生学位: ",yanjiu_degree_result
'''
		########### 识别 研究生起止时间24 ###########
		yanjiu_time_result = '无'
		if table_style == 1:
			x_com = 931
			y_com = 1210
		if table_style == 2:
			if len(docu_num) == 9 and (docu_num[6] == '2' and (docu_num[7] == '6' or docu_num[7] == '8' or docu_num[7] == '9')):
				y_com = 1230
			else:
				y_com = 1175
			x_com = 931
		if table_style == 3:
			x_com = 981
			y_com = 1233
		if table_style == 4:
			x_com = 981
			y_com = 1238
		try:
			if need_stronger == 1:
				yanjiu_time_result = '无'
			else:
				for j in range(0,len(request_info_set)):
					(x, y, w, h) = request_info_set[j]
					if x < x_com and y < y_com and x+w > x_com and y+h > y_com :
						pic_mean = cv.mean(th2_copy[y:y+h, x:x+w])
						print "pic_mean[0]:", pic_mean[0]
						if pic_mean[0] > 30:
							num_num = 10
							#调用time_out.py中的num_o()函数进行手写数字识别
							yanjiu_time_result = time_out.num_o(th2_copy, x, y, w, h, num_num)
						else:
							yanjiu_time_result = '无'
						print "研究生起止时间: ",yanjiu_time_result
		except:
			yanjiu_time_result = '无'
			print "研究生起止时间: ",yanjiu_time_result

		########### 识别 研究生是否毕业25 ###########
		left_mean = cv.mean(yanjiu_roi_left)
		right_mean = cv.mean(yanjiu_roi_right)
		print left_mean[0],right_mean[0]
		mean_diff = 14#两者间的差值超过该值则判断为是、否，否则判断为两个空框
		if left_mean[0] < right_mean[0] - mean_diff:
			yanjiu_biye_result = '是'
		else:
			if right_mean[0] < left_mean[0] - mean_diff:
				yanjiu_biye_result = '否'
			else:
				yanjiu_biye_result = '无'
		print "研究生是否毕业: ",yanjiu_biye_result

		########### 识别 其它学校名称26 ###########

		qita_name_result = '无'

		########### 识别 其它专业27 ###########

		qita_master_result = '无'

		########### 识别 其它学位28 ###########

		qita_degree_result = '无'

		########### 识别 其它起止时间29 ###########

		qita_time_result = '无'

		########### 识别 其它是否毕业30 ###########

		qita_biye_result = '无'

                ##### 向CSV文件写入数据 #####
		information = [docu_num, xingbie_result, minzu_result, weight_result, xuexing_result, jiguan_result, gaozhong_name_result, gaozhong_master_result, gaozhong_degree_result, gaozhong_time_result, gaozhong_biye_result, dazhuan_name_result, dazhuan_master_result, dazhuan_degree_result, dazhuan_time_result, dazhuan_biye_result, benke_name_result, benke_master_result, benke_degree_result, benke_time_result, benke_biye_result, yanjiu_name_result, yanjiu_master_result, yanjiu_degree_result, yanjiu_time_result, yanjiu_biye_result, qita_name_result, qita_master_result, qita_degree_result, qita_time_result, qita_biye_result]
		csvwrite.writerow(information)
