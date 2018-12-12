#!/usr/bin/env python  
# encoding: utf-8  

import cv2 as cv
import numpy as np
import math

import time_recognize

def num_o(th2, x, y, w, h, num_num):#对table_choose.py的接口
	cut = ~th2[y:y+h, x:x+w]#从表格中分割出目标矩形框
	copy_cut = ~th2[y:y+h, x:x+w]
	cv.imwrite('cut_ori.jpg', cut)

	##### 水平投影分割 #####
	b = [0 for z in range(0, h)] 
	for j in range(0,h):
		for i in range(0,w):
			if copy_cut[j,i]==0:
				b[j]+=1 
				copy_cut[j,i]=255     
	print b    
	
	blank_counting = 0
	for j in range(3,h-3):
		for i in range(3,w-3):
			if cut[j,i]==0:
				blank_counting+=1 
	print "blank_counting/w:", blank_counting/w
	if float(blank_counting)/w <= 0.5:#排除空框
		return '无'
	else:
		print "sum(b)/(8*h) = ",sum(b)/(4*h)  #4
		word_start = 0
		word_b_set = []  
		if sum(b)/(4*h) <= 2:#设定水平分割时的最低阈值
			standard = 2
		else:
			standard = sum(b)/(4*h)#4
		for j in range(1,h-1):#水平分割
			if b[j] >= standard and word_start == 0:
				word_start_b = j
				word_start = 1
			if (b[j] < standard and b[j-1] < standard or j == h-2) and word_start == 1:
				word_end_b = j
				word_start = 0
				if abs(word_end_b - word_start_b) >= 10:#行像素超过10则判断为一行手写字
					word_b = (word_start_b, word_end_b)
					word_b_set.append(word_b)
		print word_b_set
				   
		for j in range(0,h):
			for i in range(0,b[j]):
				copy_cut[j,i]=0
		cv.imwrite('copy_cut.jpg', copy_cut)

		##### 垂直投影分割 #####
		word_cut_save = []## ALL ROWS Attention!!!
		for row_cut in range(0,len(word_b_set)):
			word_set = [] 
			word_set_filter = [] 
			#word_cut_save = []## ONE ROW Attention!!!
			cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x:x+w]
			h = abs(word_b_set[row_cut][1] - word_b_set[row_cut][0])
			a = [0 for z in range(0, w)] 
			#print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
			#记录每一列的波峰
			for j in range(0,w): #遍历一列 
				for i in range(0,h):  #遍历一行
					if cut[i,j]==0:  #如果该点为黑点
						a[j]+=1  		#该列的计数器加一计数
						cut[i,j]=255  #记录完后将其变为白色 
			print a
			print sum(a)/(4*w)#2,1.2
			
			if sum(a)/(4*w) <= 4:#设定垂直分割时的最低阈值
				standard = 4
			else:
				standard = sum(a)/(4*w)#2,1.2

			word_cutting = 0
			word_cut_counting = 0
			word_cut_counting_over = 0
			for j in range(3,w-2):#垂直分割
				if a[j] >= standard and word_start == 0:
					word_start_a = j
					word_start = 1
				#连续2个像素都低于阈值则设置分割点
				if word_cutting == 1 and a[j] < standard and a[j-1] < standard:
					word_cut_end = j
					word_cut_counting+=1
				if a[j] < standard and a[j-1] < standard and word_start == 1 and word_cutting == 0 and word_cut_counting_over == 0:
					word_cut_begin = j-1
					word_cut_end = j
					word_cutting = 1
					word_cut_counting = 1
				if (a[j] >= standard and (word_cutting == 1 or word_cut_counting_over == 1)) or (word_cut_counting >= num_num and word_cutting == 1) or (j == w-3 and word_cut_counting_over == 0):
					if word_cut_counting >= num_num:
						word_cut_counting_over = 1
					else:
						word_cut_counting_over = 0
					word_cutting = 0
					word_cut_counting = 0
					if j == w-3 and word_cut_counting_over == 0:
						word_end_a = j
					else:
						word_end_a = (word_cut_begin + word_cut_end)//2
					if abs(word_end_a - word_start_a) >= 3:#列像素超过3则判断为一个手写字
						word = (word_start_a, word_end_a, word_b_set[row_cut][0], word_b_set[row_cut][1])
						word_set.append(word)#将分割结果存储入数组
					word_start_a = word_end_a
			print "word_set:",word_set
			word_count = 0
			for j in range(0,len(word_set)):
				word_a_counting = 0
				for i in range(word_set[j][0], word_set[j][1]):
					word_a_counting+= a[i]
				print "word_a_counting:",j,word_a_counting
				if word_a_counting >= 60:#当垂直投影量累计超过60则进入切割数字流程（排除空栏）
					word_cut = ~th2[y + word_set[j][2]:y + word_set[j][3], x + word_set[j][0]:x + word_set[j][1]]
					word_set_filter.append(word_set[j])
					#由于数字间一般有间隔，通过分割各连通域可得各数字
					_, contours, hierarchy = cv.findContours(~word_cut,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
					print "contours:",len(contours)
					if len(contours) > 1 :#数字中有封闭区域时1个数字会识别出多个连通域
						contours_data_set = []
						contours_outside_set = []
						inside_set = []
						#分析各连通域在x方向上互相之间是否存在包含关系
						for c_j in range(0,len(contours)):
							(x0, y0, w0, h0) = cv.boundingRect(contours[c_j])
							#cv.rectangle(word_cut,(x0,y0),(x0 + w0,y0 + h0),(0,0,0),1)
							contours_data = (x0, y0, w0, h0)
							contours_data_set.append(contours_data)
						print contours_data_set
						for c_j in range(0,len(contours_data_set)):
							for c_i in range(0,len(contours_data_set)):
								#if c_j != c_i and contours_data_set[c_j][0] <= contours_data_set[c_i][0] and contours_data_set[c_j][1] <= contours_data_set[c_i][1] and contours_data_set[c_j][0] + contours_data_set[c_j][2] >= contours_data_set[c_i][0] + contours_data_set[c_i][2] and contours_data_set[c_j][1] + contours_data_set[c_j][3] >= contours_data_set[c_i][1] + contours_data_set[c_i][3]:
								if c_j != c_i and contours_data_set[c_j][0] <= contours_data_set[c_i][0] and contours_data_set[c_j][0] + contours_data_set[c_j][2] >= contours_data_set[c_i][0] + contours_data_set[c_i][2] :
									if not c_i in inside_set:
										inside_set.append(c_i)
						for c_j in range(0,len(contours_data_set)):
							if not c_j in inside_set:
								contours_outside_set.append(contours_data_set[c_j])
						print contours_outside_set
						#取x方向上未被其他连通域包含在内侧的连通域
						if len(contours_outside_set)<=1 and len(word_set)==1:#若连通域仅1个，因表格中无单个数据，则平均切分成2段
							for c_j in range(0,2):
								word_cut = ~th2[y + word_set[j][2]:y + word_set[j][3], x + word_set[j][0] + (word_set[j][1]-word_set[j][0])*c_j/2:x + word_set[j][0] + (word_set[j][1]-word_set[j][0])*(c_j+1)/2]
								cv.imwrite('word_cut' + str(word_count) + '.jpg', word_cut)
								word_cut_save.append(word_cut)
								word_count+=1
						else:#若有多个连通域
							for c_i in range(len(contours_outside_set)-1):#依照连通域在x轴上的顺序进行冒泡排序Bubble Sort
								for c_j in range(len(contours_outside_set)-1-c_i):
									if contours_outside_set[c_j][0] > contours_outside_set[c_j+1][0]:
										contours_outside_set[c_j], contours_outside_set[c_j+1] = contours_outside_set[c_j+1], contours_outside_set[c_j]
							
							for c_j in range(0,len(contours_outside_set)):#切分数字
								wcut_a_counting = 0
								for wcut_i in range(word_set[j][0] + contours_outside_set[c_j][0], word_set[j][0] + contours_outside_set[c_j][0] + contours_outside_set[c_j][2]):
									wcut_a_counting+= a[wcut_i]
								print 'wcut_a_counting:',wcut_a_counting
								if wcut_a_counting > 120:#当垂直投影量累计超过120时则保存该段切分图像（排除干扰笔画）
									word_cut = ~th2[y + word_set[j][2]:y + word_set[j][3], x + word_set[j][0] + contours_outside_set[c_j][0]:x + word_set[j][0] + contours_outside_set[c_j][0] + contours_outside_set[c_j][2]]
									cv.imwrite('word_cut' + str(word_count) + '.jpg', word_cut)
									word_cut_save.append(word_cut)
									word_count+=1
					else:#数字中无有封闭区域时，直接保存该段图像
						cv.imwrite('word_cut' + str(word_count) + '.jpg', word_cut)
						word_cut_save.append(word_cut)
						word_count+=1
			#统计分割字段总长度、平均长度
			word_cut_TotalLength = 0
			for j in range(0,len(word_cut_save)):
				rows_word_cut,cols_word_cut = word_cut_save[j].shape
				print "len(word_cut_save[j]):",cols_word_cut
				word_cut_TotalLength+= cols_word_cut
				
			if len(word_cut_save) == 0:#排除空栏
				return ''
			
			word_cut_AverageLength = word_cut_TotalLength/len(word_cut_save)
			print "word_cut_AverageLength:",word_cut_AverageLength
			word_over_TotalLength = 0
			word_over_num = 0
			for j in range(0,len(word_cut_save)):
				rows_word_cut,cols_word_cut = word_cut_save[j].shape
				if cols_word_cut > 1.5*word_cut_AverageLength:#统计长度超过字段平均长度1.5倍的字段的总长度与个数
					word_over_TotalLength+= cols_word_cut
					word_over_num+= 1
			inside_position_set = []
			if len(word_cut_save) < num_num:#若分割出的数字个数少于设定的数字个数，则进入切割判断流程
				word_over_AverageLength = word_over_TotalLength/(num_num-len(word_cut_save) + word_over_num)
				#print word_over_AverageLength,word_over_TotalLength,8-len(word_cut_save)
				for j in range(0,len(word_cut_save)):
					rows_word_cut,cols_word_cut = word_cut_save[j].shape
					#print cols_word_cut
					if cols_word_cut > 2*word_cut_AverageLength:#若该段长度超过字段平均长度的1.5倍，进入切割流程
						should_cut_num = cols_word_cut//word_over_AverageLength#根据字段平均长度求得需切分出的字段数
						
						a_detail = [0 for z_detail in range(0, cols_word_cut)] 
						for j_detail in range(0,cols_word_cut): #遍历一列 
							for i_detail in range(0,rows_word_cut):  #遍历一行
								if word_cut_save[j][i_detail,j_detail]==0:  #如果该点为黑点
									a_detail[j_detail]+=1  		#该列的计数器加一计数
                                                #遍历寻找垂直投影字段中的峰谷
						peak_set = []
						for a_searching in range(1,cols_word_cut-1):
							if a_detail[a_searching-1] > a_detail[a_searching] and a_detail[a_searching] < a_detail[a_searching+1]:
								peak = (a_searching, a_detail[a_searching])
								peak_set.append(peak)
						if len(peak_set) == 0 and should_cut_num != 0:
							for cut_i in range(0,should_cut_num):
								#print int((cut_i+1)*cols_word_cut/should_cut_num),len(a_detail)
								peak = (int((cut_i+1)*cols_word_cut/should_cut_num-1), a_detail[int((cut_i+1)*cols_word_cut/should_cut_num-1)])
								peak_set.append(peak)
						for c_i in range(len(peak_set)-1):#Bubble Sort
							for c_j in range(len(peak_set)-1-c_i):
								if peak_set[c_j][1] > peak_set[c_j+1][1]:
									peak_set[c_j], peak_set[c_j+1] = peak_set[c_j+1], peak_set[c_j]
						if should_cut_num > len(peak_set):
							should_cut_num = len(peak_set)
						#对峰谷进行冒泡排序，找出最大的所需个数的峰谷进行切分
						peak_set_select = []
						for c_i in range(0,should_cut_num):
							peak_set_select.append(peak_set[c_i])
						for c_i in range(len(peak_set_select)-1):#Bubble Sort
							for c_j in range(len(peak_set_select)-1-c_i):
								if peak_set_select[c_j][0] > peak_set_select[c_j+1][0]:
									peak_set_select[c_j], peak_set_select[c_j+1] = peak_set_select[c_j+1], peak_set_select[c_j]
						#print "peak_set_select:",peak_set_select
						for c_i in range(0,should_cut_num):
							print peak_set_select,should_cut_num,c_i,"peak_set:",peak_set_select[c_i]
							if c_i == 0:
								word_cut = word_cut_save[j][0:rows_word_cut,0:peak_set_select[c_i][0]]
							if c_i == should_cut_num-1:
								word_cut = word_cut_save[j][0:rows_word_cut,peak_set_select[c_i-1][0]:cols_word_cut]
							if c_i != 0 and c_i != should_cut_num-1:
								word_cut = word_cut_save[j][0:rows_word_cut,peak_set_select[c_i-1][0]:peak_set_select[c_i][0]]
							cv.imwrite('word_cut' + str(j) + '_detail' + str(c_i) + '.jpg', word_cut)
							inside_position = (j, c_i, word_cut)
							inside_position_set.append(inside_position)
			
			word_cut_result = []
			if len(inside_position_set) == 0:
				word_cut_result = word_cut_save#若未切分出字段，则将原字段存入新数组
			else:
				for j in range(0,len(word_cut_save)):#若切分出字段，则将新字段和未被切分的原字段存入新数组
					check_num = 0
					for i in range(0,len(inside_position_set)):
						if j == inside_position_set[i][0]:
							word_cut_result.append(inside_position_set[i][2])
						else:
							check_num+= 1
						if check_num == len(inside_position_set):
							word_cut_result.append(word_cut_save[j])
			print len(word_cut_result)
			row_result = time_recognize.mnist_recog(word_cut_result)#调用time_recognize.py中的mnist_recog()函数进行手写数字识别
			#print "登记表编号: ",row_result

			for j in range(0,w):  #遍历每一列
				for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
					cut[i,j]=0   #涂黑
			cv.imwrite('cut' + str(row_cut) + '.jpg', cut)
		return row_result#返回该栏识别到的数字字符串
