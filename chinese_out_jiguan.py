#coding:utf-8
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

import ABO_mix

import chinese_ocr

def chinese_o(th2, shrinkTwoTimesTranslation_copy, x, y, w, h, cha_num, get_all, is_ABO):
	return_change = 0
	chinese_num_lim = 0
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
	
	b_total = 0
	for j in range(2,h-2):
		b_total = b_total + b[j]
			
	blank_counting = 0
	for j in range(3,h-3):
		for i in range(3,w-3):
			if cut[j,i]==0:
				blank_counting+=1 
	print "blank_counting/w:", blank_counting/w
	if float(blank_counting)/w <= 0.5:#排除空框
		return return_change, '无'
	else:
		print sum(b)/(8*h)  #4
		word_start = 0
		word_b_set = []  
		if sum(b)/(8*h) <= 3:#设定水平分割时的最低阈值
			standard = 3
		else:
			standard = sum(b)/(8*h)#4
		for j in range(2,h-2):
			if b[j] >= standard and word_start == 0:
				word_start_b = j
				word_start = 1
			#if (b[j] < standard or j == h-3) and word_start == 1:
			if (b_total < standard or j == h-3) and word_start == 1:
				word_end_b = j
				word_start = 0
				if abs(word_end_b - word_start_b) >= 10:#行像素超过10则判断为一行手写字
					word_b = (word_start_b, word_end_b)
					word_b_set.append(word_b)
			b_total = b_total - b[j]
		print word_b_set
				   
		for j in range(0,h):
			for i in range(0,b[j]):
				copy_cut[j,i]=0
		cv.imwrite('copy_cut.jpg', copy_cut)

		##### 垂直投影分割 #####
		word_set = [] 
		word_cluster_cut_set = []
		word_cluster_color_cut_set = []
		word_count = 0
		for row_cut in range(0,len(word_b_set)):
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
			print sum(a)/(2*w)#2
			
			if sum(a)/(2*w) <= 3:#设定垂直分割时的最低阈值
				standard = 3#3
			else:
				standard = sum(a)/(2*w)#2

			word_cutting = 0
			word_cut_counting = 0
			word_cut_counting_over = 0
			word_total_length = 0
			word_start_a = 3#避免受边框的干扰
			for j in range(3,w-2):#垂直分割
				if a[j] >= standard and word_start == 0:
					word_start_a = j
					word_start = 1
				#连续三个像素都低于阈值则设置分割点
				if word_cutting == 1 and a[j] < standard and a[j-1] < standard and a[j-2] < standard:
					word_cut_end = j
					word_cut_counting+=1
				if a[j] < standard and a[j-1] < standard and a[j-2] < standard and word_start == 1 and word_cutting == 0 and word_cut_counting_over == 0:
					word_cut_begin = j-2
					word_cut_end = j
					word_cutting = 1
					word_cut_counting = 1
				'''
				if word_cutting == 1 and a[j] < standard and a[j-1] < standard:
					word_cut_end = j
					word_cut_counting+=1
				if a[j] < standard and a[j-1] < standard and word_start == 1 and word_cutting == 0 and word_cut_counting_over == 0:
					word_cut_begin = j-1
					word_cut_end = j
					word_cutting = 1
					word_cut_counting = 1
				'''
				if (a[j] >= standard and (word_cutting == 1 or word_cut_counting_over == 1)) or (word_cut_counting >= cha_num and word_cutting == 1) or (j == w-3 and word_cut_counting_over == 0):
					if word_cut_counting >= cha_num:
						word_cut_counting_over = 1
					else:
						word_cut_counting_over = 0
					word_cutting = 0
					word_cut_counting = 0
					if j == w-3 and word_cut_counting_over == 0:
						word_end_a = j
					else:
						word_end_a = (word_cut_begin + word_cut_end)//2

					if abs(word_end_a - word_start_a) >= 12:#列像素超过12则判断为一个手写字
						word = (word_start_a, word_end_a, word_b_set[row_cut][0], word_b_set[row_cut][1])
						word_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_start_a:x + word_end_a]
						word_cut_color = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_start_a:x + word_end_a]
						word_cut_mean = cv.mean(word_cut)
						print "word_cut_mean[0]:", word_cut_mean[0]
						if word_cut_mean[0] < 245:#排除空框
							word_set.append(word)
							cv.imwrite('word_cut' + str(word_count) + '.jpg', word_cut)#保存分割出的二值化图像
							cv.imwrite('cut_color' + str(word_count) + '.jpg', word_cut_color)#保存分割出的灰度图像
							word_count+=1
					word_start_a = word_end_a
			print word_set
			
			##### 合并与切分Clustering or Cutting #####
			word_length_set = []
			word_cut_position = []
			#统计分割字段总长度、平均长度、最小长度
			word_shortest_length = word_set[0][1] - word_set[0][0]
			for j in range(0,len(word_set)):
				word_length = word_set[j][1] - word_set[j][0]
				word_total_length = word_total_length + word_length
				word_length_set.append(word_length)
				if word_length < word_shortest_length and float(word_b_set[row_cut][1] - word_b_set[row_cut][0])/word_length <= 2:
					word_shortest_length = word_length
			word_average_length = word_total_length/len(word_set)
			#print word_set,len(word_set),cha_num
			word_cluster_count = 0
			if len(word_set) >= cha_num:#当分割个数超过设定的个数则偏向合并，否则切分
				word_Cluster2Right = 0
				for j in range(0,len(word_set)):
                                        #当单个字段长度低于平均长度的0.75时，将字段向左/右合并，此处先标记合并后的新分割点
					if word_length_set[j] < word_average_length*0.75:#1.25,0.75
						if j != len(word_set)-1 and word_length_set[j-1] >= word_length_set[j+1]:#向右合并
							if j-1 < 0:#考虑边缘情况
								cut_position = word_set[j][0]
								word_cut_position.append(cut_position)
							word_Cluster2Right = 1
							print 1
						if j == len(word_set)-1 or word_length_set[j-1] < word_length_set[j+1]:#向左合并
							if word_Cluster2Right == 1:#考虑前一块向右聚合的情况
								cut_position = word_set[j][1]
								word_cut_position.append(cut_position)
								print 2
							else:
								if j-1 < 0:
									cut_position = word_set[j][0]
									word_cut_position.append(cut_position)
								else:
									if len(word_cut_position) == 0:
										cut_position = word_set[j][1]
										word_cut_position.append(cut_position)
									else:
										del word_cut_position[len(word_cut_position)-1]
										cut_position = word_set[j][1]
										word_cut_position.append(cut_position)
								print 3
							word_Cluster2Right = 0	
					else:
						if j-1 < 0:#不合并，且在边缘位置
							cut_position = word_set[j][0]
							word_cut_position.append(cut_position)
							cut_position = word_set[j][1]
							word_cut_position.append(cut_position)
							print 4
						else:#不合并，不在边缘位置
							cut_position = word_set[j][1]
							word_cut_position.append(cut_position)
							print 5
						word_Cluster2Right = 0
					#print word_cut_position
				print word_cut_position
				word_cluster_set = []
				#word_cluster_count = 0
				#根据合并后的新标记点对该行进行重新切分
				for j in range(0,len(word_cut_position)-1):
					word_cluster = (word_cut_position[j], word_cut_position[j+1], word_b_set[row_cut][0], word_b_set[row_cut][1])
					word_cluster_set.append(word_cluster)
					word_cluster_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cut_position[j]:x + word_cut_position[j+1]]
					word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cut_position[j]:x + word_cut_position[j+1]]
					cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
					word_cluster_cut_set.append(word_cluster_cut)
					cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
					word_cluster_color_cut_set.append(word_cluster_color_cut)
					word_cluster_count+=1
				print word_cluster_set
			else:
                                ##### 切分 #####
				word_over_TotalLength = 0
				is_someword_over = 0
				if len(word_set) == 1:#当为单字时
					h = word_set[0][3] - word_set[0][2]
					w = word_set[0][1] - word_set[0][0]
					b = [0 for z in range(0, h)] 
					b_total = 0
					single_cut = ~th2[y + word_set[0][2]:y + word_set[0][3], x + word_set[0][0]:x + word_set[0][1]]
                                        #通过对整体水平、垂直投影量切去除无字空白部分
					for j in range(0,h):
						for i in range(0,w):
							if single_cut[j,i] == 0:
								b[j]+= 1 
								b_total+= 1 
					start_already = 0
					b_before = 0
					cut_finish = 0
					for j in range(0,h):
						b_before+= b[j]#统计该列前方的总水平投影量
						b_after = b_total - b_before#统计该列后方的总水平投影量
						if b_before > 4 and start_already == 0 and cut_finish != 1:
							b_cut_start = j
							start_already = 1
						if start_already == 1 and b_after < 4 and cut_finish != 1:
							b_cut_end = j
							cut_finish = 1
					a = [0 for z in range(0, w)] 
					a_total = 0
					for j in range(0,w):
						for i in range(0,h):
							if single_cut[i,j] == 0:
								a[j]+= 1 
								a_total+= 1 
					start_already = 0
					a_before = 0
					cut_finish = 0
					for j in range(0,w):
						a_before+= a[j]#统计该列前方的总垂直投影量
						a_after = a_total - a_before#统计该列后方的总垂直投影量
						if a_before > 4 and start_already == 0 and cut_finish != 1:
							a_cut_start = j
							start_already = 1
						if start_already == 1 and a_after < 4 and cut_finish != 1:
							a_cut_end = j
							cut_finish = 1
					#当垂直投影长度/水平投影长度大于2倍时，则切分为2个字段
					print "len(word_set) == 1:",float(a_cut_end - a_cut_start)/(b_cut_end - b_cut_start)
					if float(a_cut_end - a_cut_start)/(b_cut_end - b_cut_start) > 2:#1.8
						word_part_length = int((a_cut_end - a_cut_start)/2)
						for i in range(0,2):
							word_cluster_cut = ~th2[y + b_cut_start + word_b_set[row_cut][0]:y + b_cut_end + word_b_set[row_cut][0], x + word_set[0][0] + a_cut_start + i*word_part_length:x + word_set[0][0] + a_cut_start + (i+1)*word_part_length]
							word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + b_cut_start + word_b_set[row_cut][0]:y + b_cut_end + word_b_set[row_cut][0], x + word_set[0][0] + a_cut_start + i*word_part_length:x + word_set[0][0] + a_cut_start + (i+1)*word_part_length]
							cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
							word_cluster_cut_set.append(word_cluster_cut)
							cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
							word_cluster_color_cut_set.append(word_cluster_color_cut)
							word_cluster_count+=1
					else:#垂直投影长度/水平投影长度小于等于2倍时，则保持单字段
						word_cluster_set = word_set
						for j in range(0,len(word_cluster_set)):
							word_cluster_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
							word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
							cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
							word_cluster_cut_set.append(word_cluster_cut)
							cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
							word_cluster_color_cut_set.append(word_cluster_color_cut)
							word_cluster_count+=1
				else:#当为多个字时
					for j in range(0,len(word_set)):#是否有超过最短字段长度2倍的字段
						if word_length_set[j] > word_shortest_length*2.0 and len(word_set)>1:#1.75
							word_over_TotalLength+= word_length_set[j]
							is_someword_over = 1
					#print word_over_TotalLength,word_average_length
					if is_someword_over == 1:#若有超过最短字段2倍的字段，进入分割流程
						#word_cluster_count = 0
						word_cluster_set = word_set
						for j in range(0,len(word_cluster_set)):
							if word_length_set[j] > word_shortest_length*3.0 :#对长度超过最短字段长度3倍的字段进行平均切分
								word_part_length = int(word_length_set[j]/math.ceil(float(word_length_set[j])/word_shortest_length))#word_average_length
								for i in range(0,int(math.ceil(float(word_length_set[j])/word_shortest_length))):#word_average_length
									word_cluster_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0] + i*word_part_length:x + word_cluster_set[j][0] + (i+1)*word_part_length]
									word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0] + i*word_part_length:x + word_cluster_set[j][0] + (i+1)*word_part_length]
									cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
									word_cluster_cut_set.append(word_cluster_cut)
									cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
									word_cluster_color_cut_set.append(word_cluster_color_cut)
									word_cluster_count+=1
							else:#对长度不超过最短字段长度3倍的字段，保持原状
								word_cluster_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
								word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
								cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
								word_cluster_cut_set.append(word_cluster_cut)
								cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
								word_cluster_color_cut_set.append(word_cluster_color_cut)
								word_cluster_count+=1
					else:#若没有超过最短字段2倍的字段，则保持原状
						word_cluster_set = word_set
						for j in range(0,len(word_cluster_set)):
							word_cluster_cut = ~th2[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
							word_cluster_color_cut = shrinkTwoTimesTranslation_copy[y + word_b_set[row_cut][0]:y + word_b_set[row_cut][1], x + word_cluster_set[j][0]:x + word_cluster_set[j][1]]
							cv.imwrite('cluster_word' + str(word_cluster_count) + '.jpg', word_cluster_cut)
							word_cluster_cut_set.append(word_cluster_cut)
							cv.imwrite('color_cluster' + str(word_cluster_count) + '.jpg', word_cluster_color_cut)
							word_cluster_color_cut_set.append(word_cluster_color_cut)
							word_cluster_count+=1

			for j in range(0,w):  #遍历每一列
				for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
					cut[i,j]=0   #涂黑
			cv.imwrite('cut' + str(row_cut) + '.jpg', cut)
		
		chinese_result = ''
		last_cha = ''
		last_cha_set = []
		if is_ABO == 1:#若是在判断血型，因该Python文件用于识别籍贯部分的汉字，因此未用上该部分
			### Last word detection ###
			img = word_cluster_cut_set[len(word_cluster_cut_set)-1]
			img_color = word_cluster_color_cut_set[len(word_cluster_cut_set)-1]
			smaller = 0.9#0.9--normal,0.5--ABO
			
			#Place to middle
			h, w = img.shape
			b = [0 for z in range(0, h)] 
			h_total = 0
			h_num = 0
			for j in range(0,h):
				for i in range(0,w):
					if img[j,i] == 0:
						b[j]+= 1 
						h_num+= 1
						h_total+= j
			print "h_num:",h_num
			if h_num != 0:
			
				h_average = h_total//h_num
				
				a = [0 for z in range(0, w)] 
				w_total = 0
				w_num = 0
				for j in range(0,w):
					for i in range(0,h):
						if img[i,j] == 0:
							a[j]+= 1 
							w_num+= 1
							w_total+= j
				w_average = w_total//w_num
						
				affineShrinkTranslation = np.array([[1, 0, int(w//2 - w_average)], [0, 1, int(h//2 - h_average)]], np.float32)
				shrinkTwoTimesTranslation = cv.warpAffine(~img, affineShrinkTranslation, (w, h))
				shrinkTwoTimesTranslation = cv.resize(shrinkTwoTimesTranslation,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
				cv.imwrite('bin_shrinkTwoTimesTranslation.jpg', shrinkTwoTimesTranslation)
				
				shrinkTwoTimesTranslation_color = cv.warpAffine(~img_color, affineShrinkTranslation, (w, h))
				shrinkTwoTimesTranslation_color = ~cv.resize(shrinkTwoTimesTranslation_color,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
				cv.imwrite('bin_shrinkTwoTimesTranslation_color.jpg', shrinkTwoTimesTranslation_color)
				
				max_one = max(h,w)
				bin = np.zeros((max_one,max_one), np.uint8)
				bin.fill(0)
				
				bin_color = np.zeros((max_one,max_one,3), np.uint8)
				bin_color.fill(255)
				
				rows_count = 0
				for j in range(int(max_one//2-h*smaller/2), int(max_one//2+h*smaller/2)):
					cols_count = 0
					for i in range(int(max_one//2-w*smaller/2), int(max_one//2+w*smaller/2)):
						if cols_count <= int(w*smaller)-1 and rows_count <= int(h*smaller)-1:
							#print rows_count,cols_count,j,i
							bin[j][i] = shrinkTwoTimesTranslation[rows_count][cols_count]
							#print shrinkTwoTimesTranslation_color[rows_count][cols_count]
							bin_color[j][i] = shrinkTwoTimesTranslation_color[rows_count][cols_count]
							cols_count+= 1
					rows_count+= 1

				cv.imwrite('bin_ori.jpg', ~bin)
				cv.imwrite('bin_ori_color.jpg', bin_color)
				
				img = Image.fromarray(cv.cvtColor(bin_color,cv.COLOR_BGR2GRAY))
				
				last_cha_set = chinese_ocr.inference(img, 1)
			if len(last_cha_set) != 0:
				#print len(last_cha_set)
				if (last_cha_set[0] == 689 or last_cha_set[1] == 689 or last_cha_set[2] == 689):
					last_cha = '型'
					finding_len = len(word_cluster_cut_set) - 1
				else:
					last_cha = ''
					finding_len = len(word_cluster_cut_set)
			else:
				last_cha = ''
				finding_len = len(word_cluster_cut_set)
			
			### Letter detection ###
			for num in range(0,finding_len):
				img = word_cluster_cut_set[num]
				img_color = word_cluster_color_cut_set[num]
				smaller = 0.5#0.9--normal,0.5--ABO
				
				#Place to middle
				h, w = img.shape
				b = [0 for z in range(0, h)] 
				h_total = 0
				h_num = 0
				for j in range(0,h):
					for i in range(0,w):
						if img[j,i] == 0:
							b[j]+= 1 
							h_num+= 1
							h_total+= j
				if h_num == 0:
						continue
				h_average = h_total//h_num
				
				a = [0 for z in range(0, w)] 
				w_total = 0
				w_num = 0
				for j in range(0,w):
					for i in range(0,h):
						if img[i,j] == 0:
							a[j]+= 1 
							w_num+= 1
							w_total+= j
				w_average = w_total//w_num
				
				affineShrinkTranslation = np.array([[1, 0, int(w//2 - w_average)], [0, 1, int(h//2 - h_average)]], np.float32)
				shrinkTwoTimesTranslation = cv.warpAffine(~img, affineShrinkTranslation, (w, h))
				shrinkTwoTimesTranslation = cv.resize(shrinkTwoTimesTranslation,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
				cv.imwrite('bin_shrinkTwoTimesTranslation.jpg', shrinkTwoTimesTranslation)
				
				shrinkTwoTimesTranslation_color = cv.warpAffine(~img_color, affineShrinkTranslation, (w, h))
				shrinkTwoTimesTranslation_color = ~cv.resize(shrinkTwoTimesTranslation_color,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
				cv.imwrite('bin_shrinkTwoTimesTranslation_color.jpg', shrinkTwoTimesTranslation_color)
				
				max_one = max(h,w)
				bin = np.zeros((max_one,max_one), np.uint8)
				bin.fill(0)
				
				bin_color = np.zeros((max_one,max_one,3), np.uint8)
				bin_color.fill(255)
				
				rows_count = 0
				for j in range(int(max_one//2-h*smaller/2), int(max_one//2+h*smaller/2)):
					cols_count = 0
					for i in range(int(max_one//2-w*smaller/2), int(max_one//2+w*smaller/2)):
						if cols_count <= int(w*smaller)-1 and rows_count <= int(h*smaller)-1:
							#print rows_count,cols_count,j,i
							bin[j][i] = shrinkTwoTimesTranslation[rows_count][cols_count]
							#print shrinkTwoTimesTranslation_color[rows_count][cols_count]
							bin_color[j][i] = shrinkTwoTimesTranslation_color[rows_count][cols_count]
							cols_count+= 1
					rows_count+= 1

				cv.imwrite('bin_ori.jpg', ~bin)
				cv.imwrite('bin_ori_color.jpg', bin_color)
				
				img = Image.fromarray(~bin)
				#img = Image.fromarray(~bin)
				
				ABO_result = ABO_mix.ABO_detection(img)
				print ABO_result
				chinese_result = chinese_result + str(ABO_result)
		else:
                        #若不是在判断血型栏
			for num in range(0,len(word_cluster_cut_set)):
				img = word_cluster_cut_set[num]
				img_color = word_cluster_color_cut_set[num]
				smaller = 0.9#0.9--normal,0.5--ABO,0.8--giguan，单字缩小倍数
				
				if smaller > 1:#若放大文字部分
					h, w = img.shape
					b = [0 for z in range(0, h)] 
					b_total = 0
					for j in range(0,h):
						for i in range(0,w):
							if img[j,i] == 0:
								b[j]+= 1 
								b_total+= 1 
					start_already = 0
					b_before = 0
					cut_finish = 0
					for j in range(0,h):
						b_before+= b[j]
						b_after = b_total - b_before
						if b_before > 4 and start_already == 0 and cut_finish != 1:
							b_cut_start = j
							start_already = 1
						if start_already == 1 and b_after < 4 and cut_finish != 1:
							b_cut_end = j
							cut_finish = 1
									
					a = [0 for z in range(0, w)] 
					a_total = 0
					for j in range(0,w):
						for i in range(0,h):
							if img[i,j] == 0:
								a[j]+= 1 
								a_total+= 1 
					start_already = 0
					a_before = 0
					cut_finish = 0
					for j in range(0,w):
						a_before+= a[j]
						a_after = a_total - a_before
						if a_before > 4 and start_already == 0 and cut_finish != 1:
							a_cut_start = j
							start_already = 1
						if start_already == 1 and a_after < 4 and cut_finish != 1:
							a_cut_end = j
							cut_finish = 1
					print a_cut_start,a_cut_end,b_cut_start,b_cut_end
                                        #直接提取出矩形中有文字的部分，周围不留空白
					roi_bigger = img[b_cut_start:b_cut_end,a_cut_start:a_cut_end]
					cv.imwrite('roi_bigger.jpg', roi_bigger)
					roi_bigger_color = img_color[b_cut_start:b_cut_end,a_cut_start:a_cut_end]
					cv.imwrite('roi_bigger_color.jpg', roi_bigger_color)
					bin = ~cv.resize(roi_bigger,(int((a_cut_end-a_cut_start)*smaller),int((b_cut_end-b_cut_start)*smaller)),interpolation=cv.INTER_AREA)
					cv.imwrite('bin_shrinkTwoTimesTranslation.jpg', bin)
					bin_color = cv.resize(roi_bigger_color,(int((a_cut_end-a_cut_start)*smaller),int((b_cut_end-b_cut_start)*smaller)),interpolation=cv.INTER_AREA)
					cv.imwrite('bin_shrinkTwoTimesTranslation_color.jpg', bin_color)
					
				else:#若缩小文字部分
					#将文字部分平移到图像中央Place to middle
					h, w = img.shape
					b = [0 for z in range(0, h)] 
					h_total = 0
					h_num = 0
					for j in range(0,h):
						for i in range(0,w):
							if img[j,i] == 0:
								b[j]+= 1 
								h_num+= 1
								h_total+= j
					if h_num == 0:
						continue
					h_average = h_total//h_num
					
					a = [0 for z in range(0, w)] 
					w_total = 0
					w_num = 0
					for j in range(0,w):
						for i in range(0,h):
							if img[i,j] == 0:
								a[j]+= 1 
								w_num+= 1
								w_total+= j
					w_average = w_total//w_num
					
					#在二值化图像中，将文字部分平移到中央
					affineShrinkTranslation = np.array([[1, 0, int(w//2 - w_average)], [0, 1, int(h//2 - h_average)]], np.float32)
					shrinkTwoTimesTranslation = cv.warpAffine(~img, affineShrinkTranslation, (w, h))
					shrinkTwoTimesTranslation = cv.resize(shrinkTwoTimesTranslation,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
					cv.imwrite('bin_shrinkTwoTimesTranslation.jpg', shrinkTwoTimesTranslation)
					#在灰度图像中，将文字部分平移到中央
					shrinkTwoTimesTranslation_color = cv.warpAffine(~img_color, affineShrinkTranslation, (w, h))
					shrinkTwoTimesTranslation_color = ~cv.resize(shrinkTwoTimesTranslation_color,(int(w*smaller),int(h*smaller)),interpolation=cv.INTER_AREA)
					cv.imwrite('bin_shrinkTwoTimesTranslation_color.jpg', shrinkTwoTimesTranslation_color)
					
					max_one = max(h,w)
					bin = np.zeros((max_one,max_one), np.uint8)
					bin.fill(0)
					bin_color = np.zeros((max_one,max_one,3), np.uint8)
					bin_color.fill(255)

					#缩小文字部分，并放置到一张更大的空白图像中
					rows_count = 0
					for j in range(int(max_one//2-h*smaller/2), int(max_one//2+h*smaller/2)):
						cols_count = 0
						for i in range(int(max_one//2-w*smaller/2), int(max_one//2+w*smaller/2)):
							if cols_count <= int(w*smaller)-1 and rows_count <= int(h*smaller)-1:
								#print rows_count,cols_count,j,i
								bin[j][i] = shrinkTwoTimesTranslation[rows_count][cols_count]
								#print shrinkTwoTimesTranslation_color[rows_count][cols_count]
								bin_color[j][i] = shrinkTwoTimesTranslation_color[rows_count][cols_count]
								cols_count+= 1
						rows_count+= 1

				'''
				horizontalsize = cols / scale
				horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalsize, 1))
				erosion = cv.erode(th2,horizontalStructure,iterations = 1)
				dilation = cv.dilate(erosion,horizontalStructure,iterations = 1)
				'''
				cv.imwrite('bin_ori.jpg', ~bin)
				cv.imwrite('bin_ori_color.jpg', bin_color)
				
				img = Image.fromarray(cv.cvtColor(bin_color,cv.COLOR_BGR2GRAY))#OpenCV Mat转PIL
				#img = Image.fromarray(~bin)
				
				if get_all == 1:
					chinese_set = chinese_ocr.inference(img, get_all)#手写汉字识别，对三个识别结果全接收
					print "chinese_set", chinese_set
				else:
					if get_all == 2 and len(word_cluster_cut_set) == 1:
						chinese_set = chinese_ocr.inference(img, 1)#分割字段唯一时民族识别，接收三个识别结果对应的字典序号
						chinese = chinese_ocr.inference(img, 0)#分割字段唯一时，接收识别出的以汉字字符形式表示的最佳结果
						return_change = str(chinese)
						print "chinese_set", chinese_set
					else:
						chinese = chinese_ocr.inference(img, get_all)#手写汉字识别，只接收识别出的以汉字字符形式表示的最佳结果
						if chinese_num_lim < 3:#控制识别出的汉字个数
							chinese_result = chinese_result + str(chinese)
							chinese_num_lim+=1
	if get_all == 1 or return_change != 0:
		return return_change, chinese_set
	if get_all != 1:
		if last_cha == '型':
			chinese_result = chinese_result + '型'
			return_change = 2
		#print "Chinese_ocr:",chinese_result
		return return_change, chinese_result
