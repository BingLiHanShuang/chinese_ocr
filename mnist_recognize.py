# encoding: utf-8

import tensorflow as tf
import cv2
from PIL import Image
import numpy as np

def normalizepic(pic):#图像预处理：归一化
    im_arr = pic
    im_nparr = []
    for x in im_arr:
        x=1-x/255
        im_nparr.append(x)
    im_nparr = np.array([im_nparr])
    return im_nparr
    
def weight_variable(shape):#weight参数
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,dtype=tf.float32,name='weight')
def bias_variable(shape):#biases参数
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name='biases')
def conv2d(x,W):#卷积层
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):#池化层
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def mnist_recog(num_img_set):#对num_out.py的接口
	tf.reset_default_graph()
	
	xs=tf.placeholder(tf.float32,[None,784])#输入是一个28*28的像素点的数据
	keep_prob=tf.placeholder(tf.float32)
	x_image=tf.reshape(xs,[-1,28,28,1])#xs的维度暂时不管，用-1表示，28,28表示xs的数据，1表示该数据是一个黑白照片，如果是彩色的，则写成3
	#卷积层1
	W_conv1=weight_variable([5,5,1,32])#抽取一个5*5像素，高度是32的点,每次抽出原图像的5*5的像素点，高度从1变成32
	b_conv1=bias_variable([32])
	h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#输出 28*28*32的图像
	h_pool1=max_pool_2x2(h_conv1)##输出14*14*32的图像，因为这个函数的步长是2*2，图像缩小一半。
	#卷积层2
	W_conv2=weight_variable([5,5,32,64])#随机生成一个5*5像素，高度是64的点,抽出原图像的5*5的像素点，高度从32变成64
	b_conv2=bias_variable([64])
	h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#输出14*14*64的图像
	h_pool2=max_pool_2x2(h_conv2)##输出7*7*64的图像，因为这个函数的步长是2*2，图像缩小一半。
	#fully connected 1
	W_fc1=weight_variable([7*7*64,1024])
	b_fc1=bias_variable([1024])
	h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#将输出的h_pool2的三维数据变成一维数据，平铺下来，（-1）代表的是有多少个例子
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
	#fully connected 2
	W_fc2=weight_variable([1024,10])
	b_fc2=bias_variable([10])
	prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)#输出层
	
	num_result = ''

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, "/home/psps/chinese_OCR/mnist/mnist_model.ckpt")#调用先前训练好的手写数字识别模型
		result = tf.argmax(prediction, 1)
		for num in range(0,len(num_img_set)):#一一读取输入的图像组
			img = num_img_set[num]
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
			#print b
			'''
			start_get = 0
			end_get = 0
			for j in range(0,h):
				if b[j] >= 2 and start_get != 1:
					word_start_b = j
					start_get = 1
				if j-2 > 0 and b[j] < 2 and b[j-1] < 2 and b[j-2] < 2 and start_get == 1 and end_get != 1:
					word_end_b = j-2
					end_get = 1
			'''
	
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
			print w_total
			
			if w_total > 0:#排除空栏
				#print "h/w:", float(h)/float(w)#1.78571428571
				smaller = 0.9#单个数字缩小倍数
				
				#将数字平移至图像中央
				affineShrinkTranslation = np.array([[1, 0, int(w//2 - w_average)], [0, 1, int(h//2 - h_average)]], np.float32)
				shrinkTwoTimesTranslation = cv2.warpAffine(~img, affineShrinkTranslation, (w, h))
				shrinkTwoTimesTranslation = cv2.resize(shrinkTwoTimesTranslation,(int(w*smaller),int(h*smaller)),interpolation=cv2.INTER_AREA)
				cv2.imwrite('bin_shrinkTwoTimesTranslation.jpg', shrinkTwoTimesTranslation)
				max_one = max(h,w)
				bin = np.zeros((max_one,max_one), np.uint8)
				bin.fill(0)
				
				rows_count = 0
				for j in range(int(max_one//2-h*smaller/2), int(max_one//2+h*smaller/2)):
					cols_count = 0
					for i in range(int(max_one//2-w*smaller/2), int(max_one//2+w*smaller/2)):
						if cols_count <= int(w*smaller)-1 and rows_count <= int(h*smaller)-1:
							#print i,j
							bin[j][i] = shrinkTwoTimesTranslation[rows_count][cols_count]
							cols_count+= 1
					rows_count+= 1
							
				cv2.imwrite('bin_ori.jpg', bin)
				#bin=cv2.cvtColor(bin,cv2.COLOR_BGR2GRAY)
				bin = cv2.resize(~bin,(28,28),interpolation=cv2.INTER_AREA)
				cv2.imwrite('bin.jpg', bin)
				img=normalizepic(bin).reshape((1,784))#OpenCV Mat转PIL
				img= img.astype(np.float32)
				result = sess.run(prediction, feed_dict={xs:img,keep_prob: 1.0})#识别
				result1 = np.argmax(result,1)
				print 'result1[0]:',result1[0]
				num_result = num_result + str(result1[0])#识别出的结果一一跟随于先前识别结果后方
	return num_result

