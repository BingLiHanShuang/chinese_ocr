1、功能描述：
该算法实现的功能是将test_data目录中的简历进行读取与校正，从中监测到表格，并从表格中提取指定类别的内容，将结果输入到csv文件中。
第一行为表头文件：
登记表编号,性别,民族,体重,血型,籍贯,高中学校名称,高中专业,高中学位,高中起止时间,高中是否毕业,大专学校名称,大专专业,大专学位,大专起止时间,大专是否毕业,本科学校名称,本科专业,本科学位,本科起止时间,本科是否毕业,研究生学校名称,研究生专业,研究生学位,研究生起止时间,研究生是否毕业,学校名称（其它）,专业（其它）,学位（其它）,起止时间（其它）,是否毕业（其它）
第二行开始为结果内容，例如：
2015002,男,汉,58 kg,无,山东枣庄,滕州二中,无,无,2004.9—2007.6,是,无,无,无,无,无,山东农业大学,计算机科学与技术,学士,2008.9—2012.7,是,北方工业大学,软件工程,硕士,2012.9—2015.7,无,无,无,无,无,无

2、编译环境如下：
（1）Ubuntu16.04 64位 支持utf-8编码
（2）Python2.7
（3）CUDA9.0
（4）cuDNN 7.1.4
（5）TensorFlow1.8.0 GPU版
（6）OpenCV3.4.3 （与Python2.7编译通）
（7）Python2.7 numpy模块、PIL模块、logging模块、pickle模块、os模块、random模块、time模块、matplotlib模块、math模块、csv模块

3、程序运行方式：
（1）下载模型“model.zip”：https://pan.baidu.com/s/1Q0dPSKILNxPMDn7i2VIhow
或不使用百度云（拷进网址栏）：https://d.pcs.baidu.com/file/8c755e26ce17286ed911405a7644f73f?fid=4031771572-250528-760858459813394&dstime=1544602011&rt=sh&sign=FDtAERVY-DCb740ccc5511e5e8fedcff06b081203-GHzypluGZoBNdm99UChQOUdYJJ0%3D&expires=8h&chkv=1&chkbd=0&chkpc=et&dp-logid=8017096450883646029&dp-callid=0&shareid=3190055425&r=806607689
（2）将“model.zip”中的“checkpoint”、“log”、“mnist”三个文件夹与装有测试数据集的“test_data”个文件夹、.py文件放在同一目录下；
（3）命令行进入该目录，并输入：python2 table_choose.py；
（4）等待一段时间，结果保存在该目录下的“try1010.csv”文件中。

4、项目调用结构图：见当前目录下“Project_call_structure_diagram.jpg”
     程序流程图：见当前目录下“Program_flow_chart.jpg”

5、算法详细介绍：
如“Program_flow_chart.jpg”所示，该算法的主体思路是：通过对旋转、平移校正的表格用开操作提取网格，获取各信息所在的矩形位置；根据各信息对应的大致中心坐标筛选出对应的矩形，进行文字分割后，将汉字、ABO字母、数字分别输入卷积神经网络进行识别，得到的结果进行校验后即写入CSV文件。
程序主要分成九个Python文件：ABO_mix.py实现的是基于卷积神经网络的手写ABO字母识别，chinese_ocr.py实现的是基于卷积神经网络的手写汉字识别，chinese_out.py实现的是对手写汉字、字母的分割，chinese_out_jiguan.py实现的是对手写汉字（籍贯部分）的分割，mnist_recognize.py实现的是基于卷积神经网络的手写数字识别，num_out.py实现的是对手写数字的分割，time_out.py实现的是对手写时间数字的分割，time_recognize.py实现的是基于卷积神经网络的手写时间数字识别，table_choose.py实现的是表格预处理、函数调用与识别结果的写入。

模型的主函数位于table_choose.py，基于程序流程图，此处对算法的具体思路进行介绍：
算法始于table_choose.py表格预处理，预处理后对各要求信息进行识别。性别、民族、血型的识别都先通过chinese_out.py进行汉字、字母分割，其中血型部分进入ABO_mix.py进行ABO字母识别，民族和性别信息则进入chinese_ocr.py进行手写汉字识别；籍贯的识别通过chinese_out_jiguan.py进行汉字（籍贯部分）分割，而后也进入chinese_ocr.py进行手写汉字识别；体重的识别通过num_out.py进行数字分割，之后进入mnist_recognize.py进行手写数字识别；高中、大专、本科、研究生、其它起止时间的识别则先通过time_out.py进行时间数字分割，再进入time_recognize.py进行手写时间数字识别；高中、大专、本科、研究生是否毕业部分的信息通过table_choose.py对“是”“否”框的均值判断得出结论；其余部分的识别暂时注释。table_choose.py结果写入部分将所有识别到的信息汇总，写入CSV文件。

