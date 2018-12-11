import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import threading
import socket
import _thread
import ast
import time

s = socket.socket()         # Create a socket object
s.bind(('0.0.0.0', 8888))        # Bind to the port
s.listen(5)
c, addr = s.accept()
#---資料庫抓取區---------------------------------------------------------------------------------------------------------------------------#

sys.path.append("..")#必須在object_detection

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'#目標在這資料夾
CWD_PATH = os.getcwd()#抓取當前工作目錄的路徑
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')#抓取pb,包含使用的模型
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt') #label map位置分別是 當前,training資料夾,裡面的labelmap資料 
NUM_CLASSES = 1 #可識別的目標數量
#label_map 名稱對應
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#將Tensorflow模型加載到內存中。
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
#定義對象檢測分類器的輸入和輸出（即數據）
#輸入是圖像
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#輸出是檢測框，分數和類別
#每個框表示檢測到特定對象的圖像的一部分
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#每個分數代表每個對象的置信水平。
#分數顯示在結果圖像上，同時顯示類標籤。
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#檢測到的對像數
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#---------------------------------------------------------------------------------------------------------------------------------------------#
video = cv2.VideoCapture('http://192.168.0.104:8080/?action=stream')
ret = video.set(3,1280)
ret = video.set(4,720)
lower = np.array([123,69,82])
upper = np.array([151,107,125])

scmode=0
xmax=0.6
xmin=0.4
cmax=0.6
cmin=0.4
old = 0.5
sc=0

def socket_reader():
	global b
	while(True):
		response = c.recv(4096)
		str(response, encoding = "utf-8")
		b=int(bytes.decode(response))


def socket_screen():
	global xmin
	global xmax
	global sc
	global scmode
	global cmax
	global cmin
	while(scmode == 0):
		ret, frame = video.read()
		frame_expanded = np.expand_dims(frame, axis=0)
	
		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: frame_expanded})
		xmin=boxes[0][0][1]
		xmax=boxes[0][0][3]
		#辨識分數
		sc=scores[0][0]
		# Draw the results of the detection (aka 'visulaize the results')
		vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.60)
		cv2.imshow('Object detector', frame)
		if cv2.waitKey(1) == ord('q'):
			break
	while(scmode == 1):
		ret, image = video.read()
		filtered = cv2.inRange(image, lower, upper)
		blurred = cv2.GaussianBlur(filtered, (15, 15), 0)
		(_, cnts, _) = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(cnts) > 0:
			cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
			cen=cv2.minAreaRect(cnt)
			rect = np.int32(cv2.boxPoints(cen))
			cv2.drawContours(image, [rect], -1, (0, 255, 0), 2)
		cv2.imshow('Capture', image)
		cmax=rect[0][0]
		cmin=rect[1][0]
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break

def socket_writer():
	global old
	global b
	global xmin
	global xmax
	global sc
	global scmode
	global cmax
	global cmin
	mode=0
	while(mode==0):
		if sc < 0.6 :
			if b > 30 :
				a = 'go'
			elif b<=30 :
				a = 'right'
		elif sc > 0.6 :
			mode = 1
			print("mode1")
		else:
			print('error')
		bytes(a, encoding = "utf8")
		c.sendall(str.encode(a))
	while(mode==1):
		new = round((xmin+xmax)/2,3)
		offset = old*0.2 + new*0.8
		if offset > 0.6:
			a = 'left'
		elif offset < 0.4:
			a = 'right'
		elif offset >=0.4 and offset <=0.6 and b>5:
			a = 'go'
		elif offset >=0.4 and offset <=0.6 and b<=5:
			scmode=1
			a = 'stop'
			bytes(a, encoding = "utf8")
			c.sendall(str.encode(a))
			mode=2
			print("mode2")
		else:
			print('error')
		bytes(a, encoding = "utf8")
		c.sendall(str.encode(a))
		old = new
	while(mode==2):
		ran=round((cmax+cmin)/2,3)
		if ran > 410:
			a = 'left'
		elif ran < 350:
			a = 'right'
		elif ran >= 350 and ran <=410 and b>=10:
			a = 'go'
		elif ran >= 350 and ran <=410 and b<10:
			a = 'doput'
			bytes(a, encoding = "utf8")
			c.sendall(str.encode(a))
			print("done")
			mode=3
		bytes(a, encoding = "utf8")
		c.sendall(str.encode(a))
	while(mode==3):
		a = 'end'
		bytes(a, encoding = "utf8")
		c.sendall(str.encode(a))
		break




def main():
	tha = threading.Thread(target=socket_reader)
	tha.start()
	thb = threading.Thread(target=socket_screen)
	thb.start()
	thc = threading.Thread(target=socket_writer)
	thc.start()

if __name__ == '__main__':
	main()