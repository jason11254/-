import threading
import socket
import sys
import _thread
import RPi.GPIO as GPIO
import time
import os
import sqlite3
import grovepi

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(10, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
c = GPIO.PWM(10, 50)
l = GPIO.PWM(16, 50)
r = GPIO.PWM(18, 50)
ultrasonic_ranger = 4
ultrasonic_rangers = 3
target_host = '192.168.0.109'
target_port = 8888
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((target_host, target_port))

def setio(p7, p11, p13, p15):
	GPIO.output(7, p7)
	GPIO.output(11, p11)
	GPIO.output(13, p13)
	GPIO.output(15, p15)

def ultrasonic():
	while True:
		try:
			a=str(grovepi.ultrasonicRead(ultrasonic_ranger))
			a2 = str(grovepi.ultrasonicRead(ultrasonic_rangers))
			if a2 > a :
				bytes(a, encoding = "utf8")
				client.sendall(str.encode(a))
			elif a > a2 :
				bytes(a2, encoding = "utf8")
				client.sendall(str.encode(a))
		except TypeError:
			print ("tError")
		except IOError:
			print ("ioError")

def socket_reader():
	var = 1
	while var == 1:
		response = client.recv(4096)
		a = str(response)
		if a.find('go') != -1 :
			print("go")
			setio(False, True, False, True)
		elif a.find('stop') != -1 :
			print("stop")
			setio(False, False, False, False)
			time.sleep(1)
			print("open")
			c.start(8.5)
			time.sleep(1)
			print("down")
			l.start(7.5)
			r.start(6.5)
			time.sleep(1)
			print("close")
			c.start(5.0)
			time.sleep(1)
			print("up")
			l.start(12.5)
			r.start(2.0)
			time.sleep(1)
		elif a.find('doput') != -1:
			print("stop")
			setio(True, False, True, False)
			print("down")
			l.start(7.5)
			r.start(6.5)
			time.sleep(1)
			print("open")
			c.start(8.5)
			time.sleep(1)
			print("up")
			l.start(12.5)
			r.start(2.0)
			time.sleep(1)
			print("close")
			c.start(5.0)
			time.sleep(1)
		elif a.find('right') != -1:
			print("r")
			setio(False, True, False, False)
		elif a.find('left') != -1:
			print("l")
			setio(False, False, False, True)
		elif a.find('end') != -1:
			print("e")
			setio(True, False, True, False)

def main():
	tha = threading.Thread(target=ultrasonic)
	tha.start()
	thb = threading.Thread(target=socket_reader)
	thb.start()

if __name__ == '__main__':
	main()


