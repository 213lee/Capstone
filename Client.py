import os, cv2, time, numpy, socket, datetime, base64, threading, sys
from datetime import datetime
import RPi.GPIO as GPIO
import requests, json
from gps import *

class ClientSocket:
    def __init__(self, ip, port, imagebuffer):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.ImageBuffer = imagebuffer
        self.connectCount = 0
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def sendImages(self):
        try:

            #1.보내는 시간(ex:221010-1930) 전송
            now = datetime.now()
            curr_time = now.strftime("%y%m%d-%H%M")
            encodebuf = curr_time.encode(('utf-8').ljust(64))
            self.sock.sendall(encodebuf)

            #2.현재 위치 전송
            gpsd = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)

            while True:
                report = gpsd.next()
                if report['class'] == 'TPV':
                    lat = getattr(report, 'lat', 0.0)
                    lon = getattr(report, 'lon', 0.0)
                    print(lat)
                    print(lon)
                    time.sleep(1)
                    break

            result = getLocation(lat, lon)
            address = result['response']['result'][0]['text']
            encodebuf = str(address).encode(('cp949').ljust(64))
            self.sock.sendall(encodebuf)
            print(address)


            #3.이미지 버퍼의 길이 전송
            bufferLen = len(self.ImageBuffer)
            encodebuf = str(bufferLen).encode(('utf-8').ljust(64))
            self.sock.sendall(encodebuf)

            print(curr_time)

            #4.버퍼에 있는 이미지(이미지 크기, 이미지 내용)를 순서대로 전송
            for i in range(bufferLen):
                img = self.ImageBuffer[i]
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, imgencode = cv2.imencode('.jpg', img, encode_param)
                data = numpy.array(imgencode)
                stringData = base64.b64encode(data)
                length = str(len(stringData))
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(stringData)
                print("Send Image[" + str(i) + "]")

            #서버에서 신호위반 판별 결과를 기다림.
            self.recvResult()
        except Exception as e:
            print(e)
            self.sock.close()

    def recvResult(self):
        try:
            violation = self.sock.recv(5).decode('utf-8')
            print(violation)

            #라즈베리파이에서 신호 결과에 따라 LED 출력
            printOutput(violation)

        except Exception as e:
            print(e)

        self.sock.close()
        print("socket.close()")


def makeSocket(ImageBuffer):
    client = ClientSocket('localhost', 9000, ImageBuffer)


#위치 받아오기
def getLocation(lat, lng):
    key = "04BBB25E-3C30-3CB8-83EA-FDFA98562C86"
    url = "http://api.vworld.kr/req/address?service=address&request=getAddress&crs=epsg:4326&point=" + str(
        lng) + "," + str(lat) + "&type=parcel&zipcode=true&simple=false&key=" + key
    result = requests.get(url)
    full_address = json.loads(result.text)
    return full_address


#LED 출력 함수
def printOutput(output):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(12, GPIO.OUT)  # GREEN Light
    GPIO.setup(16, GPIO.OUT)  # RED Light

    bool_output = (output == 'True')

    try:
        print("success")
        print("result is:" + output)
        if bool_output:
            GPIO.output(12, GPIO.HIGH)
            GPIO.output(16, GPIO.LOW)
            time.sleep(10.0)
        else:
            GPIO.output(16, GPIO.HIGH)
            GPIO.output(12, GPIO.LOW)
            time.sleep(10.0)


    except KeyboardInterrupt:
        print("error")
        pass

    GPIO.cleanup()


exit = False
ImageBuffer = []

#라즈베리파이에서 입력 콜백함수
def button_callback(channel):
    global exit
    print("Button pushed!")
    makeSocket(ImageBuffer.copy())
    exit = True

#라즈베리파이의 캠을 사용하는 버전
def Cam():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    GPIO.add_event_detect(21, GPIO.RISING, callback=button_callback)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 315)

    fps = 10                       # frame
    recordSec = 8                  # 저장할 시간(초)
    recordFrame = fps * recordSec  # 버퍼에 담을 프레임 개수

    #ImageBuffer = []    # 프레임을 저장하기 위한 버퍼
    prevTime = 0
    while not exit:
        currTime = time.time() - prevTime
        if currTime > 1. / fps:
            ret, frame = capture.read()
            cv2.imshow("frame", frame)

            if recordFrame <= len(ImageBuffer) : ImageBuffer.pop(0)
            ImageBuffer.append(frame)
            prevTime = time.time()
            key = cv2.waitKey(1)
            if key == 27:       #ESC 종료
                break
            elif key == 13:     #Input(Enter) 이미지 전송을 위해 스레드 시작
                makeSocket(ImageBuffer.copy())
                break

    capture.release()
    GPIO.cleanup()

#로컬 비디오 버전
def LocalVideo():
    _inputPath = "localVideo/NIGHT_03.MOV"
    #_inputPath = "./sample/week5/day/output_3.mp4"
    capture = cv2.VideoCapture(_inputPath)

    print("Frame Widht ", capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Frame Height ", capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10  # frame
    recordFrame = 100  # 버퍼에 담을 프레임 개수

    framecounter = 0

    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", 1280, 960)
    while True:
            ret, frame = capture.read()
            if not ret : break
            cv2.imshow("frame", frame)
            framecounter += 1
            if recordFrame <= len(ImageBuffer): ImageBuffer.pop(0)
            ImageBuffer.append(frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 종료
                break
            elif key == 13:  # Input(Enter) 이미지 전송을 위해 스레드 시작
                client = ClientSocket('localhost', 9000, ImageBuffer.copy())
                break

    capture.release()


def main():
    # 로컬비디오 or 라즈베리파이 선택
     Cam()
    # LocalVideo()

if __name__ == "__main__":
    main()