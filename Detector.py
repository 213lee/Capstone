import time
import cv2
from enum import Enum, auto
from datetime import datetime
from math import dist
import numpy as np, cv2, os
import customLib as cl

#현재 신호 상태 Enum
class Sign_Status(Enum):
    NONE = auto()           #신호가 식별되지 않는 상태
    RED = auto()            #빨간불
    GREEN = auto()          #초록불
    RED_GREEN = auto()      #빨간불과 초록불 모두 식별된 상태(ex: 좌회전)

class ObjectNum(Enum):
    BIKE = 3
    TRAFFIC_LIGHT = 9

class Detector:
    ObjectNet = cv2.dnn.readNet("yolofiles/Object.weights", "yolofiles/Object.cfg")                     # 오토바이, 신호등 검출
    ObjectOutput = None
    PlateNet = cv2.dnn.readNet("yolofiles/Plate.weights", "yolofiles/Plate.cfg")      # 번호판 검출용
    PlateOutput = None

    def __init__(self, frameset, path):
        self.OutputPath = path                      #프레임 결과를 저장할 경로
        self.FrameSet = frameset                    #서버에서 받을 프레임 셋
        self.BikeDetect = False                     #현재 번호판이 검출된 오토바이를 찾았는지의 여부
        self.CurrentSign = Sign_Status.NONE         #현재 신호 상태
        self.StartCoord = np.zeros(2).astype(int)   #검출할 오토바이의 시작(x, y) 좌표
        self.MaxDiff = 0                            #추적할 오토바이의 시작좌표와의 차이값
        self.PlateWriteIndex = 0                    #번호판이 저장될 이름의 인덱스
        self.IsViolation = False                    #신호위반 했는지 여부
        self.tracker = None                         #트레킹에 사용될 객체
        self.SignalQ = []                           #신호가 연속으로 잡히고 있는지 확인하는 큐
        self.ResetCoord = False                     #오토바이의 시작 좌표를 재설정해야 하는지 여부
        self.thresholdDist = 0                      #움직임 임계값

        self.Initialize()

    #초기화
    def Initialize(self):
        if Detector.ObjectNet is None or Detector.PlateOutput is None:
            layer_names = Detector.ObjectNet.getLayerNames()
            Detector.ObjectOutput = [layer_names[i - 1] for i in Detector.ObjectNet.getUnconnectedOutLayers()]
            layer_names = Detector.PlateNet.getLayerNames()
            Detector.PlateOutput = [layer_names[i - 1] for i in Detector.PlateNet.getUnconnectedOutLayers()]

    #현재 신호 식별자
    def Traffic_Light_Identifier(self, traffic_light_img):
        #신호등 영역을 64 x 64로 resize 한 후에
        #색으로 현재 신호 판별하기 위해 HSV 색 영역으로 변환
        try:
            traffic_light_img = cv2.resize(traffic_light_img, (64, 64))
            hsv = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2HSV)
        except Exception as e: print(str(e))

        # 빨간색 영역의 값을 추출
        lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
        upper_red = cv2.inRange(hsv, (170, 90, 100), (180, 255, 255))
        red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)

        #초록색 영역의 값을 추출
        green = cv2.inRange(hsv, (80, 100, 100), (90, 255, 255))

        red_cnt = np.count_nonzero(red)
        green_cnt = np.count_nonzero(green)

        thisSign = None
        # 빨간불
        if red_cnt > 50 and green_cnt < 10:
            thisSign = Sign_Status.RED
        # 초록불
        elif red_cnt < 10 and green_cnt > 50:
            thisSign = Sign_Status.GREEN
        # 둘다 ON
        elif red_cnt > 50 and green_cnt > 30:
            thisSign = Sign_Status.RED_GREEN
        else:
            thisSign = Sign_Status.NONE

        if thisSign is not None: self.SignalQ.append(thisSign)

        qSize = len(self.SignalQ)
        if qSize > 2:
            if self.SignalQ[qSize - 2] == self.SignalQ[qSize - 1] == thisSign:
                if self.CurrentSign != Sign_Status.RED and self.CurrentSign != Sign_Status.NONE and thisSign == Sign_Status.RED:
                    self.ResetCoord = True
        elif thisSign is not None:
            self.CurrentSign = thisSign

    #현재 신호에 맞게 신호등 그리기
    def DrawTrafficLight(self, img):
        # 빨 노 좌 초
        cv2.rectangle(img, (50, 50), (450, 150), (0, 0, 0), -1)
        cv2.circle(img, (100, 100), 40, (0, 0, 255))
        cv2.circle(img, (200, 100), 40, (0, 212, 255))
        cv2.circle(img, (300, 100), 40, (0, 255, 0))
        cv2.circle(img, (400, 100), 40, (0, 255, 0))

        if self.CurrentSign == Sign_Status.RED:                     #빨간불
            cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
        elif self.CurrentSign == Sign_Status.GREEN:                 #초록불
            cv2.circle(img, (400, 100), 40, (0, 255, 0), -1)
        elif self.CurrentSign == Sign_Status.RED_GREEN:             #둘다 ON
            cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
            cv2.line(img, (330, 100), (270, 100), (0, 255, 0), 5, cv2.LINE_AA)
            cv2.line(img, (300, 70), (270, 100), (0, 255, 0), 5, cv2.LINE_AA)
            cv2.line(img, (300, 130), (270, 100), (0, 255, 0), 5, cv2.LINE_AA)

    #트레킹 시작 전 Initiailize
    def Tracker_Init(self, img, init_roi):
        trackers = ["cv2.legacy.TrackerBoosting_create",    #0. not bad, 오락가락함
                    "cv2.legacy.TrackerMIL_create",         #1. 일정 이상 추적이 안됨
                    "cv2.legacy.TrackerKCF_create",         #2. Best
                    "cv2.legacy.TrackerTLD_create",         #3. 제일 이상하게 잡힘
                    "cv2.legacy.TrackerMedianFlow_create",  #4. 추적 경계상자 거의 안움직임
                    #cv2.legacy.TrackerGOTURN_create,       #5. 버그로 오류 발생
                    "cv2.legacy.TrackerCSRT_create",        #6. 따라가다가 멈춰서 처음자리로 돌아옴 == MIL 1번
                    "cv2.legacy.TrackerMOSSE_create"]       #7. KCF > MOSSE > ...

        trackerIdx = 2
        self.tracker = eval(trackers[trackerIdx])()
        self.tracker.init(img, init_roi)

    #트레킹
    def Tracking(self, img):
        if self.tracker is None:
            print("Tracker is None")
        else:
            ok, bbox = self.tracker.update(img)
            (x, y, w, h) = bbox
            x = int(x);  y = int(y); w = int(w); h = int(h)
            if ok:  #추적 성공
                self.Detection_Plate(img[y:y+h, x:x+w].copy())
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), \
                              (0, 255, 0), 2, 1)

                if self.ResetCoord:
                    self.ResetCoord = False
                    self.StartCoord[0] = x; self.StartCoord[1] = y
                    self.MaxDiff = 0

                #추적 시작 좌표와 거리 계산
                diff = dist(self.StartCoord, (x, y))
                self.MaxDiff = max(diff, self.MaxDiff)
            else: #추적 실패
                print("Tracking fail")
                self.BikeDetect = False



    #일반 객체(오토바이, 신호등) 검출
    def Detection_Object(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        Detector.ObjectNet.setInput(blob)
        outs = Detector.ObjectNet.forward(Detector.ObjectOutput)

        #검출된 정보 저장 리스트
        class_ids = []
        confidences = []
        boxes = []

        #해당 프레임에서 객체를 모두 검출하는 반복문
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # 3 = 오토바이 9 신호등
                # 확률 0.5이상, 신호등, 번호판이 검출된 오토바이를 발견하지 못했을 때 오토바이
                if not (confidence > 0.5 and ((self.BikeDetect == False and class_id == ObjectNum.BIKE.value) or class_id == ObjectNum.TRAFFIC_LIGHT.value)): continue

                #중심좌표 detection[0-4] = [center_x, center_y, width, height]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #시작 x,y 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        #검출된 신호등 중 가장 위에(y) 있는 신호등 인덱스 찾기
        minY = 1.0
        traffic_lgiht_idx = -1
        for i in range(len(boxes)):
            if class_ids[i] == ObjectNum.TRAFFIC_LIGHT.value and boxes[i][1] > minY:
                minY = boxes[i][1]
                traffic_lgiht_idx = i

        # 위에서 만족하는 신호등이 있다면
        # 해당 신호등의 영역으로 현재 신호를 판별 Call : Traffic_Light_Identifier()
        if traffic_lgiht_idx != -1:
            x, y, w, h = boxes[traffic_lgiht_idx]
            dst = img[y:y + h, x:x + w].copy()
            self.Traffic_Light_Identifier(dst)

        # 번호판이 검출된 오토바이가 아직 없다면 해당 오토바이를 찾음
        if not self.BikeDetect:
            for i in range(len(boxes)):
                if class_ids[i] == ObjectNum.BIKE.value and i in indexes:
                    x, y, w, h = boxes[i]
                    bike = img[y:y + h, x:x+w].copy()
                    if self.Detection_Plate(bike):
                        self.BikeDetect = True
                        self.StartCoord[0] = x
                        self.StartCoord[1] = y
                        self.Tracker_Init(img.copy(), boxes[i])
                        break
        # 있다면 트레킹
        else:
            self.Tracking(img)


    # 번호판(오토바이 영역을 받아서) 검출
    def Detection_Plate(self, bike):
        try:
            height, width, channels = bike.shape
            blobBike = cv2.dnn.blobFromImage(bike, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            Detector.PlateNet.setInput(blobBike)
            outs = Detector.PlateNet.forward(Detector.PlateOutput)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


            if len(confidences) <= 0 : return False
            maxIdx = confidences.index(max(confidences))

            x, y, w, h = boxes[maxIdx]
            tmp = bike[y:y + h, x:x + w].copy()
            cv2.imwrite(self.OutputPath + "/plate/" + str(self.PlateWriteIndex) + ".jpg", tmp)
            self.PlateWriteIndex += 1
        except Exception as e:
            print(str(e))

        return True

    #현재 신호위반이라면 신호위반 문구를 출력하고 아니라면 신호위반 조건을 검사
    def Violation(self, img):
        width = img.shape[1]; height = img.shape[0]
        if self.IsViolation:
            #영상에 Violation 그리기
            #cv2.putText(img, "Violation!", (int(width * 0.1), int(height * 0.65)), \
            #            cv2.FONT_HERSHEY_TRIPLEX, 8, (0, 0, 255), 8, cv2.LINE_AA)
            print(self.IsViolation)
        elif self.MaxDiff > self.thresholdDist and self.CurrentSign == Sign_Status.RED:
            self.IsViolation = True

    def StartDetection(self):
        os.makedirs(self.OutputPath + "/images")
        os.makedirs(self.OutputPath + "/plate")

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 10
        height = self.FrameSet[0].shape[0]
        width = self.FrameSet[0].shape[1]
        writer = cv2.VideoWriter(self.OutputPath + "/images/video.mp4"
                                 , fourcc,  10, (width, height))

        #영상에서 움직임 임계값 설정 영상의 대각선 길이 5%
        self.thresholdDist = dist((0, 0), (width, height)) * 0.05

        for i in range(len(self.FrameSet)):
            img = self.FrameSet[i]
            self.Detection_Object(img)
            self.DrawTrafficLight(img)
            self.Violation(img)
            cv2.imwrite(self.OutputPath + "/images/" + str(i) + ".jpg", img)
            print("Write " + str(i) + ".JPG")
            writer.write(img)
            print(self.SignalQ[-1], len(self.SignalQ))

        return self.IsViolation

    #번호판 텍스트로 추출
    def PlateToString(self):
        string_list = []
        plateLength = len(os.listdir(self.OutputPath + "/plate"))
        if plateLength <= 0: return "Extraction failed"

        for i in range(plateLength): string_list.append(cl.Plate.PlateExtraction(self.OutputPath + "/plate/" + str(i) + ".jpg"))
        print(string_list)
        letter = cl.LetterVoting.VoteSystem(string_list, 13)
        letter = " ".join(letter.split())
        print(letter)

        return letter

#로컬 비디오로 Detector 돌리기
def main():
    frameset = []
    now = datetime.now()
    curr_time = now.strftime("%y%m%d-%H%M")
    path = "result/" + curr_time

    #Input Video 경로 주의
    _inputPath = "Data/green/day/DAY_03.MOV"
    capture = cv2.VideoCapture(_inputPath)

    #n 배수 프레임만 추출.
    frameCounter = 0
    while True:
        frameCounter += 1
        ret, frame = capture.read()
        if not ret: break
        if frameCounter % 2 == 0: frameset.append(frame)
        print("READ " + str(frameCounter) + " FRAME")

    det = Detector(frameset, path)
    textViolation = None

    if det.StartDetection(): textViolation = "TRUE"
    else: textViolation = "FALSE"

    textPlate = det.PlateToString()

    # 6.받은 시간, 위치, 신호위반여부(True or False), 차량번호를 메모장에 취합해서 저장
    # 경로는 현재프로젝트위치/result/받은시간/RESULT.txt
    resultFile = open(path + "/RESULT.txt", "w")
    resultFile.write("신고 시간 : " + curr_time + "\n")
    resultFile.write("신고 위치 : " + "\n")
    resultFile.write("신호위반 여부 : " + textViolation + "\n")
    resultFile.write("차량 번호 : " + textPlate + "\n")

if __name__ == "__main__":
    main()

