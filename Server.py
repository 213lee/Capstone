import os, cv2, time, numpy, socket, threading, datetime, base64, traceback
from Detector import Detector

class ServerSocket:
    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.ImageBuffer = []
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        try:
            startTime = time.time()

            #1.클라이언트에서 보낸 시간 받기
            # 파일 저장경로 result/시간 으로 폴더 만들어 놓기
            violationTime = self.conn.recv(11).decode('utf-8')
            print(violationTime)
            outputPath = "result/" + violationTime
            os.makedirs(outputPath)

            #2.클라이언트에서 보낸 위치 받기
            violationLocation = self.conn.recv(64).decode('cp949')
            print(violationLocation)

            #3.이미지 버퍼의 길이 받기
            ImageBuffer = []
            buflen = self.conn.recv(10)
            bufferLen = buflen.decode('utf-8')
            print(bufferLen)

            #4.순서대로 이미지를 받아 버퍼에 저장
            for i in range(int(bufferLen)):
                length = self.recvall(self.conn, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.conn, int(length1))
                print('receive time: ' + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)
                ImageBuffer.append(decimg)

            #5.저장된 이미지들(ImageBuffer)을 가지고 신호위반 여부 검사후 true or false를 리턴(violation)받음
            detector = Detector(ImageBuffer, outputPath)
            violation_bool = detector.StartDetection()
            print("Violation Result : ", violation_bool)
            textViolation = ""
            if violation_bool: textViolation = "TRUE"
            else: textViolation = "FALSE"


            #6 번호판 검출
            plateNumber = detector.PlateToString()

            #7. 받은 시간, 위치, 신호위반여부(True or False), 차량번호를 메모장에 취합해서 저장
            #경로는 현재프로젝트위치/result/받은시간/RESULT.txt
            resultFile = open(outputPath + "/RESULT.txt", "w")
            resultFile.write("신고 시간 : " + violationTime + "\n")
            resultFile.write("신고 위치 : " + violationLocation + "\n")
            resultFile.write("신호위반 여부 : " + textViolation + "\n")
            resultFile.write("차량 번호 : " + plateNumber + "\n")
            resultFile.close()

            #8. 신호위반 여부를 마지막으로 클라이언트에 전송
            encodebuf = textViolation.encode(('utf-8').ljust(64))
            self.conn.sendall(encodebuf)

            endTime = time.time()
            print(f"{endTime - startTime : .5f} sec")
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.socketClose()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def main():
    server = ServerSocket('', 9000)

if __name__ == "__main__":
    main()