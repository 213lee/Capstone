졸업작품으로 제작한 신호위반 오토바이 검출 시스템.
customLib, customCV는 다른 인원이 작성한 코드로 포함X

Client.py 
라즈베리 파이에서 동작하는 client.
버튼 입력 시 서버에 연결하여 위치, 시간, 영상을 전송 후 서버에서 신호위반 결과를 받고 LED 출력.

Server.py
개인 pc에서 동작하는 server.
client에서 보낸 정보들을 받아 취합하여 저장하고 신호위반 판별 결과를 다시 client로 전송.

Detector.py
Server에서 수신받은 영상을 받아 객체를 검출하여 신호위반을 판별.

customLib.py
추출된 번호판에서 정보 추출을 위한 라이브러리(문자 투표시스템, 문자인식 등)

customCV.py
검출된 번호판 추출을 위한 OpenCV 라이브러리(임계처리, 허프변환, 외곽선 검출 등)
