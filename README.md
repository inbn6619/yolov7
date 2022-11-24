// EC2 env

	sudo apt update -y
	
	sudo apt install python3-pip

	git clone https://github.com/inbn6619/yolov7.git

	cd yolov7

	git clone https://github.com/inbn6619/ByteTrack.git

	cd ByteTrack
<<<<<<< HEAD
<<<<<<< HEAD
=======
	
	sudo apt install python3-pip -y
>>>>>>> 02f7433d0a3bb26c823e0c1f93a93bec909d6927
=======
	
	sudo apt install python3-pip -y
>>>>>>> 02f7433d0a3bb26c823e0c1f93a93bec909d6927

	sudo pip3 install -r requirements.txt

	sudo python3 setup.py develop

	sudo pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

	sudo pip3 install cython_bbox

	sudo pip3 install seaborn

	sudo apt-get update
	sudo apt-get -y install libgl1-mesa-glx

// 추가 라이브러리

	sudo pip3 install shapely














// cv2 Error Check

	version check
		python3
			import cv2
			불러오기 안된다면 인식, 설치 안된거임
			print(cv2.__version__) // 버전확인

// cv2 Error Fix


	sudo apt-get update
	sudo apt-get -y install libgl1-mesa-glx
	or
	sudo pip3 install opencv-python-headless
	
	sudo pip3 install -r requirements.txt



// torch Error Check

	version check
		python3
			import torch
			print(torch.__version__)
	cuda check(gpu on or off)
		python3
			import torch
			torch.cuda.is_available()
				True: GPU ON
				False: GPU OFF
