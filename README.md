// EC2 env

	sudo apt update -y

	git clone https://github.com/inbn6619/yolov7.git

	cd yolov7

	git clone https://github.com/inbn6619/ByteTrack.git

	cd ByteTrack

	sudo pip3 install -r requirements.txt

	sudo python3 setup.py develop

	sudo pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

	sudo pip3 install cython_bbox

	sudo pip3 install seaborn

	sudo apt-get update
	sudo apt-get -y install libgl1-mesa-glx
















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
