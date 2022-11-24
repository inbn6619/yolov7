import os







### 디텍팅 자동화 실패 스크립트 (csv가 저장이 안됨)




# path = '/root/cow/yolov7/weights/'

# weights_paths = os.listdir(path)

# for i in range(len(weights_paths)):
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p5_tiny_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p5_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p5_x_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p6_d6_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p6_e6_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p6_e6e_ver01.pt')
os.system('python3 detect.py --weights /root/cow/yolov7/weights/yolov7_p6_ver01.pt')


print('test')