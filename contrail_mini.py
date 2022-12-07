import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_box_tracked
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from utils.Contrail import *







# cowpool
from utils.cowpool import CowPool
from utils.make_center import *

# Distance 거리 계산(피타고라스 정리 사선 루트 적용시켜서 제곱값을 없에줌)

import math

# DB 만들 때 필요한 라이브러리

import pandas as pd

# 보기불편한(?) 등의 이유로 변수 몰아놓은 스크립트

from utils.VariableGroup import *


# Points in Polygon (다각형 내외부 점 판별)

from shapely.geometry import Point, Polygon
from ByteTrack.yolox.tracker.Points_in_polygon  import mealarea_poly, waterarea_poly


# 미니맵 알고리즘

from utils.PixelMapper import pm1

# Queue 사용하여 Contrail 알고리즘 생성 목표

from collections import deque


# column time

from datetime import datetime

### 수정 및 생성한 파일

## /home/ubuntu/yolov7/utils/plots.py
## /home/ubuntu/yolov7/utils/PixelMapper.py
## /home/ubuntu/yolov7/ByteTrack/yolox/tracker/Points_in_polygon.py

### 생성 됬으나 불필요 하다고 느낀 파일

## /home/ubuntu/yolov7/utils/VariableGroup.py


# cowpool 
from CowManager.CowManager import CowManager

manager = CowManager(20)


def detect(save_img=False):

    ### 미니맵 동영상 저장 // cv2.VideoWriter(비디오저장 경로 및 이름, 비디오 코덱, 프레임, 비디오 크기(1920, 1080))

    fcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 15

    minimap_size = (1280, 720)

    resized_mini_size = (640, 480)
    
    out_minimap = cv2.VideoWriter('/content/yolov7/minimap.mp4', fcc, fps, resized_mini_size)



    # 변수 모음
    past_track_id_dict = dict()    

    for i in range(1, 9):
        past_track_id_dict[str(i)] = i
    
    nowdict = dict()

    pastdict = dict()

    result = list()

    ### 전체 데이터프레임
    test = pd.DataFrame()

    PFrame = pd.DataFrame()
    
    NFrame = pd.DataFrame()

    contrail_dict = {}

    

    # fps 동영상에서 찾아서 불러오기
    tracker = BYTETracker(opt, frame_rate=15)

    ### Default 값 불러오기 (opt = parser.parse_args() : 400번대 줄)
    
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    ### 저장 경로
    # Directories 디렉토리
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize 초기화
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    ### 동영상 저장 할때 사용하는 코드
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors // 이 컬러 사용안함
    ### 원래 용도 == 클래스 당 컬러 (사람 = 빨강, 차 = 노랑 등)
    names = model.module.names if hasattr(model, 'module') else model.names

    ### 예측 시작(트래커, 디텍션 돌아가는 곳)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()

        # 사실상 inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        ### 디텍션 한 값을 인덱스(i), 좌표와 스코어로 나누어 주는 코드
        ### enumerate() == 안의 값을 인덱스와 원소로 구성되는 튜플로 만들어줌
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            ### 미니맵 이미지 생성 코드

            canvas = cv2.imread('/home/ubuntu/minimap_png.png')


            ### 디텍션 했는지 파악해줌 // 아무것도 디텍팅 못했을 경우
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                ## Tracker
                tracked_targets = tracker.update(det[:, :5].cpu().numpy(), im0.shape)


                # 변수 모음
                NFrame = pd.DataFrame()

                cowidpool = set([1, 2, 3, 4, 5, 6, 7, 8])
                setlist = set()
                ### 프레임 전송 시간 코드
                realtime = datetime.now()


                for num in range(len(tracked_targets)):


                    # track_id 1~8값으로 고정
                    track_id = tracked_targets[num].track_id

                    checklist = sorted(list(cowidpool - setlist))

                    if str(track_id) not in past_track_id_dict.keys():
                        past_track_id_dict[str(track_id)] = checklist[0]
                    if past_track_id_dict[str(track_id)] in setlist:
                        try:
                            past_track_id_dict[str(track_id)] = checklist[0]
                        except:
                            print(len(tracked_targets))
                    setlist.add(past_track_id_dict[str(track_id)])
                    




                    # 필요 변수
                    xm, ym, xM, yM = tracked_targets[num].tlbr
                    
                    xc, yc = make_center(tracked_targets[num].tlbr)

                    nowdict[past_track_id_dict[str(track_id)]] = [xm, ym, xM, yM]




                    ### 데이터 컬럼 및 데이터 값으로 DataFrame 생성하여 DB 생성 코드

                    ### 식사량, 음수량은 Points in Polygon을 사용하여 해당 Dot이 Polygon(다각형)내에 있는지 판별 후
                    ### 있다면 1, 없다면 0의 값을 전송

                    ### 식사 판단 코드
                    ### meal_amount
                    dot = Point(xc, yc)
                    if dot.within(mealarea_poly):
                        meal_amount = 1
                    else:
                        meal_amount = 0

                    ### 음수 판단 코드
                    ### water_amount
                    if dot.within(waterarea_poly):
                        water_intake = 1
                    else:
                        water_intake = 0
                    
                    
            
                    # object pooling 생성 및 새로 추가 된 track_id 대조 하여 field에 옮기기
                    if len(PFrame) == 0:
                        manager.choiceCow(past_track_id_dict[str(track_id)], xc, yc)
                        travel_distance = 0
                    else:
                        if past_track_id_dict[str(track_id)] not in list(PFrame['cow_id']):
                            # if manager.comparePool(track_id, xc, yc):
                            manager.comparePool(past_track_id_dict[str(track_id)], xc, yc)
                            # else:
                            #     manager.choiceCow(track_id, xc, yc)
                            travel_distance = 0
                            
                        else:
                            manager.field_update(past_track_id_dict[str(track_id)], xc, yc)

                            travel_distance = find_distance(nowdict[past_track_id_dict[str(track_id)]], pastdict[past_track_id_dict[str(track_id)]])
                    
                    # object pooling Track_id로 field 인덱스 구하기
                    idx = manager.find_idx(past_track_id_dict[str(track_id)])

                    mxc, myc = pm1.pixel_to_lonlat((xc, yc))[0]

                    data = [
                        frame,
                        past_track_id_dict[str(track_id)], 
                        int(mxc),
                        int(myc),
                        int(travel_distance),
                        meal_amount,
                        water_intake,
                        ]

                    columns = [
                        'frame',
                        'cow_id',
                        'xc',
                        'yc',
                        'distance',
                        'meal',
                        'water',
                        ]


                    # track_id가 존재하는지 체크
                    if tracked_targets[num].track_id in contrail_dict.keys():
                        # 중심좌표 추가
                        contrail_dict[track_id].appendleft((int(xc), int(yc)))


                        # contrail 그리기
                        im0=tracking_tail(contrail_dict[track_id], im0, colors[past_track_id_dict[str(track_id)] % len(colors)], meal_amount, water_intake)


                        canvas = minimap_tail(contrail_dict[track_id], canvas, colors[past_track_id_dict[str(track_id)] % len(colors)], meal_amount, water_intake)
                    else:
                        contrail_dict[track_id] = deque(maxlen=45)
                        contrail_dict[track_id].appendleft((int(xc), int(yc)))

                    # 데이터 저장
                    df = pd.DataFrame([data], columns=columns)
                    # .set_index('origin_frame')
                    NFrame = pd.concat([NFrame, df])


                    # cv2 동영상 제작


                    plot_one_box_tracked(tracked_targets[num], xc, yc, meal_amount, water_intake,past_track_id_dict[str(track_id)], im0, canvas, colors[past_track_id_dict[str(track_id)] % len(colors)])
                


                # object pooling 없어진 Track_id
                if len(PFrame) != 0:
                    # newlist = list(set(NFrame['track_id']) - set(PFrame['track_id']))
                    dislist = list(set(PFrame['cow_id']) - set(NFrame['cow_id']))
                    if len(dislist) != 0:
                        manager.fieldToPool(dislist)





                PFrame = pd.DataFrame()

                PFrame = NFrame



                pastdict = dict()

                pastdict = nowdict
                
                nowdict = dict()


                test = pd.concat([test, NFrame])


                # if len(PFrame) != 0:
                #     pool = CowPool(NFrame, PFrame)

                #     if len(PFrame) == len(NFrame):
                #         if (np.array(PFrame['track_id']) == np.array(NFrame['track_id'])).all() == False:
                #             pool.change_track_id(length=True)
                #     else:
                #         pool.change_track_id()


                #     pool.add_distance()


                # if len(PFrame) != 0:
                #     if len(PFrame) == len(NFrame):
                #         boolean_track_id = (np.array(NFrame['track_id']) == np.array(PFrame['track_id']))
                #         if boolean_track_id.all():
                #             pass
                #         else:
                #             ids = list()
                #             lost_track_id = np.array(PFrame)[boolean_track_id]
                #             for id in lost_track_id:
                #                 ids.append(id[2])
                #             manager.fieldToPool(ids)
                #     else:
                #         
                #         if len(newlist) != 0:
                #             manager.fieldToPool(newlist)




                ### 바운딩박스, 미니맵, DB 생성 코드
                # for num in range(len(tracked_targets)):

                #     xm, ym, xM, yM = tracked_targets[num].tlbr
                #     xc = xm + (xM - xm) / 2
                #     yc = ym + (yM - ym) / 2

                    # ### Distance 생성 코드
                    # travel_distance = 0


                    # corrected_xc, corrected_yc = pm1.pixel_to_lonlat((xc, yc))[0]
                    
                    # nowdict[str(tracked_targets[num].track_id)] = [corrected_xc, corrected_yc]

                    # ### Distance 전 프레임과 현재 프레임의 Track_id가 같은 것을 찾아주는 코드
                    # if str(tracked_targets[num].track_id) in pastdict.keys():

                    #     ### 전프레임과 현프레임의 Center 값의 차이를 구해 Center의 이동거리(삼각형)을 구하고 제곱된 값을 Root를 씌워 Distance로 바꿔주는 코드
                    #     center = [x-y for x, y in zip(nowdict[str(tracked_targets[num].track_id)], pastdict[str(tracked_targets[num].track_id)])]

                    #     travel_distance = math.sqrt(center[0] ** 2 + center[1] ** 2)
                        
                    #     ### Distance 오차값 선정 코드
                    #     if travel_distance <= 4:
                    #         travel_distance = 0

                    ## 박스 및 미니맵 생성
                    
                    # plot_one_box_tracked(tracked_targets[num], xc, yc, past_track_id_dict[str(tracked_targets[num].track_id)], im0, canvas, colors[past_track_id_dict[str(tracked_targets[num].track_id)] % len(colors)])


                ### Distance 값을 위해 현프레임의 좌표를 과거 프레임 변수에 옮기고 현프레임은 초기화 하는 코드




            ### 각 프레임당 CV2로 추가된 미니맵 이미지를 저장해주는 코드
            # result.append(canvas)
            canvas = cv2.resize(canvas, resized_mini_size, interpolation=cv2.INTER_AREA)
            out_minimap.write(canvas)
            # print('test : ', len(result))



            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            # map_writer.release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resized_mini_size)
                        
                    im0 = cv2.resize(im0, resized_mini_size, interpolation=cv2.INTER_AREA)
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    ### 데이터 저장

    test.to_csv('data' + '.csv', index = True)

    
    ### release() == 선언된 변수에게 데이터 그만 보내라는 함수
    ### 사용 이유 : https://kali-live.tistory.com/8
    # out.release()
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/content/yolov7/yolov7_p6_e6e_ver01.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/content/yolov7/cowfarmB_ch3_2022072519_016.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()










                ## Overlay
                # for bbox in tracked_targets:
                #     plot_one_box_tracked(bbox, im0)



                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if save_txt:  # Write to file
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
                # print(weights)
                # print(save_name)
                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     # 박스 치는 곳
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)