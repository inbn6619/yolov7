import os

weights_paths = os.listdir('/root/cow/yolov7/weights')

sources_paths = os.listdir('/root/cow/yolov7/videos')

def Changedetectdefault():
    for weight in weights_paths:
        print(weight, end='\n')
    for source in sources_paths:
        print(source, end='\n')
    weight = int(input("""select weight : """)) - 1
    print("""your select : %s""" % weights_paths[weight])
    source = int(input("""select source : """)) - 1
    print("""your select : %s""" % sources_paths[source])
    return weights_paths[weight], sources_paths[source]

print(weights_paths)

print('test is vaild ', len(weights_paths))