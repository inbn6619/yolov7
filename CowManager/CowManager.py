from math import sqrt
class Cow():
    Cownum = 0
    def __init__(self):
        Cow.Cownum += 1
        self.cow_id = Cow.Cownum
        self.track_id = None
        self.x = None
        self.y = None

    def setTrack_id(self, num):
        self.track_id = num
        
    def setRocation(self, x, y):
        self.x = x
        self.y = y
        


# 싱글턴 필요
# RTSP _ live 시 싱글턴 thread 관리 필요
class CowManager():
    # 생성
    def __init__(self, maxcownum):
        # 큐로 구현시 더 편함
        self.field = []
        self.pool = []
        self.cowcount = maxcownum
        self.__createCow()

    def __createCow(self):
        for _ in range(self.cowcount):
            self.pool.append(Cow())

    # 디텍션 시작 하고 track_id가 부여되면 호출함 
    def choiceCow(self, track_id, x, y):
        cow = self.pool.pop()
        cow.setTrack_id(track_id)
        cow.setRocation(x, y)
        self.field.append(cow)
        print("남은 POOL : ", len(self.pool))
        

    # 필드에서 사라졌을 경우 pool에 넣을 것
    def fieldToPool(self, lost_track_id):
        for disnum in lost_track_id:
            for cow in self.field:
                if disnum == cow.track_id:
                    discow = self.field.pop(self.field.index(cow))
                    self.pool.append(discow)
                    print("<<<<<<<<<<ADD POOL>>>>>>>>>>>")
                    break
        


    # 풀에서 비교할것
    def comparePool(self, track_id, x, y):
        check = True
        # 잠깐 비교를 위해 담아둠
        answer = []
        # 한바퀴 돌면서 찾음
        for cow in self.pool:
            if cow.x == None or cow.y == None or cow.track_id == None:
                continue
            else:
                answer.append((cow.x - x)**2 + (cow.y -y)**2)
        # 가장 작은 값이 근처 소라고 생각
        # 아닐수도 있음 
        if len(answer) != 0:
            target = min(answer)
            targetindex = answer.index(target)
            findcow = self.pool.pop(targetindex)
            findcow.setRocation(x, y)
            findcow.setTrack_id(track_id)
            self.field.append(findcow)
            print("남은 POOL : ", len(self.pool))
        else:
            check = False

        return check

    
    def find_idx(self, track_id):
        
        for cow in self.field:
            if cow.track_id == track_id:
                global idx 
                idx = self.field.index(cow)
        return idx
        

    def field_update(self, track_id, xc, yc):
        cow = self.find_idx(track_id)
        self.field[cow].setRocation(xc, yc)
        


        


        # 대충 짠거임 <- !!! 
        # 내 기억에 python list를 self.list 형식으로 쓰면 오류났던거 같음

        # 뭔가 겹치는게 많은거 같은데 알아서 잘 짜보도록 ...


    # 필요없는거 같은데 ?
    # # 풀에서 꺼내서 다시 필드로 보냄
    # def poolToField(self, cow_id, field_id):
    #     for cow in self.pool:
    #         if cow.cow_id == cow_id:
    #             cow.setTrack_id(field_id)
    #             findcow = self.pool.pop(self.pool.index(cow))
    #             self.field.append(findcow)
    #             break









# test = CowManager(10)


# print(test)


# test.choiceCow(1, 2, 3)


# test.choiceCow(1, 5, 8)




# print('test')
# print('test')
# print('test')
# print('test')
# print('test')
# print('test')
# print('test')
# print('test')
# print('test')
# print('test')