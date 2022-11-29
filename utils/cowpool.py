from utils.make_center import *

class CowPool:
    def __init__(self, NFrame, PFrame):
        self.NFrame = NFrame
        self.PFrame = PFrame
        self.num = 0
    def cow_id(self):
        self.num += 1
        self.cow_id = self.num

    def change_track_id(self, length=False):
        # boolean_sFrame = (np.array(NFrame['start_frame']) == np.array(PFrame['start_frame']))

        boolean_track_id = (np.array(self.NFrame['track_id']) == np.array(self.PFrame['track_id']))

        
        if length == True:
            for nframe in np.array(self.NFrame)[boolean_track_id]:
                lst = list()
                result = dict()
                count = 0
                dic = dict()
                for pframe in np.array(self.PFrame):
                    distance = find_distance(list(nframe[3:7]), list(pframe[3:7]))

                    dic[distance] = count
                    lst.append(distance)
                    count += 1
                lst.sort()
                result[nframe[2]] = np.array(self.PFrame['track_id'])[dic[lst[0]]]
                self.NFrame.iloc[nframe[2], 2] = result[nframe[2]]

        else:
            for nframe in np.array(self.NFrame)[boolean_track_id]:
                lst = list()
                result = dict()
                count = 0
                dic = dict()
                for pframe in np.array(self.PFrame):
                    dic[distance] = count
                    lst.append(distance)
                    count += 1
                lst.sort()
                result[nframe[2]] = np.array(self.PFrame['track_id'])[dic[lst[0]]]
                self.NFrame.iloc[nframe[2], 2] = result[nframe[2]]

    def add_distance(self):
        for pframe in np.array(self.PFrame):
            count = 0
            for nframe in np.array(self.NFrame):
                if pframe[2] == nframe[2]:
                    distance = find_distance(np.array(pframe[3:7]), np.array(nframe[3:7]))
                    self.NFrame.iloc[count, 8] = distance
                count += 1






    # def boolean(self, NFrame, PFrame):
    #     if len(PFrame) == len(NFrame):
    #         if (np.array(PFrame[['start_frame', 'track_id']]) != np.array(PFrame[['start_frame', 'track_id']])).all():
                    # change_track_id(NFrame, PFrame, length=True)
    #     else:
                # change_track_id(NFrame, PFrame, length=False)

