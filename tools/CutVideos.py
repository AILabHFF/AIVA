import os
import pandas as pd
import cv2

# Extraction of frames

# Set csv path
csv1_path = '../Geminiden/kam1.csv'
csv2_path = '../Geminiden/kam2.csv'


path_c1 = "/media/disk1/KILabDaten/Geminiden2021/Kamera1/"
path_c2 = "/media/disk1/KILabDaten/Geminiden2021/Kamera2/"
fileend = '.mov'

outpath_c1 = path_c1 + "MeteorFrames/"
outpath_c2 = path_c2 + "MeteorFrames/"

camnum = 2

camdict = {1:[csv1_path, path_c1, outpath_c1], 2:[csv2_path, path_c2, outpath_c2]}

file_path = camdict[camnum][1] #path_c1
csv_path = camdict[camnum][0] #csv1_path
outpath = camdict[camnum][2] #outpath_c1
if not os.path.isdir(outpath):
  os.makedirs(outpath)

df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
df['CTL'] = pd.to_datetime(df['CTL'].str.strip(), format='%H:%M:%S')#.dt.time

df['CTLend'] = df['CTL'] + pd.Timedelta(seconds=4)
df.CTL = df.CTL#.dt.time
df.CTLend = df.CTLend#.dt.time

df['CTL'] = df['CTL'].dt.hour + df['CTL'].dt.minute*60 + df['CTL'].dt.second
df['CTLend'] = df['CTLend'].dt.hour + df['CTLend'].dt.minute*60 + df['CTLend'].dt.second

print(df.head())

col_data_name = df['Dateiname'].tolist()
col_start = df['CTL'].tolist()
col_end = df['CTLend'].tolist()
col_num = df['Nr.']

#return video length in seconds
def video_length(filename):
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count/fps

    video.release()
    return int(duration), frame_count, fps

name_time_id_dict = {}

for name, start, id in zip(col_data_name, col_start, col_num):
    if name not in name_time_id_dict:
        name_time_id_dict[name] = []
    
    
    name_time_id_dict[name].append([id, start])

n_dict = {}

for file in name_time_id_dict:
    fdir = file_path + file + '.MOV'
    l, framecount, _ = video_length(fdir)
    n_dict[file] = []
    for x in name_time_id_dict[file]:
        i, s = x
        nstart = s*25-5*25
        nend = s*25+5*25
        if nstart<1:
            nstart = 1
        if nend > int(framecount):
            nend = int(framecount)
        n_dict[file].append([i, nstart, nend])

frame_to_id_dict= {}


for file in n_dict:
    fdir = file_path + file + '.MOV'
    l, framecount, _ = video_length(fdir)

    frame_to_id_dict[file] = {}

    ids = [i[0] for i in n_dict[file]]
    starttimes = [i[1] for i in n_dict[file]]
    endtimes = [i[2] for i in n_dict[file]]

    for id, s, e in zip(ids,starttimes, endtimes):
        for j in range(s,e+1):
            if j not in frame_to_id_dict[file]:
                frame_to_id_dict[file][j]=[]
            if id not in frame_to_id_dict[file][j]:
                frame_to_id_dict[file][j].append(id)

ignore_list = ['NINJA2_S001_S001_T007', 'NINJA2_S001_S001_T009',  #eventuell nicht vollständig
                'NINJA2_S001_S001_T016', 'NINJA2_S001_S001_T022', 'NINJA2_S001_S001_T023']

ignore_list = []


for file in frame_to_id_dict:
    fdir = file_path + file + '.MOV'
    l, framecount, _ = video_length(fdir)


    cap = cv2.VideoCapture(fdir)
    print(file)
    if file in ignore_list:
        cap.release()
        print('skipping:', file)
        continue


    while cap.isOpened():
        ret, frame = cap.read()
        frameid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))


        p_out = str(frameid) + ' / ' + str(framecount)
        print (p_out, end="\r")

        if ret:
            if frameid in frame_to_id_dict[file].keys():

                for obj_id in frame_to_id_dict[file][frameid]:
                    out_object_path = outpath + str(obj_id) + '/'
                    if not os.path.isdir(out_object_path):
                        os.makedirs(out_object_path)
                    out_img_path = out_object_path + str(frameid) + '.png'
                    if not os.path.isfile(out_img_path):
                        img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
                        cv2.imwrite(out_img_path, img_rgb)
        
        else:
            cap.release()
            break
cap.release()