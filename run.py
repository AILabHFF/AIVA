import os
from main import run_video

folderpath = '/media/disk1/KILabDaten/Geminiden 2021/Kamera2/CutVideos/'

if os.path.isdir(folderpath):
    print('jes')

i = 0
for file in os.listdir(folderpath):
    i+=1
    print(folderpath+file)
    run_video(file_path=folderpath+file, show=False, capture_pngs=False)
print('done')