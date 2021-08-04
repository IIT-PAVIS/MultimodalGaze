import os, cv2

root = '/home/sanketthakur/Downloads/Pavis_Social_Interaction_Attention_dataset/'

total_mins = 0.0
total_sec = 0.0
samples = 0
for index, subDir in enumerate(sorted(os.listdir(root))):
    samples += 1
    os.chdir(root + subDir)
    file = 'scenevideo.mp4'
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (frame_count/fps) + 1
    mins = int(duration / 60)
    sec = int(duration%60)
    total_mins += mins
    total_sec += sec
    if total_sec > 60 :
        total_mins += int(total_sec / 60)
        total_sec = total_sec % 60

    print(subDir + ' : ', str(mins) + ' mins', str(sec) + ' secs')
    # print(total_mins, total_sec)
