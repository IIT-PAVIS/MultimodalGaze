import os

for index, subDir in enumerate(sorted(os.listdir('/data/sanket/Pavis_Social_Interaction_Attention_dataset/'))):
    #if 'train_BookShelf' in subDir:
    #    continue
    if 'train_' in subDir:
        _ = os.system('mkdir /data/sanket/Pavis_Social_Interaction_Attention_dataset/datasets/' + subDir[6:])
        filename = 'vision_checkpointAdam_' + subDir[6:] + '.pth'
        _ = os.system('scp sanketthakur@10.245.82.31:/data/gaze_data_1/datasets/CoffeeVendingMachine_S1/' + filename + '.pth /data/sanket/Pavis_Social_Interaction_Attention_dataset/datasets/' + subDir[6:])
