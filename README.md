# MultimodalGaze
***Code of ICMI 2021 paper: "Predicting Gaze from Egocentric Social Interaction Videos and IMU Data"***

You need to download additional modules to run the repo. 

**Flownet**
```
git clone https://github.com/ClementPinard/FlowNetPytorch.git
```
**I3d network:**
```
https://github.com/piergiaj/pytorch-i3d.git 
```

To run the model : 
```
python3 main.py --model <model_type> 
```
(model types : 0 - `signal`, 1 - `vision`, 2 - `multimodal`)

To specify which multimodal & vision based model you need, you can change it in main.py file on line no. [29](https://github.com/IIT-PAVIS/MultimodalGaze/blob/762206f704a4fac03dcb9567f0bc75e6f7a43575/main.py#L29) for vision model (optical flow / i3d), and line no. [31](https://github.com/IIT-PAVIS/MultimodalGaze/blob/762206f704a4fac03dcb9567f0bc75e6f7a43575/main.py#L31)

Select model(s) from [models.py](https://github.com/IIT-PAVIS/MultimodalGaze/blob/main/models.py)

**i3d models:**

Vision only : `VISION_PIPELINE`, Multimodal: `FUSION_PIPELINE`

**flownet models:**

Vision only : `Flownet_PIPELINE`, Multimodal: `OF_FUSION_PIPELINE`

**Data**
The models were trained on a new dataset collected by the authors for Social Interaction based scenarios in the wild. It will be made public in future. 

All the checkpoints will be made available when the dataset will be made public. 

