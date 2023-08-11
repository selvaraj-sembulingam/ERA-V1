# YOLO-V3

YOLO (You Only Look Once) is a state-of-the-art real-time object detection system. YOLOv3 is an improved version of the YOLO algorithm that is faster and more accurate. In this project, we train YOLOv3 on the Pascal VOC dataset, which is a widely used benchmark dataset for object detection.

## Architecture
![YOLOv3-architecture](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/043a5fff-9619-4863-8a23-67c5a87a94fe)

## Folder Structure
```
└── README.md
└── config.py
└── dataset.py
└── dataset_org.py
└── loss.py
└── main.ipynb
└── model.py
└── train.py
└── utils.py
```

## How to Run the code
Clone the repo and run
Change your current directory to S13
```
python train.py
```


## Multi-Resolution Training

## Mosaic Augmentation

## Train Loss
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/44682b76-2874-4450-aee2-5f3f3d2f76ee)

## Obj Accuracy
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/51719c5e-2b7c-4eb5-857e-96da5172b856)

## No obj Accuracy
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/46fddff2-69d4-4b0f-8626-fbe98b2a7d2b)

## Class Accuracy
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/a88fc0ae-2691-4654-ba33-063090ce5c4d)

## MAP
**52.8%**

### Link to Gradio Interface
https://huggingface.co/spaces/selvaraj-sembulingam/object-detection-with-yolov3


