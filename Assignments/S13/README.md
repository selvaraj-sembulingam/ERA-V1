# Object Detection with YOLO-V3

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
Spatial Pyramid Pooling (SPP) is a technique used to handle input images of different sizes in convolutional neural networks, allowing the network to efficiently process images with varying dimensions without the need for resizing. In the context of YOLOv3, SPP plays a crucial role in improving the model's flexibility and performance.

In YOLOv3, the network processes an image at three different scales: large, medium, and small, using different detection layers. Each detection layer is responsible for detecting objects of different sizes. However, the YOLO architecture requires a fixed-size input image, which can limit the model's ability to handle various object scales and aspect ratios.

SPP addresses this limitation by allowing the network to process images of different sizes and aspect ratios without the need for resizing. This is achieved by pooling features from different sub-regions of the input image at multiple scales. By doing so, SPP captures context and spatial information at different levels of granularity, enabling YOLOv3 to handle objects of varying sizes effectively.


## Mosaic Augmentation
Mosaic data augmentation combines 4 training images into one in random proportions. The algorithms is the following:

* Take 4 images from the train set;
* Resize them into the same size;
* Integrate the images into a 4x4 grid;
* Crop a random image patch from the center. This will be the final augmented image.

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/6a19fb32-ca83-4e0f-98df-42fb278a8956)

Mosaic augmentation teaches the model to recognize objects in different localizations without relying too much on one specific context. This boosts the model’s performance by making the algorithm more robust to the surroundings of the objects.


## Train Metrics
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/a54b957d-bca9-4117-a7f8-98d0be8ff0c8)


## Val Metrics
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/4ffac9f0-2f64-411d-a038-10e1dddc8648)


## Epoch
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/eece0145-9cae-435c-a0f6-891277a6c33e)


## LR
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/8e33f6f5-af0b-4fe2-9708-a2ff4735a2ec)


## MAP
**43.8%**

## Validation Images

### Epoch 10
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/c6bdffdb-9a06-4134-97c1-08fd175e0910)


### Epoch 20
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/67f08b98-359f-48df-a7a2-2f1de8ac8bde)


### Epoch 30
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/24ca5831-617b-457d-b9c9-a235bff2dff8)


### Epoch 40
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/581eb40c-aa7a-4fb8-8a04-3f599bb3aca3)


### Link to Gradio Interface
https://huggingface.co/spaces/selvaraj-sembulingam/object-detection-with-yolov3


