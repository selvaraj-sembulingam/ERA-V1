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

## Validation Images

### Epoch 10
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/25281872-351a-4c94-b3f5-0252d05a8b2a)

### Epoch 20
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/bb521e05-d7be-43eb-b431-a7a2b377ec46)

### Epoch 30
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/b4d9a0f4-d5cc-4d1a-994b-e0b40f79d2f3)

### Epoch 40
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ea9ccca0-724c-40d7-b395-3584f7c0db51)

### Link to Gradio Interface
https://huggingface.co/spaces/selvaraj-sembulingam/object-detection-with-yolov3


