# NN & DL

### Assignments

HW1: Compare different architectures of fully connected networks by using [UTKFace Dataset](https://susanqq.github.io/UTKFace/) to determine the gender of an image and find the optimum architecture.

> Note: HW2 doesn't have the coding part.

HW3: Part A is about defining a convolutional neural network to classify images of CIFAR-10 dataset and also showing the two layers of feature maps. Part B is about adding Salt & Pepper noise to the images and then denoise them by a U-Net network.

### Project

In this project, two datasets have been used: [CASIA V.1 - Normalized Images](https://drive.google.com/drive/folders/1PP7XMeDjpv5ya2joV-AceemvrJefDQxw) and [CASIA V.1 - Feature Extracted Images](https://drive.google.com/drive/folders/16_qJWCvOwtNcyL44niUeIwgvIJUciC8S) in which both are about Iris images of 108 people. There are 3 photos of the left eye and 4 photos of the right eye for each person, leading to 756 photos total (648 train, 108 test).

Four models have been defined to learn these datasets and try to find the optimum one:
1. RBFNN
2. CNN
3. Transfer Learning based on VGG16
4. SOM + MLP

As you can see in the project file, the CNN model shows the highest accuracy in both datasets.

Also, there is a user interface with the help of the Gradio library in which you can give one of the images of datasets and the optimum model will show you its label.





