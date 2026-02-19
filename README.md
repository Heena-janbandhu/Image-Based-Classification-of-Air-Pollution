**🌫 Air Pollution Image Classification using Transfer Learning**
📌 Project Description

This project applies transfer learning using a pretrained ResNet50 convolutional neural network to classify air pollution levels from outdoor images. The model is fine-tuned on a real-world air quality image dataset and evaluated using multiple classification metrics. The results are compared with a published research paper to analyze performance differences and improvements.

**📚 Research Paper Used**

Image-Based Classification of Air Pollution Using Different Pretrained CNN Models and a Small Dataset

The paper explores the use of pretrained CNN architectures such as ResNet50 for air pollution image classification and reports strong accuracy using transfer learning techniques.

**📂 Dataset**

Name: Air Quality Image Dataset (Multi-Class)
Source: Kaggle

Link:
https://www.kaggle.com/datasets/pratik2901/air-quality-image-dataset

Classes:

Good

Moderate

Poor

Severe

The dataset contains outdoor images representing different air pollution conditions.

**⚙ Methodology**

Dataset loading and preprocessing

Image resizing to 224×224

Data augmentation

Train-validation-test split

Transfer learning using ResNet50

Fine-tuning of top layers

Model evaluation

**🧠 Model Architecture**

Base Model: ResNet50 (ImageNet pretrained)

Custom Layers:

Global Average Pooling

Dense (256 neurons, ReLU)

Dropout (0.5)

Softmax output layer

Optimizer: Adam

Loss Function: Categorical Crossentropy

**📊 Results**
Metric	Value
Accuracy	76%
Precision (Weighted)	0.79
Recall (Weighted)	0.76
F1-score (Weighted)	0.76

The model performs best on high pollution categories and shows some confusion between visually similar classes.

**📈 Visualizations Included**

Training & validation accuracy curve

Training & validation loss curve

Confusion matrix

Classification report

**📊 Comparison with Research Paper**
Model	Paper Accuracy	Our Accuracy
ResNet50	~84%	76%

The difference is mainly due to dataset diversity and real-world variability.

**🔍 Key Observations**

Transfer learning significantly improved performance

Fine-tuning enhanced classification accuracy

Visual similarity affected lower pollution class predictions

**🚀 Future Improvements**

Use EfficientNet or advanced CNNs

Apply stronger data augmentation

Handle class imbalance

Learning rate scheduling

Model explainability techniques

**▶ How to Run**

Install dependencies:

pip install tensorflow numpy matplotlib scikit-learn


Open and run:

Image_Based_Classification_of_Air_Pollution.ipynb


Download dataset from Kaggle link provided.

**👩‍💻 Author**

Heena Janbandhu
B.Tech Electronics & Telecommunication
