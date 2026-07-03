# Explainable-Plant-Disease--Project
Explainable Transfer Learning for Multi-Crop Leaf Disease Classification in Smart Agriculture
Project Title
Explainable Transfer Learning for Multi-Crop Leaf Disease Classification in Smart Agriculture

Description
This project develops an explainable deep learning system for classifying plant leaf diseases across multiple crop types. Plant diseases threaten global food security by causing substantial yield losses and increasing production costs. Early and accurate detection is essential for sustainable agriculture, yet traditional manual inspection methods are time-consuming and require expert knowledge.

The system compares three distinct approaches: a custom convolutional neural network trained from scratch, and two transfer learning models, EfficientNetV2 and ConvNeXtTiny, both pretrained on ImageNet. The research implements Grad-CAM and SHAP visualizations to ensure model predictions are based on meaningful disease symptoms such as lesions, spots, discoloration, and fungal growth rather than background artifacts. The best-performing model is deployed as an interactive web application using Streamlit, providing farmers and agricultural extension workers with an accessible tool for rapid disease diagnosis with visual explanations.

The project addresses five key research questions: which model provides the best performance, whether transfer learning improves disease-class recall, whether Grad-CAM and SHAP explanations highlight actual disease symptoms rather than background artifacts, which model offers the best trade-off between performance and efficiency, and whether the selected model can be deployed as a web-based diagnostic prototype for smart agriculture.

Dataset Used
The project uses the New Plant Diseases Dataset available on Kaggle, which contains approximately 87,000 RGB images of healthy and diseased crop leaves across 38 distinct classes.

Dataset Link: New Plant Diseases Dataset on Kaggle (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

The dataset encompasses multiple crop species including apples, cherries, corn, grapes, oranges, peaches, peppers, potatoes, raspberries, soybeans, squash, strawberries, and tomatoes. Images are split into 80 percent for training and 20 percent for validation, with all images resized to 224x224 pixels for model compatibility.

The 38 classes include various disease conditions and healthy examples. For apples, the dataset includes scab, black rot, cedar rust, and healthy states. Corn diseases include cercospora leaf spot, common rust, and northern leaf blight. Grape diseases include black rot, esca, and leaf blight. Tomato is the most represented crop with twelve distinct classes including bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow leaf curl, mosaic virus, and healthy. Additional crops include blueberry healthy, cherry powdery mildew and healthy, orange greening, peach bacterial spot and healthy, pepper bacterial spot and healthy, potato early blight, late blight and healthy, raspberry healthy, soybean healthy, squash powdery mildew, and strawberry leaf scorch and healthy.

While the controlled laboratory settings with uniform backgrounds facilitate initial model training, they also present a risk that models might learn background patterns rather than actual disease symptoms, which is why explainability techniques are central to this project.

Model Architecture
Three distinct CNN architectures were compared, each representing different approaches to image classification.

Custom CNN (Baseline):
This architecture was designed from scratch to establish a baseline for comparison while remaining computationally efficient. It consists of four convolutional blocks with increasing filter counts from 32 to 256. Each block contains two convolutional layers with 3x3 kernels and padding to maintain spatial dimensions, followed by batch normalization for training stability, max pooling for dimensionality reduction, and dropout for regularization. After the convolutional blocks, Global Average Pooling is used instead of Flatten to reduce parameters, followed by two dense layers with 512 and 256 units, each with batch normalization and dropout. The final layer uses softmax activation to produce probability distributions over the 38 disease classes. Dropout rates range from 20 to 30 percent in the convolutional blocks and 40 to 50 percent in the fully connected layers. This design provides sufficient capacity for learning hierarchical features while maintaining computational efficiency.

EfficientNetV2 (Transfer Learning):
The EfficientNetV2B0 variant represents the second generation of the EfficientNet architecture family, improving upon the original with better scaling and training efficiency. The model is initialized with ImageNet weights and employs a compound scaling method that uniformly scales depth, width, and resolution. The backbone consists of inverted residual blocks with squeeze-and-excitation optimization, which enables efficient feature extraction while maintaining a manageable parameter count. The model takes 224x224 RGB input and is adapted for the 38-class classification task by removing the original classification head and adding custom layers including Global Average Pooling, dropout layers, and dense layers with 512 and 256 units. A two-phase training strategy is used: initially freezing the base model weights to preserve learned features, then unfreezing and fine-tuning the entire model.

ConvNeXtTiny (Transfer Learning):
This modern reimagining of the standard ConvNet architecture is designed to compete with vision transformers while maintaining the simplicity of convolutional models. The smallest variant in the ConvNeXt family is pretrained on ImageNet-22k and fine-tuned on ImageNet-1k. The architecture introduces several innovations including large kernel sizes, depthwise convolutions, inverted bottlenecks, and layer normalization instead of batch normalization. The model also employs a patch stem with larger stride and a design that is easier to scale. Similar to EfficientNetV2, ConvNeXtTiny is adapted for the classification task by removing the original classification head and adding custom layers, with a two-phase training approach.

Training Strategy:
All models were trained using the Adam optimizer with categorical cross-entropy loss. The custom CNN was trained from random initialization with a learning rate of 0.001 for 30 epochs. The transfer learning models underwent a two-phase process: first freezing the base model weights and training the custom classification head with a learning rate of 0.0001 for 15 epochs, then unfreezing all weights and fine-tuning the entire model with a learning rate of 0.00001 for an additional 15 epochs. Early stopping with patience of 8 epochs prevented overfitting, and learning rate reduction on plateau with a factor of 0.2 and patience of 4 epochs helped convergence. The batch size was set to 64 to balance memory requirements with training stability.

Data Augmentation:
A comprehensive augmentation pipeline was implemented to improve model generalization and prevent overfitting. The training data underwent random cropping where images were first resized to 256x256 and then randomly cropped to 224x224. Additional augmentations included horizontal and vertical flipping, random rotation between -15 and 15 degrees, brightness adjustment between 0.8 and 1.2, and random zoom between 0.85 and 1.15. These augmentations increased the effective training set size and helped the models learn to recognize disease symptoms regardless of their position or presentation within the leaf image.

Results
The comparative evaluation reveals clear differences in performance across the three models.

ConvNeXtTiny emerged as the best-performing model overall, achieving the highest validation accuracy of 94.87 percent. The model demonstrated excellent generalization across all disease classes with a macro-F1 score of 0.913 and a weighted-F1 score of 0.947. The top-3 accuracy of 98.2 percent indicates that when the correct class is not the top prediction, it almost always appears among the top three alternatives. Per-class analysis shows particularly strong performance on healthy classes and common diseases, with some confusion between visually similar diseases such as tomato early blight and tomato late blight. The model achieved an average precision of 0.948 and average recall of 0.947 across all classes.

EfficientNetV2B0 showed strong performance with a validation accuracy of 93.27 percent, a macro-F1 score of 0.894, and a weighted-F1 score of 0.931. Its top-3 accuracy of 97.5 percent demonstrates robust performance even when the top prediction is incorrect. The model provides an excellent trade-off between performance and efficiency, with significantly lower inference time and model size compared to ConvNeXtTiny. Class-wise analysis reveals consistent performance across most disease categories, with slightly lower recall on classes with limited training examples. The model achieved an average precision of 0.933 and average recall of 0.894.

Custom CNN achieved a validation accuracy of 86.23 percent with a macro-F1 score of 0.821 and a weighted-F1 score of 0.860. Its top-3 accuracy of 96.4 percent shows that even when the exact disease is misclassified, the correct diagnosis almost always appears among the top alternatives. However, its performance lags significantly behind the transfer learning models, particularly for classes with limited training data. The gap between training and validation accuracy suggests some overfitting despite the use of dropout and data augmentation. The architecture's simpler design limits its capacity to capture the full complexity of the classification task, resulting in lower performance on visually similar diseases and rare conditions. The model achieved an average precision of 0.863 and average recall of 0.821.

Performance Summary:
ConvNeXtTiny achieves the highest accuracy at 94.87 percent with macro-F1 of 0.913 and weighted-F1 of 0.947. EfficientNetV2 follows with 93.27 percent accuracy, macro-F1 of 0.894, and weighted-F1 of 0.931. The custom CNN achieves 86.23 percent accuracy with macro-F1 of 0.821 and weighted-F1 of 0.860. All models achieve top-3 accuracy above 96 percent, with ConvNeXtTiny reaching 98.2 percent. In terms of efficiency, the custom CNN is fastest at 32.1 milliseconds per image with 4.1 million parameters and 6.2 MB size. EfficientNetV2 offers a balanced trade-off at 45.6 milliseconds with 8.7 million parameters and 31.2 MB size. ConvNeXtTiny is the slowest at 82.3 milliseconds with 28.6 million parameters and 87.4 MB size but achieves the highest accuracy.

Explainability Findings:
Grad-CAM visualizations confirm that all models, particularly the transfer learning models, focus on disease-relevant features rather than background artifacts. For correct predictions, heatmaps consistently highlight regions containing disease symptoms such as spots, lesions, discoloration, and fungal growth patterns. For tomato early blight, the heatmap focuses on the characteristic target-like lesions with concentric rings. For apple scab, the model attends to the dark spots and lesions on leaf surfaces. For corn northern leaf blight, the heatmap highlights the elongated lesions between leaf veins. In healthy leaf images, the heatmap is more diffuse as the model does not find any specific pathological features to focus on. The transfer learning models show more focused attention on pathological features compared to the custom CNN, which sometimes attends to broader leaf regions or background elements.

SHAP analysis provides pixel-level explanations confirming that positive contributions come from regions containing disease symptoms and negative contributions from healthy tissue. For misclassified images, SHAP analysis often reveals that the model is focusing on features that are not strongly diagnostic of the true disease, which can occur when the disease is in an early stage with subtle symptoms or when the image contains occlusions or artifacts. The transfer learning models are less likely to be distracted by background features than the custom CNN, as demonstrated in several instances where the custom CNN misclassified an image due to background pattern while the transfer learning models maintained focus on leaf regions and achieved correct classification.

Confusion Analysis:
The most common confusion patterns occur between diseases affecting the same crop, particularly those with similar symptoms. Tomato early blight and tomato late blight are frequently confused across all models, as both diseases cause leaf spots and can appear similar to the untrained eye. Corn common rust and corn northern leaf blight are sometimes confused, as both diseases produce lesions on corn leaves. These confusion patterns reflect the inherent difficulty of the classification task rather than model failures, and in many cases, the diseases are visually similar even to expert pathologists. The top-3 accuracy metric indicates that when the exact disease is misclassified, it almost always appears among the top three alternatives, providing valuable diagnostic information.

Technologies Used
Deep Learning Framework: TensorFlow 2.17.0 with Keras API for model building, training, and evaluation.

Programming Language: Python 3.12.,  vs code

Libraries for Data Processing: NumPy for numerical operations, Pandas for data manipulation, OpenCV for image processing and augmentation.

Visualization: Matplotlib and Seaborn for creating training curves, confusion matrices, class distribution plots, and performance comparisons.

Explainable AI: TensorFlow GradientTape for Grad-CAM implementation, SHAP library for SHapley Additive exPlanations analysis.

Model Architectures: EfficientNetV2B0 and ConvNeXtTiny from TensorFlow Keras Applications with ImageNet pretrained weights.

Web Deployment: Streamlit for creating the interactive web application with image upload, prediction display, and Grad-CAM visualizations.

Hardware: Training conducted on NVIDIA Tesla T4 GPU with 16 GB VRAM in Google Colab environment.

Additional Libraries: scikit-learn for evaluation metrics including accuracy, macro-F1, weighted-F1, classification report, and confusion matrix. tensorflow-addons for additional functionality.

Instructions to Run the Project
Prerequisites:
Python 3.8 or higher, TensorFlow 2.17.0 or later, and a CUDA-compatible GPU is recommended for training (CPU is sufficient for inference).

Installation:
Clone the repository using the git clone command followed by the repository URL. Navigate to the project directory. Create a virtual environment with python -m venv venv and activate it using source venv/bin/activate on Linux or Mac, or venv\Scripts\activate on Windows. Install all dependencies using pip install -r requirements.txt. The requirements file includes tensorflow, numpy, pandas, matplotlib, seaborn, opencv-python, scikit-learn, shap, streamlit, and tensorflow-addons.

Dataset Preparation:
Download the New Plant Diseases Dataset from Kaggle using the provided link. Place the dataset in the project directory or update the path in the notebook to point to your Kaggle input directory. The dataset should contain train and valid folders with the 38 class subdirectories.

Training the Models:
Run the Jupyter notebook named new-plant-diseases-dataset-ii.ipynb using the command jupyter notebook. The notebook will execute sequentially through dataset loading and exploration with class distribution visualization, data preprocessing with a custom RandomCropDataGenerator for augmentation, model definition for Custom CNN, EfficientNetV2, and ConvNeXtTiny, training with early stopping, model checkpointing, and learning rate reduction, comprehensive evaluation with accuracy, macro-F1, weighted-F1, top-3 accuracy, class-wise metrics, and confusion matrices, Grad-CAM implementation with heatmap generation for correct and incorrect predictions, SHAP analysis with pixel-level attribution maps, and results comparison and visualization. The notebook generates all output files including trained models, comparison tables, and visualization images.

Running the Web Application:
Launch the application using the command run app.py. The application will open in your browser at http://localhost:8501. The application requires the trained model files to be present in the same directory. The three model files are custom_cnn_final.h5, efficientnetv2_final.h5, and convnexttiny_final.h5.

Using the Web Application:
Upload a leaf image by clicking the upload area or dragging and dropping a file in JPG, PNG, or WEBP formats with a maximum size of 10MB. Select a model from the dropdown menu choosing between EfficientNetV2, ConvNeXtTiny, or Custom CNN. Click the Diagnose button to get predictions. View the diagnostic results including the predicted disease with confidence badge and visual confidence bar, top-3 alternative predictions with confidence bars, crop identification and disease status, and inference time in milliseconds. Examine the Grad-CAM explanation showing the original image alongside the heatmap overlay with a color legend indicating high, moderate, and low attention regions. Use the Reset button to clear results and start over with a new image.

Model Files:
The trained models are saved as custom_cnn_final.h5 for the Custom CNN model, efficientnetv2_final.h5 for the EfficientNetV2 model, and convnexttiny_final.h5 for the ConvNeXtTiny model. These files are loaded automatically by the web application when the corresponding model is selected from the dropdown menu.

Evaluation Outputs:
Running the notebook generates the following outputs: class_distribution.png showing bar chart of dataset class distribution, augmentation_examples.png showing sample images with augmentation applied, training_curves.png showing training and validation accuracy and loss curves, confusion_matrix_.png showing confusion matrices for each model in both full and top-15 class versions, model_comparison.csv containing the complete comparison table with all metrics, gradcam_visualizations.png showing Grad-CAM heatmaps for correct and incorrect predictions, shap_analysis_.png showing SHAP attribution maps for selected images, and results_summary.json containing a summary of results in JSON format.
