# Description of Melanoma Competition

## Overview
This competition invites participants to develop a machine learning model that **aids in detecting the possibility of melanoma**. The goal is to create a model that can identify patterns in data that are associated with an increased likelihood of melanoma in visual recognition.

### Objective
The primary objective is to develop a model that can analyze photos taken by users of their skin lesions or areas of concern. 
The model should **assist users** by providing a risk assessment or likelihood score that helps them decide if they should seek further medical advice.
As a result, best model will be released in Skin Scan mobile app to run locally on the phone, and a website that will host it, free for anyone to use. 

## Evaluation Criteria
Models will be evaluated based on described **performance metrics** of the model.
The evaluation will be calculaded on following metrics with described weights.

### Performance Metrics

 The models will be assessed on the following metrics with the corresponding weights:

| **Metric**  | **Description**                                       | **Weight** |
|-------------|-------------------------------------------------------|------------|
| **F-beta**  | Prioritizes recall, with a high beta to emphasize it. $\beta = 2$ | 0.60       |
| **Accuracy**| Measures the overall correctness of predictions.      | 0.30       |
| **AUC**     | Evaluates the model's ability to distinguish classes. | 0.10       |

### Mathematical Formulas

1. **F-beta Score $F\_\beta\$**

   
   $$F_\beta = \left(1 + \beta^2\right) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\left(\beta^2 \cdot \text{Precision}\right) + \text{Recall}}$$
   

   Where:
   - **$\beta$** is the weight of recall in the combined score
   - in our case $\beta = 2$ for higher recall importance

2. **Accuracy**

   $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Number of Samples}}$$

3. **Area Under the Curve (AUC)**

   AUC is the area under the Receiver Operating Characteristic (ROC) curve. It is calculated using the trapezoidal rule:

   $$\text{AUC} = \int_0^1 \text{TPR} \, d(\text{FPR})$$

   Where:
   - **TPR** = True Positive Rate
   - **FPR** = False Positive Rate


## Model Inputs and Outputs

### Inputs
- **Input Format**: Multiple images in JPEG or PNG format.
- **Input Features**: During preprocessing, images are resized to 224x224 pixels. Images are converted to numpy arrays with a datatype of `np.float32`, normalized to the range [0, 1].

### Outputs
- **Output Format**: A numerical value between 0 and 1, represented as a `float`. This value indicates the likelihood or risk score of the area of concern warranting further investigation.

### Submission Requirements
- **Model Submission**: Models must be submitted in ONNX format. They should be capable of handling dynamic batch sizes and accept inputs with the shape `(batch , 3 , 224 , 224)`, where `batch` represents the batch dimension. This ensures that the model can process a variable number of images in a single batch.


## Rules and Guidelines

- **Timeline**:
 - every day competition will be run one or more times a day. Timings are defined in [competition_config.json](config/competition_config.json)
 - couple of minutes before start of competition, new part of dataset will be published for testing.
- Results of competition will be available on the dashboard
