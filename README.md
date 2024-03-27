<h1 align="center">Attentive Diagnostics: Enhancing Chest X-Ray Analysis Through Integrated Attention Mechanisms and SVM Classification</h1>
<p align="center">
  Juan David Gomez Villalba<br>
  Department of Computer Science<br>
  University of London
  <br>
  <span>
  <a href="mailto:mjdgv1@student.london.ac.uk">Student Email</a>
  <a href="mailto:jdavidgomezca@gmail.com">Personal Email</a><br>
  </span>
  <br>
  Deep Learning On A Public Dataset<br>
  Final Project For CM3070 Computer Science Final Project<br>
</p>


![Project Status](https://img.shields.io/badge/status-complete-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Final Report

For a comprehensive understanding of the project, methodologies, results, and conclusions, please refer to our final report:

[Read the Final Report](Attentive%20Diagnostics_%20Enhancing%20Chest%20X-Ray%20Analysis%20Through%20Integrated%20Attention%20Mechanisms%20and%20SVM%20Classification.pdf)

"Attentive Diagnostics" is the culmination of a rigorous investigation into advancing the field of medical diagnostics through artificial intelligence. This project's centerpiece is the novel integration of attention mechanisms with convolutional neural networks (CNNs), synergized with Support Vector Machine (SVM) classification to enhance diagnostic accuracy in analyzing chest X-rays (CXRs).

### Abstract
Chest X-rays are pivotal in diagnosing various thoracic diseases. However, their interpretation can be complex and error-prone. Leveraging recent strides in AI, this project introduces a methodical approach to refining the diagnosis process. By combining CNNs with attention mechanisms and SVM classification, this project sets out to create an efficient, AI-assisted method for medical image analysis.

### Datasets
This project utilizes extensive dataset of the NIH ChestX-ray14.

### Implementation
The project consists of several phases, each dedicated to different facets of model development—from prototype models using basic CNN architectures to the sophisticated integration of ResNet-18 with Convolutional Block Attention Module (CBAM). Evaluation strategies are meticulously devised, prioritizing AUC-ROC, precision-recall curves, and F-1 scores.

#### Supported Models:

- The final model is located here: [Baseline Model Notebook](models/Baseline%20Final/Baseline.ipynb)
- For an end-to-end data to model example, see: [ResNet-18 CBAM Notebook](models/Resnet-18-CBAM/Resnet-18-CBAM.ipynb)


- Baseline 
- Baseline Final
- Mobile Net 
- Resnet-18-CBAM
- Resnet-18 Old
- VGGNet19_transferLearning
- VGG_SVM
- VVGNet-16

### Results
The implementation of CBAM has notably boosted model performance, as reflected by the enhanced AUC scores across different pathologies.

### How to Use
1. Clone this repository.
3. Run the Jupyter notebooks to recreate the models and observe their performances.

### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

### Acknowledgements
A heartfelt thanks to the mentors, peers, and academic facilitators who guided and supported this project's fruition.

### Contact
For queries, please reach out to Juan David Gomez Villalba at mjdgv1@student.london.ac.uk.
