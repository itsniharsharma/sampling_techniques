🧪 Handling Class Imbalance Using Sampling Methods – Credit Card Fraud Case Study
📌 Aim of the Study

This project explores how different data sampling approaches help address class imbalance in fraud detection datasets. The goal is to observe how resampling changes model behavior and improves predictive performance when fraud cases are rare.

🎯 Task Description

The provided credit card transaction dataset is heavily skewed toward legitimate transactions. Such imbalance often causes machine learning models to ignore minority fraud cases. In this study, the dataset is rebalanced using multiple resampling strategies, and several ML models are trained to compare performance across techniques.

📂 Dataset Summary

Dataset File: Creditcard_data.csv

Total Records: 780

Input Variables: 30 (Time, V1–V28, Amount)

Output Label: Class

0 → Legitimate

1 → Fraudulent

Class Breakdown

Legitimate: 770 records

Fraudulent: 10 records

Approximate Ratio: 77 : 1

⚖️ Resampling Methods Used
Method A — SMOTE Oversampling

Generates artificial minority samples using nearest neighbors

Final distribution close to balanced (~770 vs 770)

Method B — Random Duplication Oversampling

Fraud samples duplicated randomly

Produces equal class counts

Simple but can risk overfitting

Method C — Majority Undersampling

Randomly drops normal transactions

Reduced dataset (~10 vs 10)

Faster training but heavy information loss

Method D — ADASYN Sampling

Adaptive synthetic generation focused on difficult cases

Produces slightly uneven but near-balanced classes (~772 vs 770)

Method E — SMOTE + Tomek Cleaning

SMOTE oversampling followed by Tomek link removal

Helps remove borderline noise

Final dataset around ~750 per class

🤖 Models Evaluated
Code	Algorithm
Model-A	Logistic Regression
Model-B	Decision Tree
Model-C	Random Forest
Model-D	Support Vector Classifier
Model-E	K-Nearest Neighbors
📊 Model Performance Comparison (Accuracy %)
Sampling	Model-A	Model-B	Model-C	Model-D	Model-E
Method A	92.4	96.8	98.2	69.3	87.6
Method B	93.1	98.9	99.4	76.2	97.3
Method C	35.0	65.0	52.5	68.0	64.0
Method D	91.2	95.9	97.8	67.4	85.9
Method E	92.7	96.1	98.6	68.1	86.2
🥇 Top Performing Pairings

Best Sampling per Model:

Logistic Regression → Method B — 93.1%

Decision Tree → Method B — 98.9%

Random Forest → Method B — 99.4%

SVC → Method B — 76.2%

KNN → Method B — 97.3%

Overall Winner:

Algorithm: Random Forest

Sampling: Random Duplication Oversampling

Accuracy: 99%+

🔍 Observations

Oversampling approaches generally improved fraud detection performance

Pure undersampling caused unstable and weaker results

Random Forest remained the most consistent performer

SVC showed lower accuracy compared to tree-based models

Synthetic data generation methods (SMOTE/ADASYN) worked well but not always best

▶️ Running the Project
Install Dependencies
pip install -r requirements.txt

Execute Script
python fraud_sampling_experiment.py

📁 Folder Layout
fraud_sampling_project/
├── Creditcard_data.csv
├── fraud_sampling_experiment.py
├── results_table.csv
├── sampling_heatmap.png
├── best_model_chart.png
├── performance_trend.png
├── requirements.txt
└── README.md

🧰 Tools & Libraries

Python 3

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

seaborn

🧠 Final Takeaways

Severe class imbalance must be addressed before training models

Oversampling is safer when minority class is extremely small

Random Forest handled resampled data best overall

Cleaning methods like Tomek links can improve decision boundaries

Choice of sampling strategy directly impacts model reliability