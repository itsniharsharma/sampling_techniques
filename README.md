# Handling Class Imbalance Using Sampling Methods – Credit Card Fraud Case Study

## Overview

This project explores how different data sampling approaches help address **class imbalance** in fraud detection datasets. The goal is to observe how resampling changes model behavior and improves predictive performance when fraud cases are rare.

## Problem Statement

The provided credit card transaction dataset is heavily skewed toward legitimate transactions. Such imbalance often causes machine learning models to ignore minority fraud cases. This study rebalances the dataset using multiple resampling strategies and trains several ML models to compare performance across techniques.

## Dataset Summary

| Attribute | Details |
|-----------|---------|
| **File** | `Creditcard_data.csv` |
| **Total Records** | 780 |
| **Input Variables** | 30 (Time, V1–V28, Amount) |
| **Output Label** | Class |
| **Class Distribution** | |
| Legitimate (0) | 770 records |
| Fraudulent (1) | 10 records |
| **Imbalance Ratio** | 77:1 |

## Resampling Methods

### Method A: SMOTE Oversampling
- Generates artificial minority samples using nearest neighbors
- Final distribution: ~balanced (770 vs 770)

### Method B: Random Duplication Oversampling
- Duplicates fraud samples randomly
- Produces equal class counts
- Simple approach but risk of overfitting

### Method C: Majority Undersampling
- Randomly removes normal transactions
- Reduced dataset (~10 vs 10)
- Faster training but significant information loss

### Method D: ADASYN Sampling
- Adaptive synthetic generation focused on difficult cases
- Produces near-balanced classes (~772 vs 770)

### Method E: SMOTE + Tomek Cleaning
- SMOTE oversampling followed by Tomek link removal
- Reduces borderline noise
- Final dataset: ~750 per class

## Models Evaluated

| Code | Algorithm |
|------|-----------|
| Model-A | Logistic Regression |
| Model-B | Decision Tree |
| Model-C | Random Forest |
| Model-D | Support Vector Classifier |
| Model-E | K-Nearest Neighbors |

## Performance Results

### Accuracy Comparison (%)

| Sampling Method | Model-A | Model-B | Model-C | Model-D | Model-E |
|-----------------|---------|---------|---------|---------|---------|
| Method A (SMOTE) | 92.4 | 96.8 | 98.2 | 69.3 | 87.6 |
| Method B (Random Duplication) | 93.1 | 98.9 | **99.4** | 76.2 | 97.3 |
| Method C (Undersampling) | 35.0 | 65.0 | 52.5 | 68.0 | 64.0 |
| Method D (ADASYN) | 91.2 | 95.9 | 97.8 | 67.4 | 85.9 |
| Method E (SMOTE + Tomek) | 92.7 | 96.1 | 98.6 | 68.1 | 86.2 |

### Top Performing Configuration

- **Best Algorithm:** Random Forest
- **Best Sampling Method:** Method B (Random Duplication Oversampling)
- **Accuracy:** 99.4%

### Results by Model

| Model | Best Method | Accuracy |
|-------|-------------|----------|
| Logistic Regression | Method B | 93.1% |
| Decision Tree | Method B | 98.9% |
| Random Forest | Method B | **99.4%** ⭐ |
| SVC | Method B | 76.2% |
| KNN | Method B | 97.3% |

## Key Observations

- ✅ Oversampling approaches generally improved fraud detection performance
- ❌ Pure undersampling caused unstable and weaker results
- 📈 Random Forest remained the most consistent performer
- 📊 SVC showed lower accuracy compared to tree-based models
- 🔄 Synthetic data generation methods (SMOTE/ADASYN) worked well but not always best

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sampling_techniques.git
cd sampling_techniques
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the analysis script:
```bash
python sampling_assignment.py
```

## Project Structure

```
sampling_techniques/
├── Creditcard_data.csv           # Input dataset
├── sampling_assignment.py        # Main analysis script
├── sampling_results.csv          # Output results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dependencies

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models
- **imbalanced-learn** - Resampling methods (SMOTE, ADASYN)
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization

See [requirements.txt](requirements.txt) for specific versions.

## Conclusions

1. **Severe class imbalance must be addressed** before training models
2. **Oversampling is safer** when the minority class is extremely small
3. **Random Forest handled resampled data best** overall
4. **Cleaning methods** like Tomek links can improve decision boundaries
5. **Sampling strategy directly impacts model reliability**

## Author

[Your Name](https://github.com/itsniharsharma)

## License

This project is open source and available under the MIT License.