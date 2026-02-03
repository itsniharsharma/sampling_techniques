import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Sampling techniques
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='Creditcard_data.csv'):
    """Load the credit card dataset"""
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    df = pd.read_csv(filepath)
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")
    print(f"Imbalance ratio: {Counter(y)[0]/Counter(y)[1]:.2f}:1")
    return X,y

def create_five_samples(X, y):
    """Create 5 different samples using 5 sampling techniques"""
    print("\n" + "="*70)
    print("CREATING 5 SAMPLES USING DIFFERENT SAMPLING TECHNIQUES")
    print("="*70)
    samples = {}
    # Sampling1: SMOTE (Synthetic Minority Over-sampling Technique)
    print("\n1. Sampling1: SMOTE (Synthetic Minority Over-sampling)")
    smote = SMOTE(random_state=42)
    X_sample1, y_sample1 = smote.fit_resample(X, y)
    samples['Sampling1'] = (X_sample1, y_sample1)
    print(f"   After SMOTE: {Counter(y_sample1)}")
    
    # Sampling2: Random Over-sampling
    print("\n2. Sampling2: Random Over-sampling")
    ros = RandomOverSampler(random_state=42)
    X_sample2, y_sample2 = ros.fit_resample(X, y)
    samples['Sampling2'] = (X_sample2, y_sample2)
    print(f"   After Random Oversampling: {Counter(y_sample2)}")
    
    # Sampling3: Random Under-sampling
    print("\n3. Sampling3: Random Under-sampling")
    rus = RandomUnderSampler(random_state=42)
    X_sample3, y_sample3 = rus.fit_resample(X, y)
    samples['Sampling3'] = (X_sample3, y_sample3)
    print(f"   After Random Undersampling: {Counter(y_sample3)}")
    
    # Sampling4: ADASYN (Adaptive Synthetic Sampling)
    print("\n4. Sampling4: ADASYN (Adaptive Synthetic Sampling)")
    adasyn = ADASYN(random_state=42)
    X_sample4, y_sample4 = adasyn.fit_resample(X, y)
    samples['Sampling4'] = (X_sample4, y_sample4)
    print(f"   After ADASYN: {Counter(y_sample4)}")
    
    # Sampling5: SMOTETomek (Combined method)
    print("\n5. Sampling5: SMOTETomek (SMOTE + Tomek Links)")
    smote_tomek = SMOTETomek(random_state=42)
    X_sample5, y_sample5 = smote_tomek.fit_resample(X, y)
    samples['Sampling5'] = (X_sample5, y_sample5)
    print(f"   After SMOTETomek: {Counter(y_sample5)}")
    
    return samples


def create_five_models():
    """Create 5 different ML models"""
    print("\n" + "="*70)
    print("INITIALIZING 5 MACHINE LEARNING MODELS")
    print("="*70)
    
    models = {
        'M1': LogisticRegression(max_iter=1000, random_state=42),
        'M2': DecisionTreeClassifier(random_state=42),
        'M3': RandomForestClassifier(n_estimators=100, random_state=42),
        'M4': SVC(kernel='rbf', random_state=42),
        'M5': KNeighborsClassifier(n_neighbors=5)
    }
    
    model_names = {
        'M1': 'Logistic Regression',
        'M2': 'Decision Tree',
        'M3': 'Random Forest',
        'M4': 'Support Vector Machine (SVM)',
        'M5': 'K-Nearest Neighbors (KNN)'
    }
    
    for key, name in model_names.items():
        print(f"{key}: {name}")
    
    return models, model_names


def evaluate_all_combinations(samples, models):
    """Apply 5 sampling techniques on 5 ML models and calculate accuracy"""
    print("\n" + "="*70)
    print("EVALUATING ALL COMBINATIONS (5 Samples × 5 Models = 25 Tests)")
    print("="*70)
    
    results = {}
    
    for sample_name, (X_sample, y_sample) in samples.items():
        results[sample_name] = {}
        
        print(f"\n{sample_name}:")
        print("-" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
        )
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred) * 100
            results[sample_name][model_name] = accuracy
            
            print(f"  {model_name}: {accuracy:.2f}%")
    
    return results


def create_results_table(results, model_names):
    """Create and display results table"""
    print("\n" + "="*70)
    print("RESULTS TABLE - ACCURACY SCORES (%)")
    print("="*70)
    
    # Create DataFrame
    df_results = pd.DataFrame(results).T
    df_results = df_results[['M1', 'M2', 'M3', 'M4', 'M5']]  # Ensure column order
    
    print("\n" + df_results.to_string())
    
    # Save to CSV
    df_results.to_csv('sampling_results.csv')
    print("\n✓ Results saved to: sampling_results.csv")
    
    return df_results


def find_best_combinations(df_results, model_names):
    """Determine which sampling technique gives higher accuracy on which model"""
    print("\n" + "="*70)
    print("ANALYSIS - BEST SAMPLING TECHNIQUE FOR EACH MODEL")
    print("="*70)
    
    print("\n📊 Best Sampling Technique for Each Model:")
    print("-" * 50)
    
    for model in df_results.columns:
        best_sampling = df_results[model].idxmax()
        best_accuracy = df_results[model].max()
        print(f"{model} ({model_names[model]}): {best_sampling} with {best_accuracy:.2f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS - BEST MODEL FOR EACH SAMPLING TECHNIQUE")
    print("="*70)
    
    print("\n📊 Best Model for Each Sampling Technique:")
    print("-" * 50)
    
    for sampling in df_results.index:
        best_model = df_results.loc[sampling].idxmax()
        best_accuracy = df_results.loc[sampling].max()
        print(f"{sampling}: {best_model} ({model_names[best_model]}) with {best_accuracy:.2f}%")
    
    # Overall best combination
    print("\n" + "="*70)
    print("🏆 OVERALL BEST COMBINATION")
    print("="*70)
    
    max_accuracy = df_results.max().max()
    best_model = df_results.max().idxmax()
    best_sampling = df_results[best_model].idxmax()
    
    print(f"\nBest Model: {best_model} ({model_names[best_model]})")
    print(f"Best Sampling: {best_sampling}")
    print(f"Accuracy: {max_accuracy:.2f}%")


def visualize_results(df_results, model_names):
    """Create visualizations of results"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_results, annot=True, fmt='.2f', cmap='YlGnBu', 
                cbar_kws={'label': 'Accuracy (%)'}, linewidths=0.5)
    plt.title('Accuracy Scores: Sampling Techniques vs ML Models', fontsize=16, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Sampling Techniques', fontsize=12)
    plt.tight_layout()
    plt.savefig('heatmap_sampling_vs_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: heatmap_sampling_vs_models.png")
    
    # 2. Bar plot - Best sampling for each model
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    best_per_model = df_results.idxmax()
    best_acc_per_model = df_results.max()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = plt.bar(range(len(best_acc_per_model)), best_acc_per_model, color=colors)
    plt.xticks(range(len(best_acc_per_model)), 
               [f"{m}\n({model_names[m]})" for m in best_acc_per_model.index], 
               fontsize=9)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Best Accuracy for Each Model', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, val, samp) in enumerate(zip(bars, best_acc_per_model, best_per_model)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%\n{samp}', ha='center', va='bottom', fontsize=8)
    
    # 3. Bar plot - Best model for each sampling
    plt.subplot(1, 2, 2)
    best_per_sampling = df_results.T.idxmax()
    best_acc_per_sampling = df_results.T.max()
    
    bars = plt.bar(range(len(best_acc_per_sampling)), best_acc_per_sampling, color=colors)
    plt.xticks(range(len(best_acc_per_sampling)), best_acc_per_sampling.index, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Best Accuracy for Each Sampling Technique', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, val, mod) in enumerate(zip(bars, best_acc_per_sampling, best_per_sampling)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%\n{mod}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('best_combinations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: best_combinations.png")
    
    # 4. Line plot comparing all models across sampling techniques
    plt.figure(figsize=(12, 6))
    for model in df_results.columns:
        plt.plot(df_results.index, df_results[model], marker='o', label=f"{model} ({model_names[model]})", linewidth=2)
    
    plt.xlabel('Sampling Techniques', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance Across Different Sampling Techniques', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('performance_comparison_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: performance_comparison_line.png")


def main():
    # Step 1: Load data
    X, y = load_data()
    
    # Step 2: Create 5 samples using different sampling techniques
    samples = create_five_samples(X, y)
    
    # Step 3: Create 5 ML models
    models, model_names = create_five_models()
    
    # Step 4: Apply sampling techniques on ML models
    results = evaluate_all_combinations(samples, models)
    
    # Step 5: Create results table
    df_results = create_results_table(results, model_names)
    
    # Step 6: Determine best combinations
    find_best_combinations(df_results, model_names)
    
    # Step 7: Visualize results
    visualize_results(df_results, model_names)
    
    print("  1. sampling_results.csv - Accuracy results table")
    print("  2. heatmap_sampling_vs_models.png - Heatmap visualization")
    print("  3. best_combinations.png - Best combinations bar charts")
    print("  4. performance_comparison_line.png - Line plot comparison")
    print("\n")


if __name__ == "__main__":
    main()
