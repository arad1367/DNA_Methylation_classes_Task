import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import os

class MethylationAnalyzer:
    def __init__(self, data_path, random_state=42):
        """
        Initialize the Methylation Data Analyzer.
        
        Parameters:
        -----------
        data_path : str
            Path to the methylation data CSV file
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"analysis_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and preprocess data
        self.data = self._load_data(data_path)
        self.X, self.y = self._initial_preprocessing()
        
    def _load_data(self, data_path):
        """Load and perform initial data checks."""
        data = pd.read_csv(data_path)
        print("\nData Loading Summary:")
        print(f"Total samples: {data.shape[0]}")
        print(f"Total features: {data.shape[1]}")
        return data
    
    def _initial_preprocessing(self):
        """Perform initial preprocessing steps."""
        X = self.data.drop(['Unnamed: 0', 'label'], axis=1)
        y = self.data['label']
        
        # Check for missing values
        missing_stats = self._analyze_missing_values(X)
        self.missing_stats = missing_stats
        
        return X, y
    
    def _analyze_missing_values(self, X):
        """Analyze missing values in the dataset."""
        missing_stats = pd.DataFrame({
            'Missing Count': X.isnull().sum(),
            'Missing Percentage': (X.isnull().sum() / len(X) * 100).round(2)
        }).sort_values('Missing Percentage', ascending=False)
        
        print("\nMissing Values Analysis:")
        print(f"Features with missing values: {(missing_stats['Missing Count'] > 0).sum()}")
        return missing_stats
    
    def preprocess_data(self, scaler_type='standard'):
        """
        Preprocess the data with comprehensive checks and transformations.
        
        Parameters:
        -----------
        scaler_type : str
            Type of scaler to use ('standard' or 'robust')
        """
        # Label encoding
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)
        print("\nClass Encoding:")
        for i, label in enumerate(self.le.classes_):
            print(f"{label}: {i}")
        
        # Feature scaling
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Feature selection
        self.selector = SelectKBest(f_classif, k=min(1000, self.X.shape[1]))
        self.X_selected = self.selector.fit_transform(self.X_scaled, self.y_encoded)
        
        # Save important features
        feature_scores = pd.DataFrame({
            'Feature': self.X.columns[self.selector.get_support()],
            'Score': self.selector.scores_[self.selector.get_support()]
        }).sort_values('Score', ascending=False)
        
        feature_scores.to_csv(f"{self.output_dir}/important_features.csv", index=False)
        
        return self.X_selected
    
    def perform_eda(self):
        """Perform comprehensive exploratory data analysis."""
        # Class distribution
        class_dist = pd.DataFrame({
            'Count': self.y.value_counts(),
            'Percentage': (self.y.value_counts() / len(self.y) * 100).round(2)
        })
        
        # Feature statistics
        feature_stats = pd.DataFrame({
            'Mean': self.X_scaled.mean(axis=0),
            'Std': self.X_scaled.std(axis=0),
            'Min': self.X_scaled.min(axis=0),
            'Max': self.X_scaled.max(axis=0)
        })
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_dist.index, y='Count', data=class_dist)
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_distribution.png")
        
        return {
            'class_distribution': class_dist,
            'feature_statistics': feature_stats
        }
    
    def reduce_dimensions(self):
        """Perform multiple dimensionality reduction techniques with visualizations."""
        reducers = {
            'PCA': PCA(n_components=3, random_state=self.random_state),
            'TSNE': TSNE(n_components=3, random_state=self.random_state),
            'UMAP': UMAP(n_components=3, random_state=self.random_state)
        }
        
        reduced_data = {}
        for name, reducer in reducers.items():
            # Perform reduction
            reduced = reducer.fit_transform(self.X_selected)
            reduced_data[name] = reduced
            
            # Create 3D visualization
            fig = px.scatter_3d(
                x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                color=self.y,
                title=f'3D {name} Visualization',
                labels={'color': 'Class'},
                template='plotly_white'
            )
            
            # Save visualization
            pio.write_html(fig, f"{self.output_dir}/{name.lower()}_3d_viz.html")
            
            # Calculate explained variance for PCA
            if name == 'PCA':
                explained_var = reducer.explained_variance_ratio_
                print(f"\nPCA Explained Variance Ratio: {explained_var.cumsum()[-1]:.3f}")
        
        return reduced_data
    
    def train_and_evaluate_models(self, X_reduced):
        """Train and evaluate multiple classification models."""
        models = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'SVM': SVC(probability=True, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state)
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, self.y_encoded,
            test_size=0.2,
            stratify=self.y_encoded,
            random_state=self.random_state
        )
        
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'classification_report': classification_report(
                    y_test, y_pred,
                    target_names=self.le.classes_,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                results[name]['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.le.classes_,
                yticklabels=self.le.classes_
            )
            plt.title(f'Confusion Matrix - {name}')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/confusion_matrix_{name.lower().replace(' ', '_')}.png")
            
            # Plot ROC curves
            self._plot_roc_curves(y_test, y_prob, name)
        
        # Save results summary
        results_df = pd.DataFrame({
            name: {
                'Accuracy': res['accuracy'],
                'F1 Score (Macro)': res['f1_macro']
            }
            for name, res in results.items()
        }).T
        
        results_df.to_csv(f"{self.output_dir}/model_performance_summary.csv")
        
        return results
    
    def _plot_roc_curves(self, y_test, y_prob, model_name):
        """Plot ROC curves for multiclass classification."""
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.le.classes_):
            y_true_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/roc_curves_{model_name.lower().replace(' ', '_')}.png")
        plt.close()

def main():
    # Initialize analyzer
    analyzer = MethylationAnalyzer('Data/methylation_with_labels.csv')
    
    # Preprocess data
    X_selected = analyzer.preprocess_data(scaler_type='robust')
    
    # Perform EDA
    eda_results = analyzer.perform_eda()
    print("\nClass Distribution Summary:")
    print(eda_results['class_distribution'])
    
    # Perform dimensionality reduction
    reduced_data = analyzer.reduce_dimensions()
    
    # Train and evaluate models
    results = analyzer.train_and_evaluate_models(reduced_data['PCA'])
    
    print("\nAnalysis Complete! Check the output directory for detailed results and visualizations.")

if __name__ == "__main__":
    main()