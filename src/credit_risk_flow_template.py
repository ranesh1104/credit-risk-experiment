"""
Metaflow pipeline template for Credit Risk Modeling

This is a TEMPLATE to help you get started with the bonus section.
Feel free to modify, extend, or completely rewrite this structure.

To run:
    python credit_risk_flow_template.py run
    
To view results:
    python credit_risk_flow_template.py show
"""

from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class CreditRiskFlow(FlowSpec):
    """
    Credit Risk Modeling Pipeline
    
    A production-ready ML pipeline that:
    1. Loads and validates data
    2. Handles special values and missing data
    3. Engineers features
    4. Trains model with out-of-time validation
    5. Evaluates and saves artifacts
    """
    
    # Parameters
    data_path = Parameter(
        'data_path',
        default='../data/credit_risk_data_enhanced.csv',
        help='Path to input data'
    )
    
    train_vintage_end = Parameter(
        'train_vintage_end',
        default='202212',
        help='Last vintage to include in training (YYYYMM)'
    )
    
    test_vintage_start = Parameter(
        'test_vintage_start',
        default='202401',
        help='First vintage to include in test set (YYYYMM)'
    )
    
    use_sample_weights = Parameter(
        'use_sample_weights',
        default=True,
        help='Whether to use sample weights in training'
    )
    
    @card
    @step
    def start(self):
        """
        Start: Load and validate dataset
        
        TODO for candidates:
        - Load data from data_path
        - Validate required columns exist
        - Check for data quality issues
        - Log basic statistics
        """
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} records")
        print(f"✓ Columns: {list(self.df.columns)}")
        print(f"✓ Date range: {self.df['vintage'].min()} to {self.df['vintage'].max()}")
        print(f"✓ Default rate: {self.df['default_12m'].mean():.2%}")
        
        # TODO: Add more validation checks
        # - Check for unexpected values
        # - Validate data types
        # - Check for duplicates
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """
        Preprocessing: Handle special values and split data by vintage
        
        TODO for candidates:
        - Handle special values (99999, -1, 99)
        - Identify and remove leakage features
        - Create train/validation/test splits by vintage
        - Basic feature engineering
        """
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing")
        print("=" * 60)
        
        df = self.df.copy()
        
        # TODO: Handle special values
        # Example:
        # df['fico_score'] = df['fico_score'].replace(99999, np.nan)
        # df['income'] = df['income'].replace(-1, np.nan)
        # df['inquiries_last_6m'] = df['inquiries_last_6m'].replace(99, np.nan)
        
        print("✓ Handling special values...")
        
        # TODO: Identify and remove leakage features
        # CRITICAL: Think about the timeline - what data wouldn't exist at loan application time?
        # Analyze the columns and determine which ones represent post-origination information.
        # leakage_cols = []  # Fill this in after analyzing the data
        # df = df.drop(columns=leakage_cols, errors='ignore')
        # print(f"✓ Removed leakage features: {leakage_cols}")
        
        # TODO: Create vintage-based splits
        self.df_train = df[df['vintage'] <= self.train_vintage_end].copy()
        self.df_test = df[df['vintage'] >= self.test_vintage_start].copy()
        
        print(f"✓ Train set: {len(self.df_train):,} records ({self.df_train['vintage'].min()}-{self.df_train['vintage'].max()})")
        print(f"✓ Test set: {len(self.df_test):,} records ({self.df_test['vintage'].min()}-{self.df_test['vintage'].max()})")
        print(f"✓ Train default rate: {self.df_train['default_12m'].mean():.2%}")
        print(f"✓ Test default rate: {self.df_test['default_12m'].mean():.2%}")
        
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        """
        Feature Engineering: Create modeling features
        
        TODO for candidates:
        - Impute missing values
        - Encode categorical variables
        - Create feature interactions
        - Scale numerical features
        - Create feature matrix (X) and target (y)
        """
        print("\n" + "=" * 60)
        print("STEP 3: Feature Engineering")
        print("=" * 60)
        
        # TODO: Implement feature engineering
        # This is a simplified example - you should do much more!
        
        feature_cols = ['fico_score', 'income', 'debt_to_income', 'loan_amount', 'term', 'apr']
        
        # Simple imputation for demo (you should do better!)
        for col in feature_cols:
            if col in self.df_train.columns:
                median_val = self.df_train[col].median()
                self.df_train[col].fillna(median_val, inplace=True)
                self.df_test[col].fillna(median_val, inplace=True)
        
        self.X_train = self.df_train[feature_cols]
        self.y_train = self.df_train['default_12m']
        self.X_test = self.df_test[feature_cols]
        self.y_test = self.df_test['default_12m']
        
        if self.use_sample_weights:
            self.sample_weights = self.df_train['sample_weight'].values
        else:
            self.sample_weights = None
        
        print(f"✓ Feature matrix shape: {self.X_train.shape}")
        print(f"✓ Features: {list(feature_cols)}")
        
        self.next(self.train)
    
    @step
    def train(self):
        """
        Training: Train model with proper handling of imbalanced data
        
        TODO for candidates:
        - Try multiple algorithms
        - Use sample weights
        - Hyperparameter tuning
        - Cross-validation within training window
        """
        print("\n" + "=" * 60)
        print("STEP 4: Model Training")
        print("=" * 60)
        
        # TODO: Implement better model training
        # This is a simple baseline - you should try multiple models!
        
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(
            self.X_train, 
            self.y_train,
            sample_weight=self.sample_weights
        )
        
        print("✓ Model trained successfully")
        print(f"✓ Model type: {type(self.model).__name__}")
        
        self.next(self.evaluate)
    
    @card
    @step
    def evaluate(self):
        """
        Evaluation: Generate metrics and analyze model performance
        
        TODO for candidates:
        - Calculate AUC-ROC, AUC-PR, KS
        - Create calibration plots
        - Analyze feature importance
        - Performance by vintage
        - Generate confusion matrix
        """
        print("\n" + "=" * 60)
        print("STEP 5: Model Evaluation")
        print("=" * 60)
        
        # Generate predictions
        self.y_pred_proba_train = self.model.predict_proba(self.X_train)[:, 1]
        self.y_pred_proba_test = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        train_auc = roc_auc_score(self.y_train, self.y_pred_proba_train)
        test_auc = roc_auc_score(self.y_test, self.y_pred_proba_test)
        
        self.metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'train_default_rate': self.y_train.mean(),
            'test_default_rate': self.y_test.mean()
        }
        
        print("\n📊 Model Performance:")
        print(f"   Train AUC: {train_auc:.4f}")
        print(f"   Test AUC:  {test_auc:.4f}")
        print(f"   Overfit:   {train_auc - test_auc:.4f}")
        
        # TODO: Add more sophisticated evaluation
        # - KS statistic
        # - AUC-PR for imbalanced data
        # - Calibration plots
        # - Performance by vintage
        # - Feature importance
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End: Save model and artifacts
        
        TODO for candidates:
        - Save model to disk
        - Save feature engineering pipeline
        - Export metrics and plots
        - Create model card
        """
        print("\n" + "=" * 60)
        print("STEP 6: Pipeline Complete!")
        print("=" * 60)
        
        print("\n✅ Final Results:")
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value}")
        
        # TODO: Save model and artifacts
        # import joblib
        # joblib.dump(self.model, 'model.pkl')
        # self.metrics.to_json('metrics.json')
        
        print("\n" + "=" * 60)
        print("Pipeline execution completed successfully!")
        print("=" * 60)


if __name__ == '__main__':
    CreditRiskFlow()

