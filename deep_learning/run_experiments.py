"""
CVD Deep Learning Experiment Runner
===================================

This script runs deep learning experiments using the preprocessed data
from the CVD analysis notebook for multiclass classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from deep_learning_cvd import CVDDeepLearningPipeline

def run_cvd_deep_learning_multiclass():
    """
    Run deep learning experiments for multiclass CVD classification.
    This function is designed to be called from the notebook.
    """
    print("🧠 Starting CVD Deep Learning Experiments - Multiclass Classification")
    print("="*80)
    
    # This function expects to be called from a notebook environment
    # where the preprocessed data variables are available
    try:
        # Import notebook variables (these should be available in the notebook)
        # We'll use the original 3-class target and the selected features
        print("📊 Accessing preprocessed data from notebook...")
        
        # Access global variables from notebook
        import __main__
        
        # Get the preprocessed features and original target
        X_processed = __main__.X_train_selected  # Shape: (1223, 540) - selected features
        y_original = __main__.y_train  # Original 3-class target
        X_test_processed = __main__.X_test_final  # Test features
        y_test_original = __main__.y_test  # Test target
        
        print(f"✅ Training features shape: {X_processed.shape}")
        print(f"✅ Training target shape: {y_original.shape}")
        print(f"✅ Test features shape: {X_test_processed.shape}")
        print(f"✅ Test target shape: {y_test_original.shape}")
        
        # Convert to pandas if needed
        if isinstance(X_processed, np.ndarray):
            X_features = pd.DataFrame(X_processed)
        else:
            X_features = X_processed
            
        # Combine train and test for proper splitting in the pipeline
        X_combined = np.vstack([X_processed, X_test_processed])
        y_combined = pd.concat([y_original, y_test_original])
        
        X_combined_df = pd.DataFrame(X_combined)
        
        print(f"✅ Combined dataset shape: {X_combined_df.shape}")
        print(f"✅ Combined target shape: {y_combined.shape}")
        
        # Check class distribution
        class_dist = y_combined.value_counts()
        print("\\n📈 Original class distribution:")
        for class_name, count in class_dist.items():
            percentage = (count / len(y_combined)) * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Initialize the deep learning pipeline
        pipeline = CVDDeepLearningPipeline(X=X_combined_df, y=y_combined)
        
        # Load and preprocess data
        X_processed, y_encoded = pipeline.load_and_preprocess_data()
        
        # Split data (80/20 split)
        X_train, X_test, y_train, y_test = pipeline.train_test_split_data(
            test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = pipeline.scale_features()
        
        # Apply SMOTE for class balancing
        X_train_balanced, y_train_balanced = pipeline.apply_smote(random_state=42)
        
        # Run deep learning experiments
        pipeline.run_deep_learning_experiments()
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error accessing notebook data: {e}")
        print("\\n💡 This function should be called from the CVD analysis notebook")
        print("   where the preprocessed data variables are available.")
        return None

def create_simple_demo():
    """Create a simple demo with synthetic data for testing."""
    print("🔬 Creating demo with synthetic data...")
    
    # Create synthetic CVD-like data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic features
    X_demo = np.random.randn(n_samples, n_features)
    
    # Generate synthetic target (3 classes)
    y_demo = np.random.choice(['LOW', 'INTERMEDIARY', 'HIGH'], size=n_samples, 
                             p=[0.4, 0.35, 0.25])
    
    X_demo_df = pd.DataFrame(X_demo, columns=[f'feature_{i}' for i in range(n_features)])
    y_demo_series = pd.Series(y_demo)
    
    print(f"Demo data shape: {X_demo_df.shape}")
    print(f"Demo target shape: {y_demo_series.shape}")
    
    # Initialize pipeline
    pipeline = CVDDeepLearningPipeline(X=X_demo_df, y=y_demo_series)
    
    # Run preprocessing steps
    pipeline.load_and_preprocess_data()
    pipeline.train_test_split_data()
    pipeline.scale_features()
    pipeline.apply_smote()
    
    # Run experiments
    pipeline.run_deep_learning_experiments()
    
    return pipeline

if __name__ == "__main__":
    print("This script is designed to be imported and called from the CVD notebook.")
    print("For demo purposes, creating synthetic data...")
    create_simple_demo()
