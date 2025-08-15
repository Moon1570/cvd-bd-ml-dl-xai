"""
Deep Learning Models for CVD Risk Prediction - Multiclass Classification
========================================================================

This module implements state-of-the-art deep learning models for predicting
cardiovascular disease risk levels (LOW, INTERMEDIARY, HIGH) using TensorFlow/Keras.

Features:
- Multiple deep learning architectures (DNN, CNN, LSTM, Transformer)
- SMOTE for class balancing
- Advanced regularization techniques
- Hyperparameter optimization
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.regularizers import l1_l2
    print("✅ TensorFlow imported successfully")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("Installing TensorFlow...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CVDDeepLearningPipeline:
    """
    A comprehensive pipeline for deep learning-based CVD risk prediction.
    """
    
    def __init__(self, data_path=None, X=None, y=None):
        """
        Initialize the pipeline with data.
        
        Args:
            data_path (str): Path to CSV file containing the data
            X (pd.DataFrame): Features dataframe
            y (pd.Series): Target series
        """
        self.data_path = data_path
        self.X = X
        self.y = y
        self.models = {}
        self.results = {}
        self.scaler = None
        self.label_encoder = None
        self.smote = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the CVD dataset."""
        print("🔄 Loading and preprocessing data...")
        
        if self.data_path:
            # Load from CSV
            df = pd.read_csv(self.data_path)
            print(f"Data shape: {df.shape}")
            
            # Basic preprocessing (assuming last column is target)
            self.X = df.iloc[:, :-1]
            self.y = df.iloc[:, -1]
        
        # Handle categorical variables
        if self.X.select_dtypes(include=['object']).shape[1] > 0:
            print("Encoding categorical variables...")
            self.X = pd.get_dummies(self.X, drop_first=True)
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.n_classes = len(self.label_encoder.classes_)
        
        print(f"Number of features: {self.X.shape[1]}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Class distribution: {np.bincount(self.y_encoded)}")
        
        return self.X, self.y_encoded
    
    def train_test_split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print(f"🔄 Splitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=test_size, 
            random_state=random_state, stratify=self.y_encoded
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler."""
        print("🔄 Scaling features...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Scaled training features shape: {self.X_train_scaled.shape}")
        return self.X_train_scaled, self.X_test_scaled
    
    def apply_smote(self, random_state=42):
        """Apply SMOTE for class balancing."""
        print("🔄 Applying SMOTE for class balancing...")
        
        # Check class distribution before SMOTE
        unique, counts = np.unique(self.y_train, return_counts=True)
        print("Class distribution before SMOTE:")
        for cls, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([cls])[0]
            percentage = (count / len(self.y_train)) * 100
            print(f"  {cls} ({class_name}): {count} samples ({percentage:.1f}%)")
        
        # Apply SMOTE
        self.smote = SMOTE(random_state=random_state)
        self.X_train_balanced, self.y_train_balanced = self.smote.fit_resample(
            self.X_train_scaled, self.y_train
        )
        
        # Check class distribution after SMOTE
        unique, counts = np.unique(self.y_train_balanced, return_counts=True)
        print("\\nClass distribution after SMOTE:")
        for cls, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([cls])[0]
            percentage = (count / len(self.y_train_balanced)) * 100
            print(f"  {cls} ({class_name}): {count} samples ({percentage:.1f}%)")
        
        print(f"Balanced training set shape: {self.X_train_balanced.shape}")
        return self.X_train_balanced, self.y_train_balanced
    
    def create_dense_model(self, input_dim, architecture='deep', dropout_rate=0.3):
        """
        Create a Deep Neural Network model.
        
        Args:
            input_dim (int): Number of input features
            architecture (str): 'simple', 'deep', or 'ultra_deep'
            dropout_rate (float): Dropout rate for regularization
        """
        model = keras.Sequential(name=f'DNN_{architecture}')
        
        if architecture == 'simple':
            # Simple architecture
            model.add(layers.Dense(128, activation='relu', input_shape=(input_dim,)))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            
        elif architecture == 'deep':
            # Deep architecture with batch normalization
            model.add(layers.Dense(512, activation='relu', input_shape=(input_dim,)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            
        elif architecture == 'ultra_deep':
            # Ultra deep architecture with residual connections
            inputs = layers.Input(shape=(input_dim,))
            
            # First block
            x = layers.Dense(512, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Second block with residual connection
            x2 = layers.Dense(512, activation='relu')(x)
            x2 = layers.BatchNormalization()(x2)
            x2 = layers.Dropout(dropout_rate)(x2)
            x = layers.Add()([x, x2])  # Residual connection
            
            # Third block
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Fourth block
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Output layer
            outputs = layers.Dense(self.n_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs, name='DNN_ultra_deep')
            
            return model
        
        # Output layer for simple and deep architectures
        model.add(layers.Dense(self.n_classes, activation='softmax'))
        
        return model
    
    def create_attention_model(self, input_dim, num_heads=8, ff_dim=32):
        """
        Create a Transformer-like model with self-attention.
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Reshape for attention (treat features as sequence)
        x = layers.Reshape((input_dim, 1))(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ff_dim
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(1),
        ])
        
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='Attention_Model')
        return model
    
    def create_cnn1d_model(self, input_dim, num_filters=64):
        """
        Create a 1D CNN model for feature extraction.
        """
        model = keras.Sequential(name='CNN1D_Model')
        
        # Reshape input for 1D convolution
        model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
        
        # First conv block
        model.add(layers.Conv1D(filters=num_filters, kernel_size=3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.25))
        
        # Second conv block
        model.add(layers.Conv1D(filters=num_filters*2, kernel_size=3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.25))
        
        # Third conv block
        model.add(layers.Conv1D(filters=num_filters*4, kernel_size=3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.5))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.n_classes, activation='softmax'))
        
        return model
    
    def create_ensemble_model(self, input_dim):
        """
        Create an ensemble model combining different architectures.
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Branch 1: Dense layers
        dense_branch = layers.Dense(256, activation='relu')(inputs)
        dense_branch = layers.BatchNormalization()(dense_branch)
        dense_branch = layers.Dropout(0.3)(dense_branch)
        dense_branch = layers.Dense(128, activation='relu')(dense_branch)
        
        # Branch 2: Attention-like mechanism
        attention_branch = layers.Dense(256, activation='relu')(inputs)
        attention_weights = layers.Dense(256, activation='softmax')(attention_branch)
        attention_branch = layers.Multiply()([attention_branch, attention_weights])
        attention_branch = layers.Dense(128, activation='relu')(attention_branch)
        
        # Combine branches
        combined = layers.Concatenate()([dense_branch, attention_branch])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(combined)
        
        model = Model(inputs=inputs, outputs=outputs, name='Ensemble_Model')
        return model
    
    def compile_model(self, model, optimizer='adam', learning_rate=0.001):
        """Compile a model with specified optimizer and learning rate."""
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            opt = AdamW(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_name, patience=50):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, model_name, epochs=100, batch_size=32, validation_split=0.2):
        """Train a deep learning model."""
        print(f"\\n🚀 Training {model_name}...")
        
        callbacks = self.get_callbacks(model_name)
        
        history = model.fit(
            self.X_train_balanced, self.y_train_balanced,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate a trained model."""
        print(f"\\n📊 Evaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict(self.X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Results for {model_name}:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1
    
    def plot_training_history(self, history, model_name):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_deep_learning_experiments(self):
        """Run comprehensive deep learning experiments."""
        print("🔬 Starting Deep Learning Experiments for CVD Risk Prediction")
        print("="*70)
        
        input_dim = self.X_train_balanced.shape[1]
        
        # Define models to train
        model_configs = [
            ('Simple_DNN', 'simple', 0.001, 32),
            ('Deep_DNN', 'deep', 0.001, 32),
            ('Ultra_Deep_DNN', 'ultra_deep', 0.0005, 64),
            ('CNN1D', 'cnn1d', 0.001, 32),
            ('Attention', 'attention', 0.0005, 32),
            ('Ensemble', 'ensemble', 0.001, 32)
        ]
        
        # Train each model
        for model_name, architecture, lr, batch_size in model_configs:
            print(f"\\n{'='*50}")
            print(f"Training: {model_name}")
            print(f"{'='*50}")
            
            try:
                # Create model
                if architecture == 'simple':
                    model = self.create_dense_model(input_dim, 'simple')
                elif architecture == 'deep':
                    model = self.create_dense_model(input_dim, 'deep')
                elif architecture == 'ultra_deep':
                    model = self.create_dense_model(input_dim, 'ultra_deep')
                elif architecture == 'cnn1d':
                    model = self.create_cnn1d_model(input_dim)
                elif architecture == 'attention':
                    model = self.create_attention_model(input_dim)
                elif architecture == 'ensemble':
                    model = self.create_ensemble_model(input_dim)
                
                # Compile model
                model = self.compile_model(model, learning_rate=lr)
                
                # Print model summary
                print(f"Model: {model_name}")
                print(f"Total parameters: {model.count_params():,}")
                
                # Train model
                history = self.train_model(
                    model, model_name, 
                    epochs=100, 
                    batch_size=batch_size
                )
                
                # Store model
                self.models[model_name] = model
                
                # Evaluate model
                self.evaluate_model(model, model_name)
                
                # Plot training history
                self.plot_training_history(history, model_name)
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Display final results
        self.display_final_results()
    
    def display_final_results(self):
        """Display comprehensive results comparison."""
        print("\\n" + "="*70)
        print("🏆 DEEP LEARNING RESULTS SUMMARY")
        print("="*70)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        # Create results dataframe
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\\nModel Performance Ranking:")
        print("-" * 70)
        for i, row in results_df.iterrows():
            accuracy_pct = row['Accuracy'] * 100
            status = "🏆 EXCELLENT" if accuracy_pct >= 80 else "🔥 VERY GOOD" if accuracy_pct >= 75 else "📈 GOOD" if accuracy_pct >= 70 else "📊 FAIR"
            print(f"{row['Model']:20} | Accuracy: {accuracy_pct:6.2f}% | {status}")
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\\n🥇 Best Model: {best_model['Model']}")
        print(f"🎯 Best Accuracy: {best_model['Accuracy']*100:.2f}%")
        
        # Check if target reached
        if best_model['Accuracy'] >= 0.80:
            print("\\n🎉 CONGRATULATIONS! Achieved 80%+ accuracy with deep learning!")
        else:
            gap = (0.80 - best_model['Accuracy']) * 100
            print(f"\\n📈 Gap to 80%: {gap:.2f}%")
        
        return results_df


def main():
    """Main function to run the deep learning pipeline."""
    print("🧠 CVD Risk Prediction with Deep Learning")
    print("=" * 50)
    
    # Note: This will be called from the notebook with actual data
    print("This module is designed to be imported and used with your CVD dataset.")
    print("Example usage:")
    print("""
    from deep_learning_cvd import CVDDeepLearningPipeline
    
    # Initialize with your data
    pipeline = CVDDeepLearningPipeline(X=your_features, y=your_target)
    
    # Run the complete pipeline
    pipeline.load_and_preprocess_data()
    pipeline.train_test_split_data()
    pipeline.scale_features()
    pipeline.apply_smote()
    pipeline.run_deep_learning_experiments()
    """)


if __name__ == "__main__":
    main()
