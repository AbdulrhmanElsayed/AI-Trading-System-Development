"""
Machine Learning Models

Individual ML model implementations for trading signal generation.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from pathlib import Path

# ML Libraries
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.performance_metrics = {}
    
    @abstractmethod
    async def train(self, training_data: pd.DataFrame):
        """Train the model with given data."""
        pass
    
    @abstractmethod
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def get_model_data(self) -> Dict[str, Any]:
        """Get model data for saving."""
        pass
    
    @abstractmethod
    def load_model(self, model_data: Dict[str, Any]):
        """Load model from saved data."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return self.performance_metrics.copy()


class LSTMModel(BaseModel):
    """LSTM Neural Network for time series prediction."""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.sequence_length = config.get('ml.lstm.sequence_length', 60)
        self.hidden_units = config.get('ml.lstm.hidden_units', 50)
        self.dropout_rate = config.get('ml.lstm.dropout_rate', 0.2)
        self.epochs = config.get('ml.training.epochs', 100)
        self.batch_size = config.get('ml.training.batch_size', 64)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")
    
    async def train(self, training_data: pd.DataFrame):
        """Train LSTM model."""
        try:
            self.logger.info("Starting LSTM model training...")
            
            # Prepare data for LSTM
            X, y = await self._prepare_lstm_data(training_data)
            
            if X is None or len(X) < 100:
                raise ValueError("Insufficient data for LSTM training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build model
            self.model = self._build_lstm_model(X_train.shape)
            
            # Train model
            history = await self._train_lstm_async(X_train, y_train, X_test, y_test)
            
            # Calculate performance metrics
            await self._calculate_performance(X_test, y_test)
            
            self.is_trained = True
            self.logger.info("LSTM model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            raise
    
    def _build_lstm_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = models.Sequential([
            layers.LSTM(
                self.hidden_units,
                return_sequences=True,
                input_shape=(input_shape[1], input_shape[2])
            ),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.hidden_units, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(25, activation='relu'),
            layers.Dense(3, activation='softmax')  # Buy, Hold, Sell
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def _train_lstm_async(self, X_train, y_train, X_test, y_test):
        """Train LSTM model asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train in executor to avoid blocking
        history = await loop.run_in_executor(
            None,
            lambda: self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
        )
        
        return history
    
    async def _prepare_lstm_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        # This is a simplified version - you'd implement proper feature engineering
        # For now, return dummy data
        return None, None
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make LSTM prediction."""
        if not self.is_trained or self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            # Reshape for LSTM if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1, 1)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            
            # Convert to trading signal format
            probs = prediction[0]
            direction = np.argmax(probs) - 1  # -1, 0, 1 for Sell, Hold, Buy
            confidence = float(np.max(probs))
            
            return {
                'direction': direction,
                'confidence': confidence,
                'signal_probability': float(probs[2]),  # Buy probability
                'raw_prediction': probs.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error making LSTM prediction: {e}")
            return {'error': str(e)}
    
    async def _calculate_performance(self, X_test, y_test):
        """Calculate model performance metrics."""
        if self.model and len(X_test) > 0:
            predictions = self.model.predict(X_test, verbose=0)
            
            # Convert to class predictions
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            
            self.performance_metrics = {
                'accuracy': float(accuracy),
                'model_type': 'LSTM',
                'last_updated': pd.Timestamp.now().isoformat()
            }
    
    def get_model_data(self) -> Dict[str, Any]:
        """Get model data for saving."""
        if not self.model:
            return {}
        
        # Save model weights and architecture
        return {
            'model_weights': self.model.get_weights(),
            'model_config': self.model.get_config(),
            'scaler': self.scaler,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
    
    def load_model(self, model_data: Dict[str, Any]):
        """Load model from saved data."""
        try:
            if 'model_config' in model_data and 'model_weights' in model_data:
                # Rebuild model from config
                self.model = tf.keras.Model.from_config(model_data['model_config'])
                self.model.set_weights(model_data['model_weights'])
                
                self.scaler = model_data.get('scaler')
                self.performance_metrics = model_data.get('performance_metrics', {})
                self.is_trained = model_data.get('is_trained', False)
                
                self.logger.info("LSTM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")


class XGBoostModel(BaseModel):
    """XGBoost model for trading predictions."""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.n_estimators = config.get('ml.xgboost.n_estimators', 100)
        self.max_depth = config.get('ml.xgboost.max_depth', 6)
        self.learning_rate = config.get('ml.xgboost.learning_rate', 0.1)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
    
    async def train(self, training_data: pd.DataFrame):
        """Train XGBoost model."""
        try:
            self.logger.info("Starting XGBoost model training...")
            
            # Prepare data
            X, y = await self._prepare_xgboost_data(training_data)
            
            if X is None or len(X) < 50:
                raise ValueError("Insufficient data for XGBoost training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42
            )
            
            # Train in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.model.fit,
                X_train_scaled,
                y_train
            )
            
            # Calculate performance
            await self._calculate_xgb_performance(X_test_scaled, y_test)
            
            self.is_trained = True
            self.logger.info("XGBoost model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {e}")
            raise
    
    async def _prepare_xgboost_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for XGBoost training."""
        # This is a placeholder - implement proper feature engineering
        return None, None
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make XGBoost prediction."""
        if not self.is_trained or self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            prediction_class = self.model.predict(features_scaled)[0]
            
            # Convert to trading signal format
            direction = prediction_class - 1  # Convert to -1, 0, 1
            confidence = float(np.max(prediction_proba))
            
            return {
                'direction': int(direction),
                'confidence': confidence,
                'signal_probability': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0.5,
                'raw_prediction': prediction_proba.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error making XGBoost prediction: {e}")
            return {'error': str(e)}
    
    async def _calculate_xgb_performance(self, X_test, y_test):
        """Calculate XGBoost performance metrics."""
        if self.model and len(X_test) > 0:
            predictions = self.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            
            self.performance_metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'model_type': 'XGBoost',
                'last_updated': pd.Timestamp.now().isoformat()
            }
    
    def get_model_data(self) -> Dict[str, Any]:
        """Get model data for saving."""
        return {
            'model': self.model,
            'scaler': self.scaler,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
    
    def load_model(self, model_data: Dict[str, Any]):
        """Load model from saved data."""
        try:
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info("XGBoost model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {e}")


# Placeholder implementations for other models
class TransformerModel(BaseModel):
    """Transformer model placeholder."""
    
    async def train(self, training_data: pd.DataFrame):
        pass
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        return {'direction': 0, 'confidence': 0.5}
    
    def get_model_data(self) -> Dict[str, Any]:
        return {}
    
    def load_model(self, model_data: Dict[str, Any]):
        pass


class RandomForestModel(BaseModel):
    """Random Forest model placeholder."""
    
    async def train(self, training_data: pd.DataFrame):
        pass
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        return {'direction': 0, 'confidence': 0.5}
    
    def get_model_data(self) -> Dict[str, Any]:
        return {}
    
    def load_model(self, model_data: Dict[str, Any]):
        pass


class SVMModel(BaseModel):
    """SVM model placeholder."""
    
    async def train(self, training_data: pd.DataFrame):
        pass
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        return {'direction': 0, 'confidence': 0.5}
    
    def get_model_data(self) -> Dict[str, Any]:
        return {}
    
    def load_model(self, model_data: Dict[str, Any]):
        pass