"""
Model Manager

Coordinates multiple machine learning models for trading signal generation.
"""

import asyncio
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger
from src.utils.types import MarketData, TradingSignal, SignalType
from src.ml.models import (
    LSTMModel,
    TransformerModel,
    XGBoostModel,
    RandomForestModel,
    SVMModel
)
from src.ml.feature_engineering import FeatureEngineer
from src.ml.ensemble import ModelEnsemble
from src.ml.rl_agent import RLAgent


class ModelManager:
    """Manages multiple ML models and coordinates predictions."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("ModelManager")
        
        # Models
        self.models = {}
        self.ensemble = None
        self.rl_agent = None
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer(config)
        
        # Model configuration
        self.model_types = config.get('ml.models', ['lstm', 'xgboost'])
        self.model_path = Path("data/models")
        self.model_path.mkdir(exist_ok=True)
        
        # Training configuration
        self.training_config = config.get_section('ml.training')
        self.retrain_interval = config.get('ml.retrain_interval_hours', 24)
        self.last_training = {}
        
        # State
        self.is_training = False
        self.models_loaded = False
    
    async def initialize(self):
        """Initialize all ML models."""
        try:
            self.logger.info("Initializing ML models...")
            
            # Initialize feature engineer
            await self.feature_engineer.initialize()
            
            # Initialize models
            await self._initialize_models()
            
            # Load or train models
            await self._load_or_train_models()
            
            # Initialize ensemble
            self.ensemble = ModelEnsemble(self.models, self.config)
            
            # Initialize RL agent if configured
            if self.config.get('ml.rl.enabled', False):
                self.rl_agent = RLAgent(self.config)
                await self.rl_agent.initialize()
            
            self.models_loaded = True
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize individual models."""
        for model_type in self.model_types:
            try:
                if model_type == 'lstm':
                    self.models['lstm'] = LSTMModel(self.config)
                elif model_type == 'transformer':
                    self.models['transformer'] = TransformerModel(self.config)
                elif model_type == 'xgboost':
                    self.models['xgboost'] = XGBoostModel(self.config)
                elif model_type == 'random_forest':
                    self.models['random_forest'] = RandomForestModel(self.config)
                elif model_type == 'svm':
                    self.models['svm'] = SVMModel(self.config)
                
                self.logger.info(f"Initialized {model_type} model")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {model_type} model: {e}")
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones."""
        for model_name, model in self.models.items():
            try:
                model_file = self.model_path / f"{model_name}_model.pkl"
                
                if model_file.exists():
                    # Load existing model
                    await self._load_model(model_name, model_file)
                    self.logger.info(f"Loaded {model_name} model from {model_file}")
                else:
                    # Train new model
                    self.logger.info(f"No saved model found for {model_name}, training new model...")
                    await self._train_model(model_name)
                
            except Exception as e:
                self.logger.error(f"Error loading/training {model_name}: {e}")
    
    async def _load_model(self, model_name: str, model_file: Path):
        """Load a saved model."""
        loop = asyncio.get_event_loop()
        
        # Load model in executor to avoid blocking
        model_data = await loop.run_in_executor(None, joblib.load, str(model_file))
        
        if hasattr(self.models[model_name], 'load_model'):
            self.models[model_name].load_model(model_data)
    
    async def _train_model(self, model_name: str):
        """Train a specific model."""
        try:
            # Get training data
            training_data = await self._prepare_training_data()
            
            if not training_data or len(training_data) < 100:
                self.logger.warning(f"Insufficient data for training {model_name}")
                return
            
            # Train model
            model = self.models[model_name]
            await model.train(training_data)
            
            # Save trained model
            model_file = self.model_path / f"{model_name}_model.pkl"
            await self._save_model(model_name, model_file)
            
            self.last_training[model_name] = datetime.now()
            self.logger.info(f"Successfully trained and saved {model_name} model")
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
    
    async def _save_model(self, model_name: str, model_file: Path):
        """Save a trained model."""
        model = self.models[model_name]
        
        if hasattr(model, 'get_model_data'):
            loop = asyncio.get_event_loop()
            model_data = model.get_model_data()
            await loop.run_in_executor(None, joblib.dump, model_data, str(model_file))
    
    async def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data for models."""
        try:
            # Get historical data for all symbols
            symbols = self.config.get('data_sources.stocks.symbols', []) + \
                     self.config.get('data_sources.crypto.symbols', [])
            
            all_data = []
            
            for symbol in symbols:
                # This would typically fetch from data storage
                # For now, we'll create a placeholder
                historical_data = await self._get_historical_training_data(symbol)
                if historical_data:
                    all_data.extend(historical_data)
            
            if not all_data:
                return None
            
            # Convert to DataFrame and engineer features
            df = pd.DataFrame(all_data)
            featured_data = await self.feature_engineer.engineer_features(df)
            
            return featured_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None
    
    async def _get_historical_training_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical data for training (placeholder implementation)."""
        # This should integrate with the data storage system
        # For now, return empty list as placeholder
        return []
    
    async def start(self):
        """Start the model manager."""
        if not self.models_loaded:
            await self.initialize()
        
        # Start background tasks
        asyncio.create_task(self._periodic_retraining())
        
        self.logger.info("Model Manager started")
    
    async def stop(self):
        """Stop the model manager."""
        self.logger.info("Stopping Model Manager...")
        
        # Stop RL agent if running
        if self.rl_agent:
            await self.rl_agent.stop()
        
        self.logger.info("Model Manager stopped")
    
    async def generate_signals(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """Generate trading signals using the model ensemble."""
        try:
            if not self.models_loaded or not market_data:
                return []
            
            signals = []
            
            for symbol, data in market_data.items():
                # Prepare features for prediction
                features = await self._prepare_features_for_prediction(symbol, data)
                
                if features is None:
                    continue
                
                # Get ensemble prediction
                prediction = await self.ensemble.predict(features)
                
                # Convert prediction to trading signal
                signal = self._create_trading_signal(symbol, data, prediction)
                
                if signal:
                    signals.append(signal)
                    self.logger.info(
                        f"Generated signal: {symbol} - {signal.signal_type.value} "
                        f"(confidence: {signal.confidence:.2f})"
                    )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _prepare_features_for_prediction(
        self,
        symbol: str,
        market_data: MarketData
    ) -> Optional[np.ndarray]:
        """Prepare features for a single prediction."""
        try:
            # Get recent historical data for context
            # This is a simplified version - in practice, you'd fetch from storage
            recent_data = [market_data]  # Placeholder
            
            # Engineer features
            df = pd.DataFrame([{
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'open': data.open or data.price,
                'high': data.high or data.price,
                'low': data.low or data.price,
                'close': data.close or data.price,
                'volume': data.volume or 0
            } for data in recent_data])
            
            features = await self.feature_engineer.engineer_features(df)
            
            if features is None or len(features) == 0:
                return None
            
            # Return the latest feature vector
            return features.iloc[-1].values
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    def _create_trading_signal(
        self,
        symbol: str,
        market_data: MarketData,
        prediction: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Create a trading signal from model prediction."""
        try:
            # Extract prediction components
            signal_prob = prediction.get('signal_probability', 0.5)
            confidence = prediction.get('confidence', 0.0)
            direction = prediction.get('direction', 0)  # -1, 0, 1
            
            # Determine signal type based on direction and confidence
            min_confidence = self.config.get('ml.min_signal_confidence', 0.6)
            
            if confidence < min_confidence:
                signal_type = SignalType.HOLD
            elif direction > 0.1:
                signal_type = SignalType.BUY
            elif direction < -0.1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=market_data.price,
                timestamp=datetime.now(),
                metadata={
                    'model_prediction': prediction,
                    'market_data': {
                        'price': market_data.price,
                        'volume': market_data.volume,
                        'timestamp': market_data.timestamp.isoformat()
                    }
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating trading signal: {e}")
            return None
    
    async def _periodic_retraining(self):
        """Periodically retrain models."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                for model_name in self.models.keys():
                    last_training = self.last_training.get(model_name)
                    
                    if (not last_training or 
                        datetime.now() - last_training > timedelta(hours=self.retrain_interval)):
                        
                        self.logger.info(f"Starting periodic retraining for {model_name}")
                        await self._train_model(model_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic retraining: {e}")
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        performance = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'get_performance_metrics'):
                    performance[model_name] = model.get_performance_metrics()
                else:
                    performance[model_name] = {'status': 'no_metrics_available'}
            except Exception as e:
                performance[model_name] = {'error': str(e)}
        
        return performance
    
    async def update_model_weights(self, performance_data: Dict[str, float]):
        """Update ensemble model weights based on performance."""
        if self.ensemble:
            await self.ensemble.update_weights(performance_data)