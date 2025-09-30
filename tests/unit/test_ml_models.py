"""
Unit Tests for Machine Learning Module

Tests for ML models, feature engineering, and prediction systems.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tests.base_test import UnitTestCase, AsyncTestCase
from tests.fixtures.mock_data import MockMarketData, quick_ohlcv
from tests.fixtures.test_config import TestConfigManager

# Import modules to test
try:
    from src.ml.model_manager import ModelManager
    from src.ml.feature_engineer import FeatureEngineer
    from src.ml.models.lstm_model import LSTMModel
    from src.ml.models.xgboost_model import XGBoostModel
    from src.ml.ensemble_model import EnsembleModel
    from src.ml.rl_agent import RLAgent
except ImportError as e:
    pytest.skip(f"ML module not available: {e}", allow_module_level=True)


@pytest.mark.unit
class TestFeatureEngineer(UnitTestCase):
    """Test cases for Feature Engineer."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.test_data = quick_ohlcv('AAPL', days=100)  # Need enough data for features
        self.feature_engineer = FeatureEngineer(self.config)
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        assert self.feature_engineer is not None
        assert hasattr(self.feature_engineer, 'config')
        assert hasattr(self.feature_engineer, 'feature_cache')
    
    def test_price_features(self):
        """Test price-based feature generation."""
        features = self.feature_engineer.generate_price_features(self.test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        
        # Should contain returns
        assert 'returns' in features.columns
        assert 'returns_1' in features.columns or 'return_1d' in features.columns
        
        # Should contain price ratios
        price_ratio_features = [col for col in features.columns if 'price_ratio' in col]
        assert len(price_ratio_features) > 0
        
        # Should contain volatility features
        volatility_features = [col for col in features.columns if 'volatility' in col or 'vol' in col]
        assert len(volatility_features) > 0
    
    def test_technical_features(self):
        """Test technical indicator features."""
        features = self.feature_engineer.generate_technical_features(self.test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        
        # Should contain moving averages
        ma_features = [col for col in features.columns if 'sma' in col or 'ema' in col]
        assert len(ma_features) > 0
        
        # Should contain momentum indicators
        momentum_features = [col for col in features.columns if any(
            ind in col for ind in ['rsi', 'macd', 'momentum']
        )]
        assert len(momentum_features) > 0
        
        # Should contain volatility indicators
        vol_features = [col for col in features.columns if any(
            ind in col for ind in ['bb', 'atr', 'volatility']
        )]
        assert len(vol_features) > 0
    
    def test_volume_features(self):
        """Test volume-based features."""
        features = self.feature_engineer.generate_volume_features(self.test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        
        # Should contain volume ratios
        volume_features = [col for col in features.columns if 'volume' in col]
        assert len(volume_features) > 0
        
        # Should contain volume-price features
        vp_features = [col for col in features.columns if any(
            term in col for term in ['vwap', 'volume_price', 'pv']
        )]
        assert len(vp_features) > 0
    
    def test_time_features(self):
        """Test time-based features."""
        features = self.feature_engineer.generate_time_features(self.test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        
        # Should contain time-based features
        time_features = [col for col in features.columns if any(
            term in col for term in ['hour', 'day', 'week', 'month', 'season']
        )]
        assert len(time_features) > 0
    
    def test_lag_features(self):
        """Test lagged features generation."""
        base_features = pd.DataFrame({
            'returns': self.test_data['close'].pct_change(),
            'volume_ratio': self.test_data['volume'] / self.test_data['volume'].rolling(20).mean()
        })
        
        lag_features = self.feature_engineer.generate_lag_features(
            base_features, 
            lags=[1, 2, 3, 5, 10]
        )
        
        assert isinstance(lag_features, pd.DataFrame)
        assert not lag_features.empty
        
        # Should contain lagged versions
        lag_columns = [col for col in lag_features.columns if any(
            f'_lag_{lag}' in col for lag in [1, 2, 3, 5, 10]
        )]
        assert len(lag_columns) > 0
    
    def test_rolling_features(self):
        """Test rolling window features."""
        base_series = self.test_data['close'].pct_change().dropna()
        
        rolling_features = self.feature_engineer.generate_rolling_features(
            base_series,
            windows=[5, 10, 20],
            feature_name='returns'
        )
        
        assert isinstance(rolling_features, pd.DataFrame)
        assert not rolling_features.empty
        
        # Should contain rolling statistics
        expected_features = [
            'returns_mean_5', 'returns_std_5', 'returns_min_5', 'returns_max_5',
            'returns_mean_10', 'returns_std_10', 'returns_min_10', 'returns_max_10',
            'returns_mean_20', 'returns_std_20', 'returns_min_20', 'returns_max_20'
        ]
        
        for feature in expected_features:
            assert any(feature in col for col in rolling_features.columns), f"Missing feature: {feature}"
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Generate comprehensive features
        all_features = self.feature_engineer.generate_all_features(self.test_data)
        
        # Create target variable (next day return)
        target = self.test_data['close'].pct_change().shift(-1).dropna()
        
        # Align features and target
        min_length = min(len(all_features), len(target))
        features_aligned = all_features.iloc[:min_length]
        target_aligned = target.iloc[:min_length]
        
        # Remove features with NaN
        features_clean = features_aligned.dropna()
        target_clean = target_aligned[features_clean.index]
        
        if len(features_clean) > 50 and len(target_clean) > 50:
            selected_features = self.feature_engineer.select_features(
                features_clean, 
                target_clean, 
                max_features=20
            )
            
            assert isinstance(selected_features, list)
            assert len(selected_features) <= 20
            assert all(feature in features_clean.columns for feature in selected_features)
    
    def test_feature_scaling(self):
        """Test feature scaling/normalization."""
        # Generate sample features
        features = pd.DataFrame({
            'feature_1': np.random.normal(100, 50, 100),
            'feature_2': np.random.normal(0.5, 2, 100),
            'feature_3': np.random.uniform(0, 1000, 100)
        })
        
        # Test standard scaling
        scaled_features = self.feature_engineer.scale_features(
            features, 
            method='standard'
        )
        
        assert isinstance(scaled_features, pd.DataFrame)
        assert scaled_features.shape == features.shape
        
        # Standard scaled features should have mean ~0 and std ~1
        for col in scaled_features.columns:
            self.assert_near_equal(scaled_features[col].mean(), 0.0, tolerance=0.1)
            self.assert_near_equal(scaled_features[col].std(), 1.0, tolerance=0.1)
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline."""
        result = self.feature_engineer.engineer_features(
            self.test_data,
            include_technical=True,
            include_volume=True,
            include_time=True,
            include_lags=True,
            max_features=50
        )
        
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'feature_names' in result
        assert 'preprocessing_pipeline' in result
        
        features = result['features']
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert len(features.columns) <= 50
        
        feature_names = result['feature_names']
        assert isinstance(feature_names, list)
        assert len(feature_names) == len(features.columns)


@pytest.mark.unit
class TestLSTMModel(AsyncTestCase):
    """Test cases for LSTM Model."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        
        # Create sample training data
        self.sample_data = quick_ohlcv('AAPL', days=200)
        self.feature_engineer = FeatureEngineer(self.config)
    
    async def test_lstm_model_initialization(self):
        """Test LSTM model initialization."""
        with patch('tensorflow.keras.Sequential'):
            model = LSTMModel(self.config)
            
            assert model is not None
            assert hasattr(model, 'model')
            assert hasattr(model, 'config')
            assert hasattr(model, 'sequence_length')
    
    async def test_prepare_sequences(self):
        """Test sequence preparation for LSTM."""
        # Generate features
        features_result = self.feature_engineer.engineer_features(
            self.sample_data,
            max_features=10
        )
        features = features_result['features'].dropna()
        
        # Create target
        target = self.sample_data['close'].pct_change().shift(-1).dropna()
        
        with patch('tensorflow.keras.Sequential'):
            model = LSTMModel(self.config)
            
            X, y = model.prepare_sequences(features, target, sequence_length=10)
            
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.ndim == 3  # (samples, sequence_length, features)
            assert y.ndim == 1  # (samples,)
            assert X.shape[0] == y.shape[0]  # Same number of samples
            assert X.shape[1] == 10  # Sequence length
            assert X.shape[2] == features.shape[1]  # Number of features
    
    async def test_model_training(self):
        """Test LSTM model training."""
        # Prepare mock training data
        X_train = np.random.random((100, 10, 5))  # 100 samples, 10 timesteps, 5 features
        y_train = np.random.random(100)
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=Mock(history={'loss': [0.1, 0.05]}))
            mock_model.predict = Mock(return_value=np.random.random((20, 1)))
            mock_sequential.return_value = mock_model
            
            model = LSTMModel(self.config)
            await model.build_model(input_shape=(10, 5))
            
            history = await model.train(X_train, y_train)
            
            assert history is not None
            assert 'loss' in history
            mock_model.fit.assert_called_once()
    
    async def test_model_prediction(self):
        """Test LSTM model prediction."""
        X_test = np.random.random((20, 10, 5))
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_predictions = np.random.random((20, 1))
            mock_model.predict = Mock(return_value=mock_predictions)
            mock_sequential.return_value = mock_model
            
            model = LSTMModel(self.config)
            model.model = mock_model
            
            predictions = await model.predict(X_test)
            
            assert isinstance(predictions, np.ndarray)
            assert predictions.shape[0] == X_test.shape[0]
            mock_model.predict.assert_called_once_with(X_test)


@pytest.mark.unit 
class TestXGBoostModel(UnitTestCase):
    """Test cases for XGBoost Model."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
        self.sample_data = quick_ohlcv('AAPL', days=100)
        self.feature_engineer = FeatureEngineer(self.config)
    
    def test_xgboost_initialization(self):
        """Test XGBoost model initialization."""
        with patch('xgboost.XGBRegressor'):
            model = XGBoostModel(self.config)
            
            assert model is not None
            assert hasattr(model, 'model')
            assert hasattr(model, 'config')
    
    def test_model_training(self):
        """Test XGBoost model training."""
        # Prepare training data
        X_train = np.random.random((100, 10))
        y_train = np.random.random(100)
        
        with patch('xgboost.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.predict = Mock(return_value=np.random.random(20))
            mock_xgb.return_value = mock_model
            
            model = XGBoostModel(self.config)
            model.train(X_train, y_train)
            
            mock_model.fit.assert_called_once_with(X_train, y_train)
    
    def test_model_prediction(self):
        """Test XGBoost model prediction."""
        X_test = np.random.random((20, 10))
        expected_predictions = np.random.random(20)
        
        with patch('xgboost.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_model.predict = Mock(return_value=expected_predictions)
            mock_xgb.return_value = mock_model
            
            model = XGBoostModel(self.config)
            model.model = mock_model
            
            predictions = model.predict(X_test)
            
            assert isinstance(predictions, np.ndarray)
            np.testing.assert_array_equal(predictions, expected_predictions)
            mock_model.predict.assert_called_once_with(X_test)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        with patch('xgboost.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_importance = np.array([0.3, 0.2, 0.1, 0.4])
            mock_model.feature_importances_ = mock_importance
            mock_xgb.return_value = mock_model
            
            model = XGBoostModel(self.config)
            model.model = mock_model
            
            feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
            importance = model.get_feature_importance(feature_names)
            
            assert isinstance(importance, dict)
            assert len(importance) == len(feature_names)
            assert all(name in importance for name in feature_names)


@pytest.mark.unit
class TestEnsembleModel(AsyncTestCase):
    """Test cases for Ensemble Model."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        with patch('src.ml.models.lstm_model.LSTMModel') as mock_lstm, \
             patch('src.ml.models.xgboost_model.XGBoostModel') as mock_xgb:
            
            ensemble = EnsembleModel(self.config)
            
            assert ensemble is not None
            assert hasattr(ensemble, 'models')
            assert hasattr(ensemble, 'weights')
    
    async def test_ensemble_training(self):
        """Test ensemble model training."""
        X_train = np.random.random((100, 10))
        y_train = np.random.random(100)
        
        # Mock individual models
        mock_lstm = AsyncMock()
        mock_lstm.train = AsyncMock()
        mock_xgb = Mock()
        mock_xgb.train = Mock()
        
        with patch('src.ml.models.lstm_model.LSTMModel', return_value=mock_lstm), \
             patch('src.ml.models.xgboost_model.XGBoostModel', return_value=mock_xgb):
            
            ensemble = EnsembleModel(self.config)
            await ensemble.train(X_train, y_train)
            
            # Verify all models were trained
            mock_lstm.train.assert_called_once()
            mock_xgb.train.assert_called_once()
    
    async def test_ensemble_prediction(self):
        """Test ensemble prediction with weighted voting."""
        X_test = np.random.random((20, 10))
        
        # Mock predictions from individual models
        lstm_predictions = np.random.random(20)
        xgb_predictions = np.random.random(20)
        
        mock_lstm = AsyncMock()
        mock_lstm.predict = AsyncMock(return_value=lstm_predictions)
        mock_xgb = Mock()
        mock_xgb.predict = Mock(return_value=xgb_predictions)
        
        with patch('src.ml.models.lstm_model.LSTMModel', return_value=mock_lstm), \
             patch('src.ml.models.xgboost_model.XGBoostModel', return_value=mock_xgb):
            
            ensemble = EnsembleModel(self.config)
            ensemble.models = {'lstm': mock_lstm, 'xgboost': mock_xgb}
            ensemble.weights = {'lstm': 0.6, 'xgboost': 0.4}
            
            predictions = await ensemble.predict(X_test)
            
            # Verify weighted average
            expected = lstm_predictions * 0.6 + xgb_predictions * 0.4
            np.testing.assert_array_almost_equal(predictions, expected)
    
    async def test_dynamic_weight_adjustment(self):
        """Test dynamic weight adjustment based on performance."""
        # Mock performance metrics
        performance_metrics = {
            'lstm': {'accuracy': 0.75, 'sharpe_ratio': 1.2},
            'xgboost': {'accuracy': 0.80, 'sharpe_ratio': 1.1}
        }
        
        ensemble = EnsembleModel(self.config)
        ensemble.adjust_weights_based_on_performance(performance_metrics)
        
        # XGBoost should get higher weight due to better accuracy
        assert ensemble.weights['xgboost'] > ensemble.weights['lstm']
        
        # Weights should sum to 1
        total_weight = sum(ensemble.weights.values())
        self.assert_near_equal(total_weight, 1.0, tolerance=0.001)


@pytest.mark.unit
class TestModelManager(AsyncTestCase):
    """Test cases for Model Manager."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        with patch('src.ml.ensemble_model.EnsembleModel'), \
             patch('src.ml.feature_engineer.FeatureEngineer'):
            
            manager = ModelManager(self.config)
            await manager.initialize()
            
            assert manager is not None
            assert hasattr(manager, 'ensemble_model')
            assert hasattr(manager, 'feature_engineer')
    
    async def test_model_training_pipeline(self):
        """Test complete model training pipeline."""
        sample_data = quick_ohlcv('AAPL', days=200)
        
        with patch('src.ml.ensemble_model.EnsembleModel') as mock_ensemble, \
             patch('src.ml.feature_engineer.FeatureEngineer') as mock_fe:
            
            # Mock feature engineering
            mock_fe_instance = Mock()
            mock_features = pd.DataFrame(np.random.random((100, 10)))
            mock_fe_instance.engineer_features.return_value = {
                'features': mock_features,
                'feature_names': [f'feature_{i}' for i in range(10)],
                'preprocessing_pipeline': Mock()
            }
            mock_fe.return_value = mock_fe_instance
            
            # Mock ensemble model
            mock_ensemble_instance = AsyncMock()
            mock_ensemble.return_value = mock_ensemble_instance
            
            manager = ModelManager(self.config)
            await manager.initialize()
            
            result = await manager.train_models(sample_data)
            
            assert result is not None
            mock_fe_instance.engineer_features.assert_called_once()
            mock_ensemble_instance.train.assert_called_once()
    
    async def test_prediction_generation(self):
        """Test prediction generation."""
        current_data = quick_ohlcv('AAPL', days=50)
        
        with patch('src.ml.ensemble_model.EnsembleModel') as mock_ensemble, \
             patch('src.ml.feature_engineer.FeatureEngineer') as mock_fe:
            
            # Mock feature engineering
            mock_fe_instance = Mock()
            mock_features = pd.DataFrame(np.random.random((1, 10)))
            mock_fe_instance.engineer_features.return_value = {
                'features': mock_features,
                'feature_names': [f'feature_{i}' for i in range(10)]
            }
            mock_fe.return_value = mock_fe_instance
            
            # Mock ensemble prediction
            mock_ensemble_instance = AsyncMock()
            mock_prediction = np.array([0.75])  # Prediction probability
            mock_ensemble_instance.predict.return_value = mock_prediction
            mock_ensemble.return_value = mock_ensemble_instance
            
            manager = ModelManager(self.config)
            await manager.initialize()
            
            prediction = await manager.generate_prediction('AAPL', current_data)
            
            assert isinstance(prediction, dict)
            assert 'symbol' in prediction
            assert 'prediction' in prediction
            assert 'confidence' in prediction
            assert 'timestamp' in prediction
    
    async def test_model_evaluation(self):
        """Test model evaluation metrics."""
        # Mock evaluation data
        y_true = np.random.choice([0, 1], size=100)  # Binary classification
        y_pred = np.random.random(100)  # Prediction probabilities
        
        manager = ModelManager(self.config)
        
        metrics = manager.evaluate_model_performance(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        
        # Metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                assert 0 <= value <= 1


@pytest.mark.unit
class TestRLAgent(AsyncTestCase):
    """Test cases for Reinforcement Learning Agent."""
    
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.config = TestConfigManager()
    
    async def test_rl_agent_initialization(self):
        """Test RL Agent initialization."""
        with patch('stable_baselines3.PPO'):
            agent = RLAgent(self.config)
            
            assert agent is not None
            assert hasattr(agent, 'model')
            assert hasattr(agent, 'env')
    
    async def test_environment_setup(self):
        """Test trading environment setup."""
        sample_data = quick_ohlcv('AAPL', days=100)
        
        with patch('stable_baselines3.PPO'), \
             patch('src.ml.trading_env.TradingEnvironment') as mock_env:
            
            agent = RLAgent(self.config)
            agent.setup_environment(sample_data)
            
            mock_env.assert_called_once()
    
    async def test_agent_training(self):
        """Test RL agent training."""
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_model.learn = Mock()
            mock_ppo.return_value = mock_model
            
            agent = RLAgent(self.config)
            agent.model = mock_model
            
            await agent.train(total_timesteps=1000)
            
            mock_model.learn.assert_called_once_with(total_timesteps=1000)
    
    async def test_action_prediction(self):
        """Test action prediction from trained agent."""
        observation = np.random.random(10)  # Mock observation
        
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_action = np.array([1])  # Action: 1 (BUY)
            mock_model.predict = Mock(return_value=(mock_action, None))
            mock_ppo.return_value = mock_model
            
            agent = RLAgent(self.config)
            agent.model = mock_model
            
            action, _ = agent.predict_action(observation)
            
            assert isinstance(action, np.ndarray)
            mock_model.predict.assert_called_once_with(observation)


# Performance and integration tests
@pytest.mark.unit
class TestMLModulePerformance(UnitTestCase):
    """Performance tests for ML module."""
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance with large datasets."""
        large_data = MockMarketData().generate_ohlcv('AAPL', days=365)
        
        config = TestConfigManager()
        feature_engineer = FeatureEngineer(config)
        
        start_time = datetime.now()
        result = feature_engineer.engineer_features(large_data, max_features=20)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert result is not None
        assert isinstance(result['features'], pd.DataFrame)
        # Should complete in reasonable time
        assert processing_time < 10.0  # Less than 10 seconds
    
    def test_prediction_latency(self):
        """Test prediction generation latency."""
        # This would test the time it takes to generate predictions
        # Important for real-time trading applications
        
        config = TestConfigManager()
        sample_features = pd.DataFrame(np.random.random((1, 20)))
        
        with patch('src.ml.ensemble_model.EnsembleModel') as mock_ensemble:
            mock_ensemble_instance = AsyncMock()
            mock_ensemble_instance.predict = AsyncMock(return_value=np.array([0.75]))
            mock_ensemble.return_value = mock_ensemble_instance
            
            start_time = datetime.now()
            
            # Simulate prediction
            prediction = mock_ensemble_instance.predict(sample_features.values)
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            # Prediction should be fast (< 100ms for real-time trading)
            assert latency < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])