"""
Model Ensemble

Combines predictions from multiple models using various ensemble techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class ModelEnsemble:
    """Ensemble method for combining multiple model predictions."""
    
    def __init__(self, models: Dict[str, Any], config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("ModelEnsemble")
        self.models = models
        
        # Ensemble configuration
        self.ensemble_method = config.get('ml.ensemble.method', 'weighted_average')
        self.voting_threshold = config.get('ml.ensemble.voting_threshold', 0.6)
        
        # Model weights (updated based on performance)
        self.model_weights = {name: 1.0 for name in models.keys()}
        self.weight_decay = config.get('ml.ensemble.weight_decay', 0.1)
        
        # Performance tracking
        self.performance_history = {name: [] for name in models.keys()}
        self.ensemble_predictions = []
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate ensemble prediction from all models."""
        try:
            if not self.models:
                return {'error': 'No models available'}
            
            # Collect predictions from all models
            model_predictions = {}
            valid_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    prediction = await model.predict(features)
                    
                    if 'error' not in prediction:
                        model_predictions[model_name] = prediction
                        valid_predictions[model_name] = prediction
                        
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not valid_predictions:
                return {'error': 'All model predictions failed'}
            
            # Apply ensemble method
            ensemble_result = await self._apply_ensemble_method(valid_predictions)
            
            # Add metadata
            ensemble_result['models_used'] = list(valid_predictions.keys())
            ensemble_result['model_count'] = len(valid_predictions)
            ensemble_result['ensemble_method'] = self.ensemble_method
            ensemble_result['timestamp'] = datetime.now().isoformat()
            
            # Store prediction for performance tracking
            self.ensemble_predictions.append({
                'prediction': ensemble_result,
                'individual_predictions': model_predictions,
                'timestamp': datetime.now()
            })
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    async def _apply_ensemble_method(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply the configured ensemble method."""
        if self.ensemble_method == 'weighted_average':
            return await self._weighted_average_ensemble(predictions)
        elif self.ensemble_method == 'majority_voting':
            return await self._majority_voting_ensemble(predictions)
        elif self.ensemble_method == 'confidence_weighted':
            return await self._confidence_weighted_ensemble(predictions)
        elif self.ensemble_method == 'stacking':
            return await self._stacking_ensemble(predictions)
        else:
            # Default to simple average
            return await self._simple_average_ensemble(predictions)
    
    async def _weighted_average_ensemble(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted average ensemble based on model performance."""
        total_weight = 0
        weighted_direction = 0
        weighted_confidence = 0
        weighted_signal_prob = 0
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.5)
            signal_prob = prediction.get('signal_probability', 0.5)
            
            weighted_direction += direction * weight * confidence
            weighted_confidence += confidence * weight
            weighted_signal_prob += signal_prob * weight
            total_weight += weight
        
        if total_weight > 0:
            final_direction = weighted_direction / total_weight
            final_confidence = weighted_confidence / total_weight
            final_signal_prob = weighted_signal_prob / total_weight
        else:
            final_direction = 0
            final_confidence = 0.5
            final_signal_prob = 0.5
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'signal_probability': final_signal_prob,
            'method': 'weighted_average'
        }
    
    async def _majority_voting_ensemble(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Majority voting ensemble."""
        votes = {'buy': 0, 'sell': 0, 'hold': 0}
        confidences = []
        
        for prediction in predictions.values():
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.5)
            
            if direction > 0.1:
                votes['buy'] += confidence
            elif direction < -0.1:
                votes['sell'] += confidence
            else:
                votes['hold'] += confidence
            
            confidences.append(confidence)
        
        # Determine winner
        winner = max(votes, key=votes.get)
        
        if winner == 'buy':
            final_direction = 1
        elif winner == 'sell':
            final_direction = -1
        else:
            final_direction = 0
        
        # Average confidence
        final_confidence = np.mean(confidences) if confidences else 0.5
        
        # Signal probability based on buy votes
        total_votes = sum(votes.values())
        signal_prob = votes['buy'] / total_votes if total_votes > 0 else 0.5
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'signal_probability': signal_prob,
            'votes': votes,
            'method': 'majority_voting'
        }
    
    async def _confidence_weighted_ensemble(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Confidence-weighted ensemble."""
        total_confidence = 0
        weighted_direction = 0
        weighted_signal_prob = 0
        
        for prediction in predictions.values():
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.5)
            signal_prob = prediction.get('signal_probability', 0.5)
            
            weighted_direction += direction * confidence
            weighted_signal_prob += signal_prob * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            final_direction = weighted_direction / total_confidence
            final_signal_prob = weighted_signal_prob / total_confidence
        else:
            final_direction = 0
            final_signal_prob = 0.5
        
        final_confidence = total_confidence / len(predictions)
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'signal_probability': final_signal_prob,
            'method': 'confidence_weighted'
        }
    
    async def _simple_average_ensemble(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Simple average ensemble."""
        directions = []
        confidences = []
        signal_probs = []
        
        for prediction in predictions.values():
            directions.append(prediction.get('direction', 0))
            confidences.append(prediction.get('confidence', 0.5))
            signal_probs.append(prediction.get('signal_probability', 0.5))
        
        return {
            'direction': np.mean(directions),
            'confidence': np.mean(confidences),
            'signal_probability': np.mean(signal_probs),
            'method': 'simple_average'
        }
    
    async def _stacking_ensemble(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Stacking ensemble (placeholder for meta-learner)."""
        # This would require a trained meta-model
        # For now, fall back to weighted average
        return await self._weighted_average_ensemble(predictions)
    
    async def update_weights(self, performance_data: Dict[str, float]):
        """Update model weights based on recent performance."""
        try:
            for model_name, performance in performance_data.items():
                if model_name in self.model_weights:
                    # Update weight based on performance
                    # Higher performance -> higher weight
                    current_weight = self.model_weights[model_name]
                    new_weight = current_weight * (1 - self.weight_decay) + performance * self.weight_decay
                    
                    # Ensure weight stays within reasonable bounds
                    self.model_weights[model_name] = max(0.1, min(2.0, new_weight))
                    
                    self.logger.info(
                        f"Updated weight for {model_name}: "
                        f"{current_weight:.3f} -> {self.model_weights[model_name]:.3f}"
                    )
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            'model_count': len(self.models),
            'model_weights': self.model_weights.copy(),
            'ensemble_method': self.ensemble_method,
            'predictions_made': len(self.ensemble_predictions),
            'average_confidence': self._calculate_average_confidence(),
            'prediction_distribution': self._get_prediction_distribution()
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence over recent predictions."""
        if not self.ensemble_predictions:
            return 0.0
        
        recent_predictions = self.ensemble_predictions[-100:]  # Last 100 predictions
        confidences = [pred['prediction'].get('confidence', 0) for pred in recent_predictions]
        
        return np.mean(confidences) if confidences else 0.0
    
    def _get_prediction_distribution(self) -> Dict[str, int]:
        """Get distribution of recent predictions."""
        if not self.ensemble_predictions:
            return {'buy': 0, 'sell': 0, 'hold': 0}
        
        distribution = {'buy': 0, 'sell': 0, 'hold': 0}
        recent_predictions = self.ensemble_predictions[-100:]  # Last 100 predictions
        
        for pred in recent_predictions:
            direction = pred['prediction'].get('direction', 0)
            
            if direction > 0.1:
                distribution['buy'] += 1
            elif direction < -0.1:
                distribution['sell'] += 1
            else:
                distribution['hold'] += 1
        
        return distribution