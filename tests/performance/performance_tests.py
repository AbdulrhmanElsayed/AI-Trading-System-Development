"""
Performance Testing Framework

Load testing and performance validation for the trading system.
"""

import time
import asyncio
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from tests.base_test import PerformanceTestCase
from tests.fixtures.mock_data import MockMarketData, MockSignalData
from tests.fixtures.test_config import TestConfigManager


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration: float  # seconds
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Throughput metrics
    operations_completed: int = 0
    operations_per_second: float = 0.0
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Error metrics
    errors: int = 0
    error_rate: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_derived_metrics(self, latencies: List[float]):
        """Calculate derived metrics from raw data."""
        if self.duration > 0:
            self.operations_per_second = self.operations_completed / self.duration
        
        if self.operations_completed > 0:
            self.error_rate = self.errors / self.operations_completed
        
        if latencies:
            self.avg_latency_ms = statistics.mean(latencies)
            self.min_latency_ms = min(latencies)
            self.max_latency_ms = max(latencies)
            
            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            if n >= 20:  # Only calculate percentiles if we have enough data
                self.p95_latency_ms = sorted_latencies[int(0.95 * n)]
                self.p99_latency_ms = sorted_latencies[int(0.99 * n)]


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.memory_samples = []
        self.cpu_samples = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        metrics = {}
        
        if self.memory_samples:
            metrics['peak_memory_mb'] = max(self.memory_samples)
            metrics['avg_memory_mb'] = statistics.mean(self.memory_samples)
        
        if self.cpu_samples:
            metrics['avg_cpu_percent'] = statistics.mean(self.cpu_samples)
            metrics['peak_cpu_percent'] = max(self.cpu_samples)
        
        return metrics
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Memory usage in MB
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                
                # CPU usage percentage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class PerformanceBenchmark:
    """Performance benchmarking framework."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.monitor = SystemMonitor()
        self.results = []
    
    async def benchmark_data_processing(self, num_symbols: int = 10, 
                                     days: int = 365) -> PerformanceMetrics:
        """Benchmark data processing performance."""
        
        test_name = f"data_processing_{num_symbols}_symbols_{days}_days"
        print(f"🔄 Running benchmark: {test_name}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = datetime.now()
        latencies = []
        operations_completed = 0
        errors = 0
        
        try:
            # Generate mock market data
            mock_data = MockMarketData()
            symbols = [f"TEST{i:03d}" for i in range(num_symbols)]
            
            for symbol in symbols:
                operation_start = time.time()
                
                try:
                    # Generate OHLCV data
                    data = mock_data.generate_ohlcv(symbol, days=days, freq='1H')
                    
                    # Calculate technical indicators
                    data = mock_data.add_technical_indicators(data)
                    
                    # Simulate data validation
                    assert len(data) > 0
                    assert all(col in data.columns for col in ['open', 'high', 'low', 'close'])
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"Error processing {symbol}: {e}")
                
                operation_end = time.time()
                latency_ms = (operation_end - operation_start) * 1000
                latencies.append(latency_ms)
        
        except Exception as e:
            print(f"Benchmark failed: {e}")
            errors += 1
        
        # Stop monitoring
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        resource_metrics = self.monitor.stop_monitoring()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operations_completed=operations_completed,
            errors=errors,
            peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
            avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0)
        )
        
        metrics.calculate_derived_metrics(latencies)
        
        # Add custom metrics
        metrics.custom_metrics['symbols_processed'] = num_symbols
        metrics.custom_metrics['days_per_symbol'] = days
        metrics.custom_metrics['total_data_points'] = operations_completed * days * 24  # Hourly data
        
        self.results.append(metrics)
        return metrics
    
    async def benchmark_ml_inference(self, batch_sizes: List[int] = None) -> List[PerformanceMetrics]:
        """Benchmark ML model inference performance."""
        
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100, 500]
        
        results = []
        
        for batch_size in batch_sizes:
            test_name = f"ml_inference_batch_{batch_size}"
            print(f"🔄 Running benchmark: {test_name}")
            
            # Start monitoring
            self.monitor.start_monitoring()
            start_time = datetime.now()
            latencies = []
            operations_completed = 0
            errors = 0
            
            try:
                # Generate mock features for ML inference
                mock_data = MockMarketData()
                
                # Simulate multiple inference requests
                num_requests = max(100, 1000 // batch_size)  # Adjust requests based on batch size
                
                for i in range(num_requests):
                    operation_start = time.time()
                    
                    try:
                        # Generate features for batch
                        features = []
                        for j in range(batch_size):
                            feature_data = mock_data.generate_ohlcv("MOCK", days=30, freq='1H')
                            feature_data = mock_data.add_technical_indicators(feature_data)
                            
                            # Simulate feature extraction
                            feature_vector = [
                                feature_data['sma_20'].iloc[-1],
                                feature_data['rsi'].iloc[-1],
                                feature_data['macd_signal'].iloc[-1],
                                feature_data['bb_upper'].iloc[-1],
                                feature_data['volume_sma'].iloc[-1]
                            ]
                            features.append(feature_vector)
                        
                        # Simulate ML inference (mock computation)
                        predictions = []
                        for feature_vector in features:
                            # Simple mock prediction based on features
                            prediction = sum(f if f else 0 for f in feature_vector) % 3  # 0, 1, 2
                            predictions.append(prediction)
                        
                        operations_completed += len(predictions)
                        
                    except Exception as e:
                        errors += 1
                        print(f"Error in ML inference batch {i}: {e}")
                    
                    operation_end = time.time()
                    latency_ms = (operation_end - operation_start) * 1000
                    latencies.append(latency_ms)
            
            except Exception as e:
                print(f"ML benchmark failed: {e}")
                errors += 1
            
            # Stop monitoring
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            resource_metrics = self.monitor.stop_monitoring()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                operations_completed=operations_completed,
                errors=errors,
                peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
                avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0)
            )
            
            metrics.calculate_derived_metrics(latencies)
            
            # Add custom metrics
            metrics.custom_metrics['batch_size'] = batch_size
            metrics.custom_metrics['predictions_per_second'] = operations_completed / duration if duration > 0 else 0
            
            results.append(metrics)
            self.results.append(metrics)
        
        return results
    
    async def benchmark_portfolio_updates(self, num_positions: int = 1000, 
                                        update_frequency: int = 100) -> PerformanceMetrics:
        """Benchmark portfolio update performance."""
        
        test_name = f"portfolio_updates_{num_positions}_positions_{update_frequency}_updates"
        print(f"🔄 Running benchmark: {test_name}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = datetime.now()
        latencies = []
        operations_completed = 0
        errors = 0
        
        try:
            # Initialize mock portfolio
            positions = {}
            mock_data = MockMarketData()
            
            # Create positions
            for i in range(num_positions):
                symbol = f"STOCK{i:04d}"
                positions[symbol] = {
                    'quantity': 100 + (i % 900),  # 100-1000 shares
                    'avg_price': 50 + (i % 200),   # $50-250 per share
                    'market_value': 0,
                    'unrealized_pnl': 0
                }
            
            # Simulate portfolio updates
            for update_round in range(update_frequency):
                operation_start = time.time()
                
                try:
                    # Generate new market prices
                    current_prices = {}
                    for symbol in positions.keys():
                        # Simulate price movement
                        base_price = positions[symbol]['avg_price']
                        price_change = (hash(f"{symbol}_{update_round}") % 21 - 10) / 100  # -10% to +10%
                        current_prices[symbol] = base_price * (1 + price_change)
                    
                    # Update all positions
                    total_portfolio_value = 0
                    for symbol, position in positions.items():
                        current_price = current_prices[symbol]
                        
                        # Update market value and P&L
                        position['market_value'] = position['quantity'] * current_price
                        position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_price'])
                        
                        total_portfolio_value += position['market_value']
                    
                    # Simulate risk calculations
                    portfolio_metrics = {
                        'total_value': total_portfolio_value,
                        'total_pnl': sum(p['unrealized_pnl'] for p in positions.values()),
                        'num_positions': len([p for p in positions.values() if p['quantity'] > 0])
                    }
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"Error in portfolio update {update_round}: {e}")
                
                operation_end = time.time()
                latency_ms = (operation_end - operation_start) * 1000
                latencies.append(latency_ms)
        
        except Exception as e:
            print(f"Portfolio benchmark failed: {e}")
            errors += 1
        
        # Stop monitoring
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        resource_metrics = self.monitor.stop_monitoring()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operations_completed=operations_completed,
            errors=errors,
            peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
            avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0)
        )
        
        metrics.calculate_derived_metrics(latencies)
        
        # Add custom metrics
        metrics.custom_metrics['positions_tracked'] = num_positions
        metrics.custom_metrics['updates_per_second'] = update_frequency / duration if duration > 0 else 0
        metrics.custom_metrics['positions_per_update'] = num_positions
        
        self.results.append(metrics)
        return metrics
    
    async def benchmark_concurrent_trading(self, num_concurrent_users: int = 10, 
                                         operations_per_user: int = 50) -> PerformanceMetrics:
        """Benchmark concurrent trading operations."""
        
        test_name = f"concurrent_trading_{num_concurrent_users}_users_{operations_per_user}_ops"
        print(f"🔄 Running benchmark: {test_name}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = datetime.now()
        all_latencies = []
        total_operations = 0
        total_errors = 0
        
        async def simulate_user_trading(user_id: int):
            """Simulate trading operations for a single user."""
            user_latencies = []
            user_operations = 0
            user_errors = 0
            
            mock_data = MockMarketData()
            mock_signals = MockSignalData()
            
            for op in range(operations_per_user):
                operation_start = time.time()
                
                try:
                    # Simulate getting market data
                    market_data = mock_data.generate_ohlcv("AAPL", days=1, freq='1T')
                    
                    # Simulate signal generation
                    signal = mock_signals.generate_single_signal("AAPL")
                    
                    # Simulate order processing
                    if signal['signal_type'] in ['BUY', 'SELL']:
                        # Mock order validation and submission
                        order_details = {
                            'user_id': user_id,
                            'symbol': signal['symbol'],
                            'side': signal['signal_type'],
                            'quantity': 100,
                            'timestamp': datetime.now()
                        }
                        
                        # Simulate order processing delay
                        await asyncio.sleep(0.001)  # 1ms processing time
                    
                    user_operations += 1
                    
                except Exception as e:
                    user_errors += 1
                
                operation_end = time.time()
                latency_ms = (operation_end - operation_start) * 1000
                user_latencies.append(latency_ms)
            
            return user_latencies, user_operations, user_errors
        
        try:
            # Create concurrent tasks for all users
            tasks = []
            for user_id in range(num_concurrent_users):
                task = simulate_user_trading(user_id)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    total_errors += 1
                    print(f"User simulation failed: {result}")
                else:
                    user_latencies, user_operations, user_errors = result
                    all_latencies.extend(user_latencies)
                    total_operations += user_operations
                    total_errors += user_errors
        
        except Exception as e:
            print(f"Concurrent benchmark failed: {e}")
            total_errors += 1
        
        # Stop monitoring
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        resource_metrics = self.monitor.stop_monitoring()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operations_completed=total_operations,
            errors=total_errors,
            peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
            avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0)
        )
        
        metrics.calculate_derived_metrics(all_latencies)
        
        # Add custom metrics
        metrics.custom_metrics['concurrent_users'] = num_concurrent_users
        metrics.custom_metrics['operations_per_user'] = operations_per_user
        metrics.custom_metrics['total_throughput'] = total_operations / duration if duration > 0 else 0
        
        self.results.append(metrics)
        return metrics
    
    def run_memory_stress_test(self, target_memory_mb: int = 500) -> PerformanceMetrics:
        """Run memory stress test."""
        
        test_name = f"memory_stress_{target_memory_mb}mb"
        print(f"🔄 Running benchmark: {test_name}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = datetime.now()
        operations_completed = 0
        errors = 0
        
        try:
            mock_data = MockMarketData()
            data_chunks = []
            
            # Gradually increase memory usage
            while True:
                try:
                    # Generate large dataset
                    large_data = mock_data.generate_ohlcv("MEMORY_TEST", days=365, freq='1T')
                    large_data = mock_data.add_technical_indicators(large_data)
                    
                    # Keep data in memory
                    data_chunks.append(large_data)
                    operations_completed += 1
                    
                    # Check current memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    if current_memory >= target_memory_mb:
                        print(f"Reached target memory usage: {current_memory:.1f} MB")
                        break
                    
                    # Small delay to allow monitoring
                    time.sleep(0.01)
                    
                except MemoryError:
                    print("Memory limit reached")
                    errors += 1
                    break
                except Exception as e:
                    print(f"Error in memory stress test: {e}")
                    errors += 1
                    break
            
            # Hold memory for a short time to measure peak usage
            time.sleep(1.0)
            
            # Cleanup
            del data_chunks
            gc.collect()
        
        except Exception as e:
            print(f"Memory stress test failed: {e}")
            errors += 1
        
        # Stop monitoring
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        resource_metrics = self.monitor.stop_monitoring()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operations_completed=operations_completed,
            errors=errors,
            peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
            avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0)
        )
        
        # Add custom metrics
        metrics.custom_metrics['target_memory_mb'] = target_memory_mb
        metrics.custom_metrics['peak_memory_achieved'] = resource_metrics.get('peak_memory_mb', 0)
        metrics.custom_metrics['memory_efficiency'] = operations_completed / resource_metrics.get('peak_memory_mb', 1)
        
        self.results.append(metrics)
        return metrics
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        
        if not self.results:
            return "No performance test results available."
        
        report = f"""
# Performance Test Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total tests executed: {len(self.results)}

## Summary

"""
        
        # Aggregate statistics
        total_operations = sum(r.operations_completed for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        avg_throughput = statistics.mean([r.operations_per_second for r in self.results if r.operations_per_second > 0])
        peak_memory = max([r.peak_memory_mb for r in self.results if r.peak_memory_mb > 0])
        
        report += f"""
- **Total Operations**: {total_operations:,}
- **Total Errors**: {total_errors}
- **Average Throughput**: {avg_throughput:.2f} ops/sec
- **Peak Memory Usage**: {peak_memory:.1f} MB

## Individual Test Results

"""
        
        for result in self.results:
            report += f"""
### {result.test_name}

- **Duration**: {result.duration:.2f} seconds
- **Operations**: {result.operations_completed:,}
- **Throughput**: {result.operations_per_second:.2f} ops/sec
- **Peak Memory**: {result.peak_memory_mb:.1f} MB
- **Average CPU**: {result.avg_cpu_percent:.1f}%
- **Error Rate**: {result.error_rate:.2%}

**Latency Statistics:**
- Average: {result.avg_latency_ms:.2f} ms
- Min/Max: {result.min_latency_ms:.2f} / {result.max_latency_ms:.2f} ms
- P95/P99: {result.p95_latency_ms:.2f} / {result.p99_latency_ms:.2f} ms

"""
            
            if result.custom_metrics:
                report += "**Custom Metrics:**\n"
                for key, value in result.custom_metrics.items():
                    report += f"- {key}: {value}\n"
                report += "\n"
        
        return report.strip()


class PerformanceTestRunner:
    """Main performance test runner."""
    
    def __init__(self, config: TestConfigManager):
        self.config = config
        self.benchmark = PerformanceBenchmark(config)
    
    async def run_all_benchmarks(self) -> str:
        """Run all performance benchmarks."""
        
        print("🚀 Starting comprehensive performance tests...")
        
        # Data processing benchmarks
        print("\n📊 Testing data processing performance...")
        await self.benchmark.benchmark_data_processing(num_symbols=5, days=100)
        await self.benchmark.benchmark_data_processing(num_symbols=20, days=365)
        
        # ML inference benchmarks
        print("\n🤖 Testing ML inference performance...")
        await self.benchmark.benchmark_ml_inference([1, 10, 50])
        
        # Portfolio update benchmarks
        print("\n💼 Testing portfolio update performance...")
        await self.benchmark.benchmark_portfolio_updates(num_positions=100, update_frequency=50)
        await self.benchmark.benchmark_portfolio_updates(num_positions=500, update_frequency=100)
        
        # Concurrent trading benchmarks
        print("\n👥 Testing concurrent trading performance...")
        await self.benchmark.benchmark_concurrent_trading(num_concurrent_users=5, operations_per_user=20)
        
        # Memory stress test
        print("\n🧠 Running memory stress test...")
        self.benchmark.run_memory_stress_test(target_memory_mb=200)
        
        print("\n✅ All performance tests completed!")
        
        # Generate comprehensive report
        return self.benchmark.generate_performance_report()


if __name__ == '__main__':
    # Example usage
    import asyncio
    
    async def main():
        config = TestConfigManager()
        runner = PerformanceTestRunner(config)
        
        report = await runner.run_all_benchmarks()
        print(report)
        
        # Save report to file
        with open('performance_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n📄 Performance report saved to 'performance_report.txt'")
    
    asyncio.run(main())