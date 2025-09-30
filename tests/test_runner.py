"""
Test Runner and Validation Suite

Comprehensive test execution and validation framework.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures.test_config import TestConfigManager


class TestRunner:
    """Comprehensive test runner for the trading system."""
    
    def __init__(self):
        self.config = TestConfigManager()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_unit_tests(self, verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
        """Run all unit tests."""
        print("🧪 Running Unit Tests...")
        
        pytest_args = [
            'tests/unit/',
            '-m', 'unit',
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        if coverage:
            pytest_args.extend(['--cov=src', '--cov-report=html'])
        
        # Run tests
        result = pytest.main(pytest_args)
        
        self.test_results['unit_tests'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['unit_tests']
    
    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        pytest_args = [
            'tests/integration/',
            '-m', 'integration',
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        self.test_results['integration_tests'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['integration_tests']
    
    def run_backtesting_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run backtesting tests."""
        print("📈 Running Backtesting Tests...")
        
        pytest_args = [
            'tests/backtesting/',
            '-m', 'backtesting',
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        self.test_results['backtesting_tests'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['backtesting_tests']
    
    def run_performance_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run performance tests."""
        print("⚡ Running Performance Tests...")
        
        pytest_args = [
            'tests/performance/',
            '-m', 'performance',
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        self.test_results['performance_tests'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['performance_tests']
    
    def run_simulation_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run simulation tests."""
        print("🎮 Running Simulation Tests...")
        
        pytest_args = [
            'tests/simulation/',
            '-m', 'simulation', 
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        self.test_results['simulation_tests'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['simulation_tests']
    
    def run_framework_integration_test(self, verbose: bool = True) -> Dict[str, Any]:
        """Run framework integration test."""
        print("🔗 Running Framework Integration Test...")
        
        pytest_args = [
            'tests/test_framework_integration.py',
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        self.test_results['framework_integration'] = {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results['framework_integration']
    
    def run_all_tests(self, verbose: bool = True, coverage: bool = False,
                     skip_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        self.start_time = datetime.now()
        
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Configure pytest markers
        pytest_args = []
        if skip_slow:
            pytest_args.extend(['-m', 'not slow'])
        
        try:
            # Run test suites
            unit_result = self.run_unit_tests(verbose, coverage)
            integration_result = self.run_integration_tests(verbose)
            backtesting_result = self.run_backtesting_tests(verbose)
            simulation_result = self.run_simulation_tests(verbose)
            
            # Run performance tests only in full mode
            if not skip_slow:
                performance_result = self.run_performance_tests(verbose)
            else:
                performance_result = {'status': 'skipped', 'message': 'Performance tests skipped in fast mode'}
            
            # Run framework integration test
            framework_result = self.run_framework_integration_test(verbose)
            
            self.end_time = datetime.now()
            
            # Generate summary
            summary = self.generate_test_summary()
            
            print("\\n" + "=" * 60)
            print("📊 Test Summary:")
            print(f"Total Duration: {summary['total_duration']:.2f} seconds")
            print(f"Unit Tests: {unit_result['status'].upper()}")
            print(f"Integration Tests: {integration_result['status'].upper()}")
            print(f"Backtesting Tests: {backtesting_result['status'].upper()}")
            print(f"Simulation Tests: {simulation_result['status'].upper()}")
            print(f"Performance Tests: {performance_result['status'].upper()}")
            print(f"Framework Integration: {framework_result['status'].upper()}")
            print(f"Overall Status: {summary['overall_status'].upper()}")
            
            return summary
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    def run_specific_test_file(self, test_file: str, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific test file."""
        print(f"🎯 Running Specific Test: {test_file}")
        
        if not os.path.exists(test_file):
            return {'status': 'error', 'message': f'Test file not found: {test_file}'}
        
        pytest_args = [test_file, '--tb=short']
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        return {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'file': test_file,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_tests_by_marker(self, marker: str, verbose: bool = True) -> Dict[str, Any]:
        """Run tests with specific marker."""
        print(f"🏷️ Running Tests with Marker: {marker}")
        
        pytest_args = [
            'tests/',
            '-m', marker,
            '--tb=short'
        ]
        
        if verbose:
            pytest_args.append('-v')
        
        result = pytest.main(pytest_args)
        
        return {
            'status': 'passed' if result == 0 else 'failed',
            'exit_code': result,
            'marker': marker,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_duration = 0
        if self.start_time and self.end_time:
            total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall status
        all_passed = all(
            result.get('status') == 'passed' 
            for result in self.test_results.values()
        )
        
        overall_status = 'passed' if all_passed else 'failed'
        
        return {
            'overall_status': overall_status,
            'total_duration': total_duration,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'test_results': self.test_results,
            'summary': {
                'total_suites': len(self.test_results),
                'passed_suites': len([r for r in self.test_results.values() if r.get('status') == 'passed']),
                'failed_suites': len([r for r in self.test_results.values() if r.get('status') == 'failed'])
            }
        }
    
    def save_test_report(self, filename: str = None) -> str:
        """Save test results to JSON report."""
        if filename is None:
            filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            },
            **self.generate_test_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Test report saved to: {filename}")
        return filename
    
    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate test environment setup."""
        validation_results = {}
        
        # Check required directories
        required_dirs = ['tests/unit', 'tests/integration', 'tests/backtesting', 'tests/performance']
        for dir_path in required_dirs:
            validation_results[f'directory_{dir_path}'] = os.path.exists(dir_path)
        
        # Check required files
        required_files = ['pytest.ini', 'tests/conftest.py']
        for file_path in required_files:
            validation_results[f'file_{file_path}'] = os.path.exists(file_path)
        
        # Check test configuration
        try:
            config = TestConfigManager()
            validation_results['test_config'] = True
            validation_results['test_config_details'] = {
                'database_url': config.get_database_url(),
                'fast_mode': config.is_fast_mode(),
                'mock_apis': config.should_mock_apis()
            }
        except Exception as e:
            validation_results['test_config'] = False
            validation_results['test_config_error'] = str(e)
        
        # Overall validation status
        all_valid = all(
            result is True for key, result in validation_results.items()
            if not key.endswith('_details') and not key.endswith('_error')
        )
        
        validation_results['overall_valid'] = all_valid
        
        return validation_results


class ContinuousIntegrationRunner:
    """CI/CD focused test runner."""
    
    def __init__(self):
        self.test_runner = TestRunner()
    
    def run_ci_pipeline(self, coverage_threshold: float = 80.0) -> Dict[str, Any]:
        """Run CI/CD pipeline tests."""
        print("🔄 Running CI/CD Pipeline Tests")
        
        # Validate environment first
        validation = self.test_runner.validate_test_environment()
        if not validation.get('overall_valid', False):
            return {
                'status': 'failed',
                'reason': 'Environment validation failed',
                'validation_results': validation
            }
        
        # Run fast tests only for CI
        config = TestConfigManager()
        config.set_environment('unit')  # Fast mode
        
        # Run tests
        results = self.test_runner.run_all_tests(
            verbose=False,
            coverage=True,
            skip_slow=True
        )
        
        # Check coverage if tests passed
        if results.get('overall_status') == 'passed':
            # In a real implementation, you'd check coverage results
            # coverage_ok = self.check_coverage_threshold(coverage_threshold)
            # results['coverage_passed'] = coverage_ok
            pass
        
        return results
    
    def run_nightly_tests(self) -> Dict[str, Any]:
        """Run comprehensive nightly test suite."""
        print("🌙 Running Nightly Test Suite")
        
        config = TestConfigManager()
        config.set_environment('performance')  # Full test mode
        
        # Run all tests including slow ones
        results = self.test_runner.run_all_tests(
            verbose=True,
            coverage=True,
            skip_slow=False
        )
        
        # Save detailed report
        report_file = self.test_runner.save_test_report(
            f"nightly_report_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        results['report_file'] = report_file
        
        return results


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='Trading System Test Runner')
    
    parser.add_argument('--suite', choices=['unit', 'integration', 'backtesting', 'performance', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slow tests')
    parser.add_argument('--marker', help='Run tests with specific marker')
    parser.add_argument('--file', help='Run specific test file')
    parser.add_argument('--ci', action='store_true', help='Run CI pipeline')
    parser.add_argument('--nightly', action='store_true', help='Run nightly tests')
    parser.add_argument('--validate', action='store_true', help='Validate test environment')
    parser.add_argument('--report', help='Save report to specific file')
    
    args = parser.parse_args()
    
    if args.validate:
        # Validate test environment
        runner = TestRunner()
        validation = runner.validate_test_environment()
        
        print("🔍 Test Environment Validation:")
        for key, value in validation.items():
            if key != 'test_config_details':
                status = "✅" if value else "❌"
                print(f"{status} {key}: {value}")
        
        if validation.get('overall_valid'):
            print("✅ Test environment is valid!")
        else:
            print("❌ Test environment validation failed!")
            sys.exit(1)
        
        return
    
    if args.ci:
        # Run CI pipeline
        ci_runner = ContinuousIntegrationRunner()
        results = ci_runner.run_ci_pipeline()
        
        if results.get('status') == 'passed':
            print("✅ CI Pipeline PASSED")
            sys.exit(0)
        else:
            print("❌ CI Pipeline FAILED")
            sys.exit(1)
    
    if args.nightly:
        # Run nightly tests
        ci_runner = ContinuousIntegrationRunner()
        results = ci_runner.run_nightly_tests()
        
        print(f"🌙 Nightly tests completed: {results.get('overall_status', 'unknown').upper()}")
        return
    
    # Regular test execution
    runner = TestRunner()
    
    if args.file:
        # Run specific file
        results = runner.run_specific_test_file(args.file, args.verbose)
        
    elif args.marker:
        # Run tests with specific marker
        results = runner.run_tests_by_marker(args.marker, args.verbose)
        
    elif args.suite == 'unit':
        results = runner.run_unit_tests(args.verbose, args.coverage)
        
    elif args.suite == 'integration':
        results = runner.run_integration_tests(args.verbose)
        
    elif args.suite == 'backtesting':
        results = runner.run_backtesting_tests(args.verbose)
        
    elif args.suite == 'performance':
        results = runner.run_performance_tests(args.verbose)
        
    else:
        # Run all tests
        results = runner.run_all_tests(args.verbose, args.coverage, args.skip_slow)
    
    # Save report if requested
    if args.report or args.suite == 'all':
        runner.save_test_report(args.report)
    
    # Exit with appropriate code
    if results.get('overall_status') == 'passed' or results.get('status') == 'passed':
        print("✅ Tests PASSED")
        sys.exit(0)
    else:
        print("❌ Tests FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()