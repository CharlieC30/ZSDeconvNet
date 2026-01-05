#!/usr/bin/env python
"""
Test runner script for ZS-DeconvNet testing suite

This script runs all tests in the tests directory and provides a summary report.
"""

import sys
import unittest
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests(verbosity=2):
    """
    Run all tests in the tests directory
    
    Args:
        verbosity: Level of detail in test output (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        True if all tests passed, False otherwise
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


def run_specific_test(test_module, test_class=None, test_method=None, verbosity=2):
    """
    Run a specific test module, class, or method
    
    Args:
        test_module: Name of the test module (e.g., 'test_patch_processing_2d')
        test_class: Optional name of test class
        test_method: Optional name of test method
        verbosity: Level of detail in test output
        
    Returns:
        True if test(s) passed, False otherwise
    """
    # Build test name
    if test_method and test_class:
        test_name = f"{test_module}.{test_class}.{test_method}"
    elif test_class:
        test_name = f"{test_module}.{test_class}"
    else:
        test_name = test_module
    
    # Load and run test
    suite = unittest.TestLoader().loadTestsFromName(test_name)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ZS-DeconvNet tests')
    parser.add_argument('--module', '-m', type=str, help='Specific test module to run')
    parser.add_argument('--class', '-c', type=str, dest='test_class', help='Specific test class to run')
    parser.add_argument('--method', '-t', type=str, help='Specific test method to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 2 if args.verbose else (0 if args.quiet else 1)
    
    # Run tests
    if args.module:
        success = run_specific_test(args.module, args.test_class, args.method, verbosity)
    else:
        success = run_all_tests(verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
