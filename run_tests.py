"""
Test Runner for Financial Anomaly Detection System

This script runs all tests and provides a comprehensive test report.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)

def main():
    """Main test runner."""
    print("🧪 Financial Anomaly Detection System - Test Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test results
    test_results = []
    
    # Test 1: Check Python version
    print(f"\n🐍 Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
    
    # Test 2: Check if required packages are installed
    print("\n📦 Checking required packages...")
    required_packages = [
        "numpy", "pandas", "scikit-learn", "torch", "matplotlib", 
        "seaborn", "plotly", "streamlit", "yfinance", "ccxt"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "scikit-learn":
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    # Test 3: Run unit tests
    success, stdout, stderr = run_command(
        "python tests/test_pipeline.py",
        "Unit Tests"
    )
    test_results.append(("Unit Tests", success))
    
    # Test 4: Run simple example
    success, stdout, stderr = run_command(
        "python examples/simple_example.py",
        "Simple Example"
    )
    test_results.append(("Simple Example", success))
    
    # Test 5: Check if dashboard can be imported
    print(f"\n{'='*60}")
    print("Testing Dashboard Import")
    print(f"{'='*60}")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard'))
        import app
        print("✅ Dashboard can be imported successfully")
        test_results.append(("Dashboard Import", True))
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        test_results.append(("Dashboard Import", False))
    
    # Test 6: Check if all modules can be imported
    print(f"\n{'='*60}")
    print("Testing Module Imports")
    print(f"{'='*60}")
    
    modules_to_test = [
        "data.collectors.yahoo_finance_collector",
        "data.collectors.crypto_collector", 
        "data.collectors.fx_collector",
        "data.processors.feature_engineer",
        "models.isolation_forest",
        "models.autoencoder",
        "models.gnn_anomaly",
        "utils.model_evaluator"
    ]
    
    import_success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            import_success = False
    
    test_results.append(("Module Imports", import_success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the dashboard: streamlit run dashboard/app.py")
        print("2. Try the simple example: python examples/simple_example.py")
        print("3. Run the complete analysis: python examples/run_analysis.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
