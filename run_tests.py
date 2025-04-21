"""
Script to run tests with coverage report using the coverage module directly
"""
import os
import sys

def run_tests_with_coverage():
    """Run tests with coverage and generate reports"""
    try:
        print("Running tests with coverage...")
        
        # Run the coverage command to start the coverage measurement
        os.system("coverage run -m pytest tests/")
        
        # Generate the coverage reports
        os.system("coverage report -m")  # Terminal report with missing lines
        os.system("coverage html")       # HTML report
        
        print("\nCoverage HTML report generated in 'htmlcov' directory")
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests_with_coverage()
