"""
Script to run tests with coverage report
"""
import subprocess
import sys

def run_tests_with_coverage():
    """Run pytest with coverage and show the report"""
    try:
        # Run pytest with coverage options
        subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=.", 
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html",
            "tests/"
        ], check=True)
        print("\nCoverage HTML report generated in 'coverage_html' directory")
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests_with_coverage()
