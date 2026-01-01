#!/usr/bin/env python3
"""Setup script for medical entity recognition project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🏥 Medical Entity Recognition - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        sys.exit(1)
    
    # Install SciSpaCy model (optional)
    print("🔄 Installing SciSpaCy biomedical model (optional)...")
    scispacy_result = run_command(
        f"{pip_cmd} install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz",
        "Installing SciSpaCy biomedical model"
    )
    if not scispacy_result:
        print("⚠️  SciSpaCy model installation failed (optional)")
    
    # Create necessary directories
    directories = [
        "data", "outputs", "checkpoints", "logs", "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run tests
    print("\n🧪 Running tests...")
    test_result = run_command(f"{pip_cmd} install pytest", "Installing pytest")
    if test_result:
        run_command(f"{pip_cmd} run pytest tests/ -v", "Running unit tests")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Generate synthetic dataset:")
    print("   python scripts/generate_data.py")
    print("3. Train the model:")
    print("   python scripts/train.py")
    print("4. Run the demo:")
    print("   streamlit run demo/app.py")
    print("\n⚠️  Remember: This is for research use only, not clinical use!")


if __name__ == "__main__":
    main()
