#!/bin/bash
set -euo pipefail

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   Conda Environment Setup Script for Drone   ${NC}"
echo -e "${BLUE}==================================================${NC}"

read -r -p "Do you want to continue with the setup? (y/n): " response
if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${RED}Setup aborted.${NC}"
    exit 0
fi

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Error occurred in the last command.${NC}"
        if [ "${1:-}" == "critical" ]; then
            echo -e "${RED}Critical error. Setup cannot continue.${NC}"
            exit 1
        fi
    fi
}

# --- Prerequisite System Dependencies ---
echo -e "\n[INFO] ${YELLOW}Please ensure system dependencies like git, curl, build-essential, cmake, and ninja-build are installed.${NC}"
echo -e "${YELLOW}This script will NOT install them using sudo. If needed, install them manually, e.g.:${NC}"
echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y git curl build-essential cmake ninja-build unixodbc${NC}"
# read -r -p "Press Enter to continue if system dependencies are met, or Ctrl+C to abort."

# --- Conda Environment Setup ---
CONDA_ENV_NAME="drone"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\n[STEP] ${YELLOW}Checking for Conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH. Please install Conda first.${NC}"
    echo -e "${RED}Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions.${NC}"
    exit 1
fi
echo "Conda found: $(command -v conda)"
check_status "critical"

# Function to run commands within the Conda environment
run_in_conda_env() {
    conda run -n "$CONDA_ENV_NAME" --no-capture-output --live-stream "$@"
}

# Subshell activation helper
activate_conda_in_subshell() {
  if [ -n "${CONDA_EXE:-}" ]; then # Use CONDA_EXE if available
    source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
  elif [ -f "$(dirname "$(which conda)")/../etc/profile.d/conda.sh" ]; then # Fallback to finding from `which conda`
    source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
  else
    echo -e "${RED}Critical: Could not source conda.sh for subshell activation. Ensure Conda is properly initialized for shell interaction.${NC}"
    exit 1
  fi
  conda activate "$CONDA_ENV_NAME"
}


echo -e "\n[STEP] ${YELLOW}Creating or updating Conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION...${NC}"
if conda env list | grep -q "^$CONDA_ENV_NAME\s"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
    read -r -p "Do you want to remove and recreate it? (y/n, default: n to proceed with existing): " recreate_response
    if [[ "$recreate_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Removing existing environment '$CONDA_ENV_NAME'..."
        conda env remove -n "$CONDA_ENV_NAME" -y
        check_status "critical"
        echo "Creating new environment '$CONDA_ENV_NAME'..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
        check_status "critical"
    else
        echo "Proceeding with the existing environment '$CONDA_ENV_NAME'."
    fi
else
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    check_status "critical"
fi


echo -e "\n[STEP] ${YELLOW}Installing dependencies from requirements.txt in $CONDA_ENV_NAME...${NC}"
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    run_in_conda_env conda install --file "$SCRIPT_DIR/requirements.txt" -y
    check_status "critical"
else
    echo -e "${YELLOW}requirements.txt not found in $SCRIPT_DIR. Skipping this step.${NC}"
fi

echo -e "\n[STEP] ${YELLOW}Installing PyTorch with CUDA 11.8 via pip in $CONDA_ENV_NAME...${NC}"
run_in_conda_env pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
check_status "critical"

echo -e "\n[VERIFY] ${YELLOW}PyTorch CUDA availability immediately after installation...${NC}"
run_in_conda_env python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version via PyTorch: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
check_status "critical"


# --- Xformers Installation ---
XFORMERS_DIR="$SCRIPT_DIR/xformers_conda"
echo -e "\n[STEP] ${YELLOW}Installing Xformers from source (maludwig/xformers)...${NC}"
(
  activate_conda_in_subshell

  if [ -d "$XFORMERS_DIR" ]; then
      echo "Removing old Xformers directory: $XFORMERS_DIR"
      rm -rf "$XFORMERS_DIR"
  fi

  echo "Cloning Xformers (maludwig/xformers)..."
  git clone https://github.com/maludwig/xformers.git "$XFORMERS_DIR"
  cd "$XFORMERS_DIR"
  echo "Initializing Xformers submodules..."
  git submodule update --init --recursive
  # The original script had a pip install -r requirements.txt here for this xformers fork
  if [ -f "requirements.txt" ]; then
    echo "Installing Xformers dependencies from requirements.txt..."
    pip install -r requirements.txt
  else
    echo "No requirements.txt found for Xformers, skipping pip install -r requirements.txt."
  fi
  echo "Building and installing Xformers..."
  pip install -v .
)
check_status "critical"

echo -e "\n${GREEN}All installation steps completed! Proceeding to verification...${NC}"

# --- Verification ---
echo -e "\n${BLUE}==================================================${NC}"
echo -e "${BLUE}   Verifying installations in $CONDA_ENV_NAME   ${NC}"
echo -e "${BLUE}==================================================${NC}"

echo -e "\n[VERIFY] ${YELLOW}PyTorch...${NC}"
run_in_conda_env python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version via PyTorch: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
check_status

echo -e "\n[VERIFY] ${YELLOW}Triton...${NC}"
run_in_conda_env python -c "import triton; import triton.language as tl; print(f'Triton imported successfully. Version: {getattr(triton, \"__version__\", \"unknown\")}')"
check_status

echo -e "\n[VERIFY] ${YELLOW}Xformers...${NC}"
run_in_conda_env python -c "import xformers; import xformers.ops as xops; print(f'Xformers imported successfully. Version: {getattr(xformers, \"__version__\", \"unknown\")}')"
check_status

echo -e "\n${GREEN}========================================================${NC}"
echo -e "${GREEN}   Setup and Verification Completed Successfully!   ${NC}"
echo -e "${GREEN}========================================================${NC}"
echo -e "${YELLOW}To activate the Conda environment in your terminal, run:${NC}"
echo -e "${GREEN}conda activate $CONDA_ENV_NAME${NC}"
echo -e "${YELLOW}The cloned repositories are located in: $SCRIPT_DIR ${NC}"