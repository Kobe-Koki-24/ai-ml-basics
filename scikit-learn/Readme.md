## 1. Create and activate your virtual environment:

### Create venv
python -m venv scikit_learn_venv

### Activate (Windows)
scikit_learn_venv\Scripts\activate

## 2. Install ipykernel in the virtual environment:
pip install ipykernel

## 3. Register the kernel with Jupyter:
python -m ipykernel install --user --name=scikit_learn_kernel --display-name="Scikit Learn Kernel"

## 4. Install your project-specific packages:
pip install scikit-learn pandas numpy matplotlib  # your project requirements