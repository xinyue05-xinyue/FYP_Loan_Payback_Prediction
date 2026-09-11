LOAN PAYBACK PREDICTION SYSTEM
================================

Final Year Project (FYP)


1. SYSTEM OVERVIEW
------------------------------------
This project is a Streamlit-based machine-learning system that predicts whether
a borrower will pay back or default on a loan. It supports four model types:

- Logistic Regression (LR)
- Generalized Additive Model (GAM)
- Explainable Boosting Machine (EBM)
- Bayesian Logistic Regression (BLR)

Predictions can use the original features, Recursive Feature Elimination (RFE),
or Principal Component Analysis (PCA). The application also provides SHAP and
LIME explanations.

The system runs locally and has no authentication page. It does not connect to
an external database. The required datasets, trained model files, transformers,
and evaluation results are included in the data/, models/, and results/ folders.


3. SOFTWARE REQUIREMENTS
------------------------------------
- Python 3.12 (Python 3.12.3 was used for this project)
- pip
- A modern web browser
- Internet access during the initial dependency installation

All Python package versions are listed in requirements.txt.


4. INSTALLATION
------------------------------------
Open a terminal in the project root folder (the folder containing app.py), then
run the commands below.

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt


5. STREAMLIT RUNNING
------------------------------------
The command must be run from the project root folder because the application
loads files using relative paths.

    python -m streamlit run app.py

Streamlit should open the system automatically in a browser. If it does not,
open the Local URL displayed in the terminal.

Press Ctrl+C in the terminal.


6. BASIC USAGE
------------------------------------
1. Select an input representation: Original, RFE, or PCA.
2. Select a model: Logistic Regression, GAM, EBM, or BLR.
3. Adjust the default classification threshold if required.
4. Enter borrower information manually, upload a CSV file, or select a random
   paid-back/default case from the included training data.
5. Run the prediction and review the probability, classification, evaluation,
   and explanation sections.

A sample upload file is provided at:

    data/sample_borrowers(Streamlit).csv


7. JUPYTER NOTEBOOKS
------------------------------------
The development, preprocessing, model training, evaluation, and explanation
notebooks are stored in notebook/. 

The main GAM notebook is: notebook/GAM(FINAL).ipynb

notebook/GAM(Additional).ipynb is create for extra hyperparameter 
tuning approach testing

The notebooks may retrain models and can take longer to complete than running
the Streamlit application. The supplied trained files allow the application to
run without rerunning the notebooks.


8. IMPORTANT PROJECT FILES
------------------------------------
- app.py                  Main Streamlit application
- requirements.txt       Required Python packages and pinned versions
- function/               Application processing and explanation functions
- models/                 Trained models and preprocessing objects
- data/                   Training, testing, and sample input data
- results/                Model evaluation and comparison results
- notebook/               FYP development and analysis notebooks
