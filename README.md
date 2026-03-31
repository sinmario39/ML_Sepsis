# Sepsis Analysis and Triage
## Description
Patients admitted in the Intensive Care Unit (ICU) usually present complex clinical pictures, often featuring various kinds of physical instabilities.
The objective of this project is not to formulate a complete clinical diagnosis, but rather to identify initial
physiological patterns associated with sepsis or different forms of non-septic instability through Machine Learning techniques.

## Dataset
This project uses the ***"PhysioNet/Computing in Cardiology Challenge 2019 – Sepsis Prediction"*** dataset.
The dataset was obtained from the Kaggle mirror for convenience.

Original source:
PhysioNet – https://physionet.org/content/challenge-2019/

Mirror source:
Kaggle - https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

All rights and licensing conditions belong to PhysioNet.
Users must comply with the original dataset license.

## Requirements
- Python 3.14
- pandas
- tqdm
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter (optional, for notebook testing, install with `pip install jupyter`)

Install dependencies with:  
`pip install -r requirements.txt`  
Or  
`pip install numpy tqdm pandas matplotlib seaborn scikit-learn`  

## Execution
1. Download the PhysioNet dataset and place all `.psv` files inside the `data/` folder.
2. Run the exploratory analysis script to generate the processed dataset snapshot:  
   `python src/eda.py`  
   Or run it from the project root with: `python main.py --eda`  
3. Run the training pipeline from the project root using:  
    - Level 1 (Sepsis vs Non-Sepsis):  
      `python main.py --level 1`  
    - Level 2 (Non-Sepsis classification):  
      `python main.py --level 2`  
   - Full pipeline (both levels):  
      `python main.py --level all`  
   - Both EDA and full pipeline:  
      `python main.py --eda --level all`  

## Citation
Reyna et al., "Early Prediction of Sepsis from Clinical Data:
The PhysioNet/Computing in Cardiology Challenge 2019"
