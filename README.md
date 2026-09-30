# ⚙️ Industrial Machine Failure Risk Predictor

A lightweight predictive-maintenance prototype: given machine sensor readings,
predict the probability of failure within the next operating window.

**ML task:** Binary classification · **Model:** see `models/` (best of Logistic
Regression / Decision Tree by ROC-AUC) · **Data:** AI4I 2020 Predictive
Maintenance Dataset (UCI, CC BY 4.0).

## Dataset citation
- Matzka, S. (2020). *Explainable artificial intelligence for predictive
  maintenance applications.* Procedia CIRP, 92, 247–252.
  https://doi.org/10.1016/j.procir.2020.05.202
- UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset.
  https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Quickstart
```bash
pip install -r requirements.txt
python train_model.py     # trains, evaluates, saves models/model.pkl
streamlit run app.py      # launches the prototype
