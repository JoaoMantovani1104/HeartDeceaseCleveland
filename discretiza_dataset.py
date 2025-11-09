import pandas as pd
import numpy as np

df = pd.read_csv("Heart_disease_cleveland_new.csv")
variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'target']
df_model = df[variables].copy()


# Idade (age)
bins_age = [28, 50, 65, df_model['age'].max() + 1]
labels_age = ['Young', 'Middle', 'Old']
df_model['age_binned'] = pd.cut(df_model['age'], bins=bins_age, labels=labels_age, right=False)

# Pressao Arterial em Repouso (trestbps) 
bins_trestbps = [93, 120, 140, df_model['trestbps'].max() + 1]
labels_trestbps = ['Normal', 'Pre_HTN', 'HTN']
df_model['trestbps_binned'] = pd.cut(df_model['trestbps'], bins=bins_trestbps, labels=labels_trestbps, right=False)

# 3. Colesterol Serico (chol)
bins_chol = [125, 200, 240, df_model['chol'].max() + 1]
labels_chol = ['Normal', 'Borderline', 'High']
df_model['chol_binned'] = pd.cut(df_model['chol'], bins=bins_chol, labels=labels_chol, right=False)

# 4. Freq. Cardiaca Max (thalach)
bins_thalach = [71, 140, df_model['thalach'].max() + 1]
labels_thalach = ['Low_Avg', 'High']
df_model['thalach_binned'] = pd.cut(df_model['thalach'], bins=bins_thalach, labels=labels_thalach, right=False)

# 5. Depressao do ST (oldpeak)
bins_oldpeak = [-0.1, 0.5, 2.5, df_model['oldpeak'].max() + 1]
labels_oldpeak = ['Low', 'Medium', 'High']
df_model['oldpeak_binned'] = pd.cut(df_model['oldpeak'], bins=bins_oldpeak, labels=labels_oldpeak, right=False)


discretized_vars = ['age_binned', 'trestbps_binned', 'chol_binned', 'thalach_binned', 'oldpeak_binned']
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'target']
df_final = df_model[categorical_vars + discretized_vars].copy()


for col in categorical_vars:
    df_final[col] = df_final[col].astype(str)

df_final.to_csv("Heart_disease_cleveland_discretized.csv", index=False)