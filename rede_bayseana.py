import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


def preparar_dados(df):
    """Realiza a seleção e discretização das 12 variáveis."""

    variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'target']
    df_model = df[variables].copy()

    # discretização
    bins_age = [28, 50, 65, df_model['age'].max() + 1]
    labels_age = ['Young', 'Middle', 'Old']
    df_model['age_binned'] = pd.cut(df_model['age'], bins=bins_age, labels=labels_age, right=False).astype(str)

    bins_trestbps = [93, 120, 140, df_model['trestbps'].max() + 1]
    labels_trestbps = ['Normal', 'Pre_HTN', 'HTN']
    df_model['trestbps_binned'] = pd.cut(df_model['trestbps'], bins=bins_trestbps, labels=labels_trestbps, right=False).astype(str)

    bins_chol = [125, 200, 240, df_model['chol'].max() + 1]
    labels_chol = ['Normal', 'Borderline', 'High']
    df_model['chol_binned'] = pd.cut(df_model['chol'], bins=bins_chol, labels=labels_chol, right=False).astype(str)

    bins_thalach = [71, 140, df_model['thalach'].max() + 1]
    labels_thalach = ['Low_Avg', 'High']
    df_model['thalach_binned'] = pd.cut(df_model['thalach'], bins=bins_thalach, labels=labels_thalach, right=False).astype(str)

    bins_oldpeak = [-0.1, 0.5, 2.5, df_model['oldpeak'].max() + 1]
    labels_oldpeak = ['Low', 'Medium', 'High']
    df_model['oldpeak_binned'] = pd.cut(df_model['oldpeak'], bins=bins_oldpeak, labels=labels_oldpeak, right=False).astype(str)

    discretized_vars = ['age_binned', 'trestbps_binned', 'chol_binned', 'thalach_binned', 'oldpeak_binned']
    categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'target']
    df_final = df_model[categorical_vars + discretized_vars]

    for col in categorical_vars:
        df_final[col] = df_final[col].astype(str)

    return df_final


def criar_e_ajustar_modelo(df_final):
    edges = [
        ('age_binned', 'trestbps_binned'),
        ('age_binned', 'chol_binned'),
        ('sex', 'chol_binned'),
        ('fbs', 'chol_binned'),
        ('chol_binned', 'exang'),
        ('trestbps_binned', 'cp'),
        ('cp', 'oldpeak_binned'),
        ('exang', 'oldpeak_binned'),
        ('age_binned', 'target'),
        ('sex', 'target'),
        ('oldpeak_binned', 'target'),
        ('ca', 'target'),
        ('thalach_binned', 'target'),
        ('restecg', 'target'),
    ]

    model = DiscreteBayesianNetwork(edges)
    model.fit(df_final, estimator=MaximumLikelihoodEstimator)
    
    return model


def calcular_todas_cpts(model):
    cpts = {}
    for node in model.nodes():
        cpts[node] = model.get_cpds(node)
        
    return cpts


def executar_inferencias(model, cpts):
    """Realiza as 4 consultas probabilísticas diferentes."""
    
    model.add_cpds(*cpts.values())
    
    inference = VariableElimination(model)
    
    print("\n" + "="*50)
    print("FASE 3: RESULTADOS DA INFERÊNCIA PROBABILÍSTICA")
    print("="*50)
    
    query1 = inference.query(
        variables=['target'],
        evidence={'age_binned': 'Old', 'sex': '1'} 
    )
    
    prob1 = query1.values[1] 
    print(f"1. P(Doença=Sim | Idade=Idoso, Sexo=Homem): {prob1:.4f}")

    query2 = inference.query(
        variables=['target'],
        evidence={
            'oldpeak_binned': 'High',     
            'thalach_binned': 'Low_Avg', 
            'ca': '3'                    
        }
    )
    prob2 = query2.values[1]
    print(f"2. P(Doença=Sim | oldpeak=High, thalach=Low, ca=3): {prob2:.4f}")
    
    query3 = inference.query(
        variables=['chol_binned'],
        evidence={'target': '1'} 
    )
    
    states_chol = list(query3.state_names['chol_binned'])
    
    index_chol_high = states_chol.index('High')
    
    prob3 = query3.values[index_chol_high]
    print(f"3. P(Colesterol=Alto | Doença=Sim): {prob3:.4f}")

    
    query4 = inference.query(
        variables=['oldpeak_binned'],
        evidence={'cp': '2', 'exang': '0'}
    )
    states_oldpeak = list(query4.state_names['oldpeak_binned'])
    index_oldpeak_high = states_oldpeak.index('High')
    prob4 = query4.values[index_oldpeak_high]
    print(f"4. P(oldpeak=High | cp=2, exang=0): {prob4:.4f}")
    
    print("\nAs 4 consultas foram realizadas com sucesso. Use estes valores no seu relatório.")


if __name__ == '__main__':
    try:
        df_original = pd.read_csv("Heart_disease_cleveland_new.csv")
    except FileNotFoundError:
        print("ERRO: O arquivo 'Heart_disease_cleveland_new.csv' não foi encontrado.")
        exit()

    print("--- INÍCIO DA FASE 2: MODELAGEM E QUANTIFICAÇÃO ---")
    
    df_discretized = preparar_dados(df_original)
    print("Discretização concluída. Variáveis prontas para modelagem.")

    model_fitted = criar_e_ajustar_modelo(df_discretized)
    print("Estrutura do DAG definida e CPTs calculadas via Máxima Verossimilhança.")
    
    all_cpts = calcular_todas_cpts(model_fitted)
    print(f"Total de {len(all_cpts)} CPTs calculadas e armazenadas.")
    
    executar_inferencias(model_fitted, all_cpts)
