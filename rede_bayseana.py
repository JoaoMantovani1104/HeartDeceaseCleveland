import pandas as pd
import numpy as np
# Usa DiscreteBayesianNetwork para compatibilidade com versões mais recentes do pgmpy
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


def preparar_dados(df):
    """Realiza a seleção e discretização das 12 variáveis."""

    # Explicacao dos nomes das variaveis estao no documento
    variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'target']
    df_model = df[variables].copy()

    # DISCRETIZACAO
    # Idade (age)
    bins_age = [28, 50, 65, df_model['age'].max() + 1]
    labels_age = ['Young', 'Middle', 'Old']
    df_model['age_binned'] = pd.cut(df_model['age'], bins=bins_age, labels=labels_age, right=False).astype(str)

    # Pressao arterial (trestbps)
    bins_trestbps = [93, 120, 140, df_model['trestbps'].max() + 1]
    labels_trestbps = ['Normal', 'Pre_HTN', 'HTN']
    df_model['trestbps_binned'] = pd.cut(df_model['trestbps'], bins=bins_trestbps, labels=labels_trestbps, right=False).astype(str)

    # Colesterol (chol)
    bins_chol = [125, 200, 240, df_model['chol'].max() + 1]
    labels_chol = ['Normal', 'Borderline', 'High']
    df_model['chol_binned'] = pd.cut(df_model['chol'], bins=bins_chol, labels=labels_chol, right=False).astype(str)

    # Freq cardiaca max (thalac)
    bins_thalach = [71, 140, df_model['thalach'].max() + 1]
    labels_thalach = ['Low_Avg', 'High']
    df_model['thalach_binned'] = pd.cut(df_model['thalach'], bins=bins_thalach, labels=labels_thalach, right=False).astype(str)

    # Depressao do ST (oldpeak)
    bins_oldpeak = [-0.1, 0.5, 2.5, df_model['oldpeak'].max() + 1]
    labels_oldpeak = ['Low', 'Medium', 'High']
    df_model['oldpeak_binned'] = pd.cut(df_model['oldpeak'], bins=bins_oldpeak, labels=labels_oldpeak, right=False).astype(str)

    discretized_vars = ['age_binned', 'trestbps_binned', 'chol_binned', 'thalach_binned', 'oldpeak_binned']
    categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'target']
    df_final = df_model[categorical_vars + discretized_vars].copy()

    for col in categorical_vars:
        df_final[col] = df_final[col].astype(str)

    return df_final


def criar_e_ajustar_modelo(df_final):
    # Arestas DAG 
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
    """Realiza as 4 consultas probabilísticas diferentes (Fase 3)."""
    
    # Adiciona as CPTs calculadas ao modelo para que o motor de inferência funcione
    model.add_cpds(*cpts.values())
    
    # Inicializa o motor de inferência (Eliminação de Variáveis)
    inference = VariableElimination(model)
    
    print("\n" + "="*50)
    print("FASE 3: RESULTADOS DA INFERÊNCIA PROBABILÍSTICA")
    print("="*50)

    # -----------------------------------------------------------
    # Consulta 1: Probabilidade Base (Evidência Simples)
    # Qual a probabilidade de ter a doença dado apenas o risco primário (idade e sexo)?
    # -----------------------------------------------------------
    
    query1 = inference.query(
        variables=['target'],
        evidence={'age_binned': 'Old', 'sex': '1'} # Homem, Idoso
    )
    # P(target=1) é a probabilidade de Doença Presente
    prob1 = query1.values[1] 
    print(f"1. P(Doença=Sim | Idade=Idoso, Sexo=Homem): {prob1:.4f}")

    # -----------------------------------------------------------
    # Consulta 2: Probabilidade Condicional (Evidência Múltipla)
    # Qual a probabilidade de ter a doença dado múltiplos achados ruins?
    # -----------------------------------------------------------
    
    query2 = inference.query(
        variables=['target'],
        evidence={
            'oldpeak_binned': 'High',     # Depressão ST alta
            'thalach_binned': 'Low_Avg', # FC máx baixa
            'ca': '3'                    # 3 vasos principais obstruídos
        }
    )
    prob2 = query2.values[1]
    print(f"2. P(Doença=Sim | oldpeak=High, thalach=Low, ca=3): {prob2:.4f}")
    
    # -----------------------------------------------------------
    # Consulta 3: Inferência Diagnóstica Reversa (Explicação)
    # Qual a probabilidade do paciente ter colesterol alto DADO que a doença foi confirmada?
    # -----------------------------------------------------------
    
    query3 = inference.query(
        variables=['chol_binned'],
        evidence={'target': '1'} # Doença Confirmada
    )
    prob3 = query3.get_value(chol_binned='High')
    print(f"3. P(Colesterol=Alto | Doença=Sim): {prob3:.4f}")

    # -----------------------------------------------------------
    # Consulta 4: Intervenção/Impacto
    # Qual a probabilidade da depressão ST ser Alta DADO que a dor no peito é atípica (cp=2) e NÃO há angina (exang=0)?
    # -----------------------------------------------------------
    
    query4 = inference.query(
        variables=['oldpeak_binned'],
        evidence={'cp': '2', 'exang': '0'}
    )
    prob4 = query4.get_value(oldpeak_binned='High')
    print(f"4. P(oldpeak=High | cp=2, exang=0): {prob4:.4f}")
    
    print("\nAs 4 consultas foram realizadas com sucesso. Use estes valores no seu relatório.")


if __name__ == '__main__':
    # --- PROGRAMA PRINCIPAL ---
    try:
        # Carregar o arquivo CSV (assumindo que está na mesma pasta)
        df_original = pd.read_csv("Heart_disease_cleveland_new.csv")
    except FileNotFoundError:
        print("ERRO: O arquivo 'Heart_disease_cleveland_new.csv' não foi encontrado.")
        exit()

    print("--- INÍCIO DA FASE 2: MODELAGEM E QUANTIFICAÇÃO ---")
    
    # 1. Executar Fase 2.1: Discretização
    df_discretized = preparar_dados(df_original)
    print("Discretização concluída. Variáveis prontas para modelagem.")

    # 2. Executar Fase 2.2: Criação do Modelo (DAG)
    model_fitted = criar_e_ajustar_modelo(df_discretized)
    print("Estrutura do DAG definida e CPTs calculadas via Máxima Verossimilhança.")
    
    # 3. Calcular e Armazenar TODAS as CPTs
    all_cpts = calcular_todas_cpts(model_fitted)

    # Print de todos as CPTs
    for node, cpd in all_cpts.items():
        print(f"\n--- CPT para o nó: {node} ---")
        print(cpd)
        print("-" * 30)

    print(f"Total de {len(all_cpts)} CPTs calculadas e armazenadas.")
    
    # --- INÍCIO DA FASE 3: INFERÊNCIA ---
    executar_inferencias(model_fitted, all_cpts)