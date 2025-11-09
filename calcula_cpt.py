import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx
import matplotlib.pyplot as plt

df_final = pd.read_csv("Heart_disease_cleveland_discretized.csv")

for col in df_final.columns:
    df_final[col] = df_final[col].astype(str)


# Definição das arestas
edges = [
    # Fatores de Risco --> Condicoes Cronicas
    ('age_binned', 'trestbps_binned'),
    ('age_binned', 'chol_binned'),
    ('sex', 'chol_binned'),
    ('fbs', 'chol_binned'),

    # Condições Cronicas --> Sintomas
    ('chol_binned', 'exang'),
    ('trestbps_binned', 'cp'),

    # Sintomas --> Teste
    ('cp', 'oldpeak_binned'),
    ('exang', 'oldpeak_binned'),

    # Convergencia para o Diagnóstico Final (Target)
    ('age_binned', 'target'),
    ('sex', 'target'),
    ('oldpeak_binned', 'target'),
    ('ca', 'target'),
    ('thalach_binned', 'target'),
    ('restecg', 'target'),
]

model = DiscreteBayesianNetwork(edges)


# O MaximumLikelihoodEstimator calcula as frequências de ocorrência
# de cada estado de um nó dado os estados de seus pais.
model.fit(df_final, estimator=MaximumLikelihoodEstimator)

print("\n--- ESTRUTURA E CÁLCULO DE CPTS CONCLUÍDOS ---")
print(f"Total de Nós: {len(model.nodes())}")
print(f"Total de Arestas: {len(model.edges())}")

# --- EXEMPLO DE CPT (Para o seu Relatório) ---
# Você deve extrair todas as CPTs para a documentação,
# mas vamos imprimir a do nó 'cp' como exemplo:
cpd_cp = model.get_cpds('cp')
print("\nCPT do nó 'cp' (Tipo de Dor no Peito) Condicionada à 'trestbps_binned':")
print(cpd_cp)
