import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. DEFINIÇÃO DA ESTRUTURA DO DAG ---
# Não é necessário carregar os dados discretizados aqui, apenas a estrutura das arestas.

# Definição das arestas (o que foi modelado na Fase 2.1)
edges = [
    # Fatores de Risco --> Condições Crônicas
    ('age_binned', 'trestbps_binned'),
    ('age_binned', 'chol_binned'),
    ('sex', 'chol_binned'),
    ('fbs', 'chol_binned'),

    # Condições Crônicas --> Sintomas/Achados
    ('chol_binned', 'exang'),
    ('trestbps_binned', 'cp'),

    # Sintomas/Achados --> Teste
    ('cp', 'oldpeak_binned'),
    ('exang', 'oldpeak_binned'),

    # Convergência para o Diagnóstico Final (Target)
    ('age_binned', 'target'),
    ('sex', 'target'),
    ('oldpeak_binned', 'target'),
    ('ca', 'target'),
    ('thalach_binned', 'target'),
    ('restecg', 'target'),
]

# Criação do modelo (apenas a estrutura)
model = DiscreteBayesianNetwork(edges)

# --- 2. GERAÇÃO E EXPORTAÇÃO DO DIAGRAMA COM AJUSTES ---

# Converte o modelo pgmpy para o formato networkx para desenho
dag_graph = nx.DiGraph(model.edges())

# Define o layout para as posições dos nós com maior separação (k=1.2)
# O k controla a distância entre os nós no spring_layout
pos = nx.spring_layout(dag_graph, k=1.2, iterations=50, seed=42) 

# Configurações de estilo
plt.figure(figsize=(14, 10)) # Aumenta o tamanho da figura
node_colors = ['lightblue' if node == 'target' else 'lightcoral' for node in dag_graph.nodes()]
node_labels = {node: node.replace('_binned', '') for node in dag_graph.nodes()} # Limpa nomes para o diagrama

# AJUSTE CHAVE: Diminui o tamanho do nó de 1500 para 1000
node_size_adjusted = 1000

# 1. Desenha os nós
nx.draw_networkx_nodes(dag_graph, pos, node_size=node_size_adjusted, node_color=node_colors, alpha=0.9)

# 2. Desenha as arestas
nx.draw_networkx_edges(dag_graph, pos, width=1.5, arrowsize=20, edge_color='gray')

# 3. Desenha os rótulos
nx.draw_networkx_labels(dag_graph, pos, labels=node_labels, font_size=10, font_weight='bold')

# --- EXPORTAÇÃO ---
plt.title("Rede Bayesiana para Diagnóstico de Doença Cardíaca (Visualização Otimizada)", fontsize=14)
plt.axis('off') # Remove os eixos do plot
plt.savefig("DAG_Rede_Bayesiana_Cardiaca_Ajustada.png", format="png", dpi=300)
plt.show()

print("\n[SUCESSO] O diagrama do DAG ajustado foi salvo como 'DAG_Rede_Bayesiana_Cardiaca_Ajustada.png'")