import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tools.process_text import ProcessText

toolProcess = ProcessText()

# Configuração inicial
st.set_page_config(
    page_title="Word2Vec Sentence Clustering",
    page_icon="📊",
    layout="wide"
)

# Inicialização do estado da aplicação
if 'sentences' not in st.session_state:
    st.session_state.sentences = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = []
if 'clusters' not in st.session_state:
    st.session_state.clusters = []
if 'need_clustering' not in st.session_state:
    st.session_state.need_clustering = False

# Função para realizar o clustering usando DBSCAN
@st.cache_data
def cluster_sentences(vectors, eps, min_samples):
    if len(vectors) < min_samples:
        return np.ones(len(vectors)) * -1  # Todos como ruído se houver poucas frases
    
    # Aplicar DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(vectors)
    return clustering.labels_

# Função para reduzir dimensionalidade dos vetores para visualização
@st.cache_data
def reduce_dimensions(vectors, method='pca', n_components=2):
    if len(vectors) < 2:
        return np.zeros((len(vectors), n_components))
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    else:  # t-SNE
        reducer = TSNE(n_components=n_components, perplexity=min(30, max(3, len(vectors)-1)))
    
    return reducer.fit_transform(vectors)


# Interface de Usuário com Streamlit
st.title("Agrupamento Semântico de Frases com Word2Vec")

# Barra lateral para configurações
with st.sidebar:
    st.header("Configurações")
    
    # Seleção do modelo Word2Vec
    model_type = st.radio(
        "Selecione o modelo Word2Vec:",
        ["CBOW", "Skip-gram"]
    )
    
    # Parâmetros do DBSCAN
    st.subheader("Parâmetros de Clustering (DBSCAN)")
    eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1, 
                    help="Distância máxima entre amostras para serem consideradas do mesmo cluster")
    min_samples = st.slider("Mínimo de Amostras", 1, 10, 2, 
                           help="Número mínimo de amostras em um cluster")
    
    # Configurações de Visualização
    st.subheader("Visualização")
    dim_reduction = st.radio(
        "Método de Redução de Dimensionalidade:",
        ["PCA", "t-SNE"]
    )
    
    # Botão para reprocessar tudo
    if st.button("Recalcular Clusters"):
        st.session_state.need_clustering = True

# Área principal - dividida em duas colunas
col1, col2 = st.columns([1, 2])

# Coluna 1: Entrada de frases e lista
with col1:
    st.header("Entrada de Frases")
    
    # Área de texto para inserir frases
    with st.form("sentence_form"):
        new_sentence = st.text_area("Digite frases (uma por linha):", height=150)
        submitted = st.form_submit_button("Adicionar Frases")
    
    if submitted and new_sentence:
        # Divide o texto em linhas e adiciona cada frase
        for sentence in new_sentence.strip().split('\n'):
            if sentence.strip() and sentence not in st.session_state.sentences:
                # Seleciona o modelo apropriado
                model = toolProcess.return_model(model_type)
                
                # Vetoriza a frase
                vector = toolProcess.sentence_to_vector(sentence, model)
                
                if vector is not None:
                    st.session_state.sentences.append(sentence)
                    st.session_state.vectors.append(vector)
                    st.session_state.need_clustering = True
                else:
                    st.warning(f"Não foi possível vetorizar: '{sentence}'")
    
    # Exibe a lista de frases inseridas
    st.subheader("Frases Inseridas")
    if st.session_state.sentences:
        for i, sentence in enumerate(st.session_state.sentences):
            st.text(f"{i+1}. {sentence}")
            
            # Opção para excluir frase
            if st.button(f"Remover", key=f"del_{i}"):
                st.session_state.sentences.pop(i)
                st.session_state.vectors.pop(i)
                st.session_state.need_clustering = True
                st.experimental_rerun()
    else:
        st.info("Nenhuma frase inserida ainda.")

# Coluna 2: Visualização e Clusters
with col2:
    # Realizar clustering se necessário
    if st.session_state.need_clustering and st.session_state.vectors:
        labels = cluster_sentences(
            np.array(st.session_state.vectors), 
            eps=eps, 
            min_samples=min_samples
        )
        st.session_state.clusters = labels
        st.session_state.need_clustering = False
        
    # Visualização dos clusters
    st.header("Visualização dos Clusters")
    
    if st.session_state.vectors:
        # Redução de dimensionalidade para visualização
        method = 'pca' if dim_reduction == 'PCA' else 'tsne'
        coords = reduce_dimensions(np.array(st.session_state.vectors), method=method)
        
        # Preparar dados para o gráfico
        cluster_data = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'sentence': st.session_state.sentences,
            'cluster': st.session_state.clusters
        })
        
        # Criar visualização
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar pontos coloridos por cluster
        scatter = sns.scatterplot(
            data=cluster_data,
            x='x', y='y',
            hue='cluster',
            palette='viridis',
            s=100,
            alpha=0.7,
            ax=ax
        )
        
        # Adicionar rótulos
        for i, row in cluster_data.iterrows():
            plt.text(row['x'] + 0.02, row['y'] + 0.02, str(i+1), 
                    fontsize=9, ha='left', va='bottom')
        
        plt.title("Agrupamento de Frases no Espaço Vetorial")
        plt.xlabel(f"Componente 1 ({dim_reduction})")
        plt.ylabel(f"Componente 2 ({dim_reduction})")
        plt.tight_layout()
        
        # Exibir o gráfico
        st.pyplot(fig)
        
        # Exibir cards para cada cluster
        st.header("Clusters Identificados")
        
        # Identificar clusters únicos (excluindo -1, que é ruído)
        unique_clusters = sorted(set(st.session_state.clusters))
        
        # Adicionar card para pontos de ruído primeiro, se existirem
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            noise_sentences = [s for i, s in enumerate(st.session_state.sentences) 
                              if st.session_state.clusters[i] == -1]
            
            if noise_sentences:
                with st.expander("⚫ Pontos de Ruído", expanded=True):
                    st.write("Frases que não pertencem a nenhum cluster:")
                    for i, sent in enumerate(noise_sentences):
                        st.markdown(f"- {sent}")
        
        # Criar um card para cada cluster
        for cluster_id in unique_clusters:
            # Obter frases do cluster atual
            cluster_sentences = [s for i, s in enumerate(st.session_state.sentences) 
                                if st.session_state.clusters[i] == cluster_id]
            
            # Extrair palavras-chave
            keywords = toolProcess.extract_keywords(cluster_sentences)
            
            # Exibir card do cluster
            with st.expander(f"🔵 Cluster {int(cluster_id)}: {', '.join(keywords[:3])}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("Frases neste cluster:")
                    for i, sent in enumerate(cluster_sentences):
                        st.markdown(f"- {sent}")
                
                with col2:
                    st.subheader("Palavras-chave:")
                    for word in keywords:
                        st.markdown(f"- {word}")
    else:
        st.info("Adicione frases para visualizar clusters.")

# Rodapé com informações adicionais
st.markdown("---")
st.caption("Aplicação de clustering semântico utilizando Word2Vec e DBSCAN")