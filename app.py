import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# 1. Configuração da Página do Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Engajamento TikToker",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Calculadora de Engajamento de TikToker")
st.markdown(
    "Preveja se o vídeo de um influenciador vai **Viralizar** ou **Flopar** com base no número de hashtags e na categoria do conteúdo."
)

# -----------------------------------------------------------------------------
# 2. Criação / Treinamento do Modelo TensorFlow (Simulação para demonstração)
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo_e_pre_processamento():
    """
    Cria e treina um modelo simples no TensorFlow/Keras com dados sintéticos.
    Utilizamos @st.cache_resource para não retreinar a rede a cada clique do usuário.
    """
    # Mapeamento fixo de categorias para números (One-Hot ou Codificação Numérica)
    categorias = ["Dancinhas", "Humor", "Educacional", "Lifehacks", "Gaming", "Moda/Beleza"]
    
    # Gerando dados sintéticos para treinamento do modelo:
    # Entradas: [categoria_id (0-5), qtd_hashtags]
    # O ideal de hashtags varia por categoria para tornar a previsão dinâmica.
    np.random.seed(42)
    X_train = []
    y_train = []
    
    for _ in range(1000):
        cat_id = np.random.randint(0, len(categorias))
        hashtags = np.random.randint(1, 30)
        
        # Regra fictícia de engajamento baseada no conhecimento de domínio:
        # Entre 3 e 10 hashtags tem mais chances de viralizar
        if 3 <= hashtags <= 10:
            viral = 1 if np.random.rand() > 0.2 else 0
        else:
            viral = 0 if np.random.rand() > 0.2 else 1
            
        X_train.append([cat_id, hashtags])
        y_train.append(viral)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    # Normalização simples das entradas
    # Categoria (0 a 5) e Hashtags (1 a 30)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Construindo o modelo de Rede Neural no TensorFlow/Keras
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Saída binária (0 = Flopado, 1 = Viral)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0, batch_size=16)

    return model, categorias, mean, std

# Carregar o modelo e artefatos
model, lista_categorias, mean_val, std_val = carregar_modelo_e_pre_processamento()

# -----------------------------------------------------------------------------
# 3. Interface Visual do Usuário (Streamlit)
# -----------------------------------------------------------------------------
st.subheader("📋 Parâmetros do Post")

# Seleção da categoria
categoria_selecionada = st.selectbox(
    "Escolha a categoria do vídeo:",
    options=lista_categorias
)

# Entrada da quantidade de hashtags
qtd_hashtags = st.slider(
    "Quantidade de hashtags no post:",
    min_value=1,
    max_value=30,
    value=5
)

st.write("---")

# -----------------------------------------------------------------------------
# 4. Tratamento de Dados e Predição
# -----------------------------------------------------------------------------
if st.button("🚀 Prever Alcance", use_container_width=True):
    # Tratamento dos dados de entrada
    cat_id = lista_categorias.index(categoria_selecionada)
    dados_entrada = np.array([[cat_id, qtd_hashtags]], dtype=np.float32)
    
    # Normalização dos dados com a mesma média e desvio padrão do treino
    dados_normalizados = (dados_entrada - mean_val) / std_val
    
    # Predição com o modelo TensorFlow
    probabilidade_viral = model.predict(dados_normalizados)[0][0]
    
    # -------------------------------------------------------------------------
    # 5. Exibição dos Alertas Visuais
    # -------------------------------------------------------------------------
    st.subheader("📊 Resultado da Previsão")
    
    porcentagem = probabilidade_viral * 100
    st.metric(label="Probabilidade de Viralizar", value=f"{porcentagem:.1f}%")

    if probabilidade_viral >= 0.5:
        st.success(f"🔥 **POST VIRAL!** Esse vídeo tem altas chances de bombar na categoria **{categoria_selecionada}** com {qtd_hashtags} hashtags!")
        st.balloons()
    else:
        st.error(f"📉 **POST FLOPADO!** O modelo indica um engajamento baixo. Tente ajustar a quantidade de hashtags.")