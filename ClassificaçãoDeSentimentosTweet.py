import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay,
)


#coleta de dados 
dataset = pd.read_csv("tweetsSentiment.csv")



#limpeza dos dados


def limparDados(texto):
    texto = re.sub(r'[^\w\s]', '', texto) # remover pontuações e caracteres especiais 
    texto = str(texto).lower() # transformar tudo em minúsculo
    return texto


dataset["tweets_limpos"] = dataset["tweet"].apply(limparDados)


def separarTexto(texto):
    return texto.split()

dataset["tweet_limpo_separado"] = dataset['tweets_limpos'].apply(separarTexto)

#definir os dados para treino e teste

X = dataset["tweet_limpo_separado"]
y = dataset["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#realizar a embedding das palavras
# transformar cada token em um vetor de 100 coordenadas, um vetor para cada token
word2vec = Word2Vec(
    sentences= X_train,
    vector_size=100,
    window=5,
    min_count=2,
    epochs=20,
    seed=42,
)



def vetorizar_tweet(tokens, modelo):
    # procura no Word2Vec as 100 coordenadas de cada palavra do tweet
    for palavra in tokens:
        if palavra in modelo.wv:
            vetores_das_palavras = [modelo.wv[palavra]]

    # se zerar tudo, retorna tudo 0
    if len(vetores_das_palavras) == 0:
        return np.zeros(modelo.vector_size)
    
    # calcula e devolve o ponto central (a média de todas as palavras)
    return np.mean(vetores_das_palavras, axis=0)


# Passa a função em cada linha e empilha os resultados em matrizes NumPy
X_train_matriz = np.array([vetorizar_tweet(t, word2vec) for t in X_train])
X_test_matriz = np.array([vetorizar_tweet(t, word2vec) for t in X_test])



# Normaliza cada documento para norma L2 igual a 1.
# Vetores nulos permanecem nulos.
X_train = normalize(X_train_matriz, norm="l2")
X_test = normalize(X_test_matriz, norm="l2")



# cria a Inteligência Artificial 
modelo_svc = LinearSVC(random_state=42)

# treinamento (A fase de estudos)
modelo_svc.fit(X_train, y_train)

#  teste final
previsoes = modelo_svc.predict(X_test)


acuracia = accuracy_score(y_test, previsoes)

print("acuracia: ", acuracia)

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    previsoes,
    labels=["negative", "neutral", "positive"],
    display_labels=["Negativo", "Neutro", "Positivo"],
    values_format="d",
    cmap="Blues",
    ax=ax
)

ax.set_title("Matriz de Confusão - Sentimentos")
plt.tight_layout()
plt.show()