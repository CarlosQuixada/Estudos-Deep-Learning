import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                              test_size=0.25)

# Criando Rede Neural
classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dense(units=1, activation='sigmoid'))

otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Treinar
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

# Visualizar peso
peso0 = classificador.layers[0].get_weights()
peso1 = classificador.layers[1].get_weights()

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)
matrix = confusion_matrix(classe_teste, previsoes)
print("Accuracy: " + str(precisao))
