import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def construir_modelo(input_dim):
    # 5. Compilado e Treinamento
    model = Sequential()
    
    # Camada de entrada e primeira oculta (64 neurônios)
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    
    # Segunda camada Oculta (32 neurônios)
    model.add(Dense(units=32, activation='relu'))
    
    # Saída
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model