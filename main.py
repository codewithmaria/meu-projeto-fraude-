from src.data_loader import carregar_e_processar
from src.model_setup import construir_modelo
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# Executando o fluxo completo
X_train, X_test, y_train, y_test = carregar_e_processar()

input_dim = X_train.shape[1]
model = construir_modelo(input_dim)

# Treinamento (parâmetros: 10 épocas, batch 256)
model.fit(X_train, y_train, epochs=10, batch_size=256, 
          validation_data=(X_test, y_test), verbose=1)

# 6. Acurácia e Previsões
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
print(f"\n========================================================")
print(f"Acurácia Final: {accuracy * 100:.4f}%")
print(f"========================================================")

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=['Não Fraude (0)', 'Fraude (1)'])
plt.title('Matriz de Confusão do Modelo de Detecção de Fraude')
plt.show()