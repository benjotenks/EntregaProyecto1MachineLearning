import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, LSTM, SimpleRNN, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json


def cargar_datos(path_csv):
    corpus, target, edad, sexo = [], [], [], []
    with open(path_csv, encoding="utf-8") as archivo:
        header = archivo.readline().strip().split(";")
        features = []
        for col in header:
            col = col.split("-")[0].strip()
            if col.startswith("Diag") or col.startswith("Proc"):
                col = col.split(" ")
                col = col[0] + col[1]
            features.append(col)
        for linea in archivo:
            row = []
            linea = linea.strip().split(";")
            for i in range(len(linea)):
                col = linea[i].split("-")[0].strip()
                if i == 67:
                    grd = col
                    target.append(grd)
                elif i == 66:
                    sexo.append(1 if col == "Mujer" else 0)
                elif i == 65:
                    edad.append(int(col))
                else:
                    row.append(col)
            corpus.append(row)
    df = pd.DataFrame(corpus, columns=features[:-3])
    df["GRD"] = target
    df["Edad"] = edad
    df["Sexo"] = sexo
    return df


def filtrar_top_grd(df, n_top=10):
    top_grd = df["GRD"].value_counts().index[:n_top]
    return df[df["GRD"].isin(top_grd)]


def obtener_features():
    features_diagnosticos = [f"Diag{str(j).zfill(2)}" for j in range(1, 36)]
    features_procedimientos = [f"Proced{str(j).zfill(2)}" for j in range(1, 31)]
    return features_diagnosticos, features_procedimientos


def construir_vocabulario(df, features_diagnosticos, features_procedimientos):
    token_diagnosticos = {}
    token_procedimientos = {}
    vocabulario = set()
    for field in features_diagnosticos:
        for row in df[field]:
            if row not in token_diagnosticos and row != '':
                token_diagnosticos[row] = len(token_diagnosticos) + 1
                vocabulario.add(row)
    for field in features_procedimientos:
        for row in df[field]:
            if row not in token_procedimientos and row != '':
                token_procedimientos[row] = len(token_procedimientos) + 1 + len(token_diagnosticos)
                vocabulario.add(row)
    return token_diagnosticos, token_procedimientos, vocabulario


def convertir_a_tokens(df, features_diagnosticos, features_procedimientos, token_diagnosticos, token_procedimientos):
    token_rows = []
    for _, row in df.iterrows():
        token_row = []
        for field in features_diagnosticos:
            token = token_diagnosticos.get(row[field], 0)
            token_row.append(token)
        for field in features_procedimientos:
            token = token_procedimientos.get(row[field], 0)
            token_row.append(token)
        token_row.append(row["Edad"])
        token_row.append(row["Sexo"])
        token_rows.append(token_row)
    return np.array(token_rows)


def analizar_datos(df):
    """Realiza un análisis exploratorio de los datos"""
    print("\n--- Análisis Exploratorio de Datos ---")
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df[["Edad", "Sexo"]].describe())
    
    # Distribución de GRDs
    print("\nDistribución de GRDs:")
    grd_counts = df["GRD"].value_counts()
    print(grd_counts)
    
    # Visualizaciones
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df["GRD"], order=df["GRD"].value_counts().index)
    plt.title("Distribución de GRDs")
    plt.tight_layout()
    plt.savefig("distribucion_grd.png")
    
    # Distribución de edad por GRD
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="GRD", y="Edad", data=df)
    plt.title("Distribución de Edad por GRD")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("edad_por_grd.png")
    
    # Distribución por sexo
    plt.figure(figsize=(10, 5))
    sexo_por_grd = pd.crosstab(df["GRD"], df["Sexo"])
    sexo_por_grd.columns = ["Hombre", "Mujer"]
    sexo_por_grd.plot(kind="bar", stacked=True)
    plt.title("Distribución de Sexo por GRD")
    plt.tight_layout()
    plt.savefig("sexo_por_grd.png")
    
    # Análisis de calidad de datos
    print("\nAnálisis de calidad de datos:")
    print(f"Valores nulos en el dataset: {df.isnull().sum().sum()}")
    
    # Verificar valores vacíos en diagnósticos y procedimientos
    diag_cols = [col for col in df.columns if col.startswith("Diag")]
    proc_cols = [col for col in df.columns if col.startswith("Proced")]
    
    vacios_diag = (df[diag_cols] == '').sum().sum()
    vacios_proc = (df[proc_cols] == '').sum().sum()
    
    print(f"Campos de diagnóstico vacíos: {vacios_diag}")
    print(f"Campos de procedimiento vacíos: {vacios_proc}")
    print(f"Total de campos: {len(df) * (len(diag_cols) + len(proc_cols))}")

def crear_modelo_lstm(vocabulario_size, maxlen, num_classes):
    """Crea un modelo LSTM"""
    model = Sequential()
    model.add(Embedding(input_dim=vocabulario_size, output_dim=128, input_length=maxlen))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def crear_modelo_simple_rnn(vocabulario_size, maxlen, num_classes):
    """Crea un modelo SimpleRNN"""
    model = Sequential()
    model.add(Embedding(input_dim=vocabulario_size, output_dim=128, input_length=maxlen))
    model.add(SimpleRNN(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def crear_modelo_denso(vocabulario_size, maxlen, num_classes):
    """Crea un modelo denso (MLP)"""
    model = Sequential()
    model.add(Embedding(input_dim=vocabulario_size, output_dim=128, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def entrenar_y_evaluar_modelo(model, X_train, y_train, X_test, y_test, label_encoder, model_name):
    """Entrena y evalúa un modelo"""
    # Crear directorio para logs y checkpoints
    log_dir = f"logs/fit/{model_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(
        filepath=f"models/{model_name}_best.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Entrenamiento
    print(f"\nEntrenando modelo {model_name}...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping]
    )
    
    # Evaluación
    print(f"\nEvaluando modelo {model_name}...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Métricas
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png")
    
    # Visualizar curvas de aprendizaje
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Precisión del modelo - {model_name}')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Pérdida del modelo - {model_name}')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"learning_curves_{model_name}.png")
    
    # Guardar el modelo
    model.save(f"models/{model_name}_final.h5")
    
    return accuracy, history


def validacion_cruzada(X, y, crear_modelo_fn, vocabulario_size, maxlen, num_classes, model_name, n_splits=5):
    """Realiza validación cruzada para evaluar el rendimiento del modelo de manera más robusta"""
    print(f"      Iniciando validación cruzada con {n_splits} folds...")
    
    # Inicializar StratifiedKFold para mantener la distribución de clases
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Listas para almacenar resultados
    fold_accuracies = []
    fold_losses = []
    all_y_true = []
    all_y_pred = []
    
    # Para visualización
    plt.figure(figsize=(15, 10))
    
    # Realizar validación cruzada
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"        Fold {fold+1}/{n_splits}: ", end="")
        
        # Dividir datos para este fold
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Crear y compilar modelo
        model = crear_modelo_fn(vocabulario_size, maxlen, num_classes)
        
        # Callbacks para este fold
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        # Entrenar modelo
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=15,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluar modelo
        loss, accuracy = model.evaluate(X_test_fold, y_test_fold, verbose=0)
        fold_accuracies.append(accuracy)
        fold_losses.append(loss)
        
        # Predicciones
        y_pred = model.predict(X_test_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Guardar para métricas globales
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred_classes)
        
        # Graficar curvas de aprendizaje para este fold
        plt.subplot(n_splits, 2, 2*fold+1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Precisión - Fold {fold+1}')
        plt.ylabel('Precisión')
        plt.xlabel('Época')
        plt.legend(['Train', 'Val'], loc='lower right')
        
        plt.subplot(n_splits, 2, 2*fold+2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Pérdida - Fold {fold+1}')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend(['Train', 'Val'], loc='upper right')
        
        print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Guardar gráfico de curvas de aprendizaje
    plt.tight_layout()
    plt.savefig(f"cv_learning_curves_{model_name}.png")
    
    # Calcular y mostrar métricas promedio
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    
    print(f"      Resultado final: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    # Visualizar distribución de accuracy por fold
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits+1), [acc*100 for acc in fold_accuracies], yerr=[std_accuracy*100]*n_splits)
    plt.axhline(y=mean_accuracy*100, color='r', linestyle='-', label=f'Media: {mean_accuracy*100:.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Distribución de Accuracy por Fold - {model_name}')
    plt.xticks(range(1, n_splits+1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cv_accuracy_distribution_{model_name}.png")
    
    return mean_accuracy, std_accuracy, all_y_true, all_y_pred

def main():
    # Crear directorios necesarios para el funcionamiento del programa
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/fit", exist_ok=True)
    
    print("\n" + "="*80)
    print("INICIO: PREDICCIÓN DE GRD CON REDES NEURONALES")
    print("="*80)
    
    path_csv = "dataset_elpino.csv"
    if not os.path.exists(path_csv):
        print(f"El archivo {path_csv} no se encuentra en el directorio actual.")
        return
    
    print("\n[1/5] Cargando y procesando datos...")
    df = cargar_datos(path_csv)
    print(f"Total de registros: {len(df)}")
    
    df = filtrar_top_grd(df, 10)
    print(f"Registros tras filtrar top 10 GRD: {len(df)}")
    
    print("\n[2/5] Realizando análisis exploratorio de datos...")
    print("      (Generando visualizaciones y estadísticas)")
    analizar_datos(df)
    
    print("\n[3/5] Preparando datos para entrenamiento...")
    features_diagnosticos, features_procedimientos = obtener_features()
    token_diagnosticos, token_procedimientos, vocabulario = construir_vocabulario(df, features_diagnosticos, features_procedimientos)
    print(f"Diagnósticos únicos: {len(token_diagnosticos)}")
    print(f"Procedimientos únicos: {len(token_procedimientos)}")
    print(f"Tamaño total del vocabulario: {len(vocabulario)}")
    
    X = convertir_a_tokens(df, features_diagnosticos, features_procedimientos, token_diagnosticos, token_procedimientos)
    y = df["GRD"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Padding
    maxlen = X.shape[1]
    X_seq = pad_sequences(X, maxlen=maxlen, padding='post')
    vocabulario_size = len(token_diagnosticos) + len(token_procedimientos) + 2
    num_classes = len(label_encoder.classes_)
    
    # Definir funciones para crear modelos
    modelo_creators = {
        "LSTM": lambda vs, ml, nc: crear_modelo_lstm(vs, ml, nc),
        "SimpleRNN": lambda vs, ml, nc: crear_modelo_simple_rnn(vs, ml, nc),
        "Dense": lambda vs, ml, nc: crear_modelo_denso(vs, ml, nc)
    }
    
    # Realizar validación cruzada para cada tipo de modelo
    print("\n[4/5] Evaluando modelos mediante validación cruzada (5-fold)...")
    print("      Este proceso puede tomar varios minutos.")
    resultados_cv = {}
    for nombre, crear_fn in modelo_creators.items():
        print(f"\n      • Evaluando modelo: {nombre}")
        mean_acc, std_acc, y_true, y_pred = validacion_cruzada(
            X_seq, y_encoded, crear_fn, vocabulario_size, maxlen, num_classes, nombre, n_splits=5
        )
        resultados_cv[nombre] = (mean_acc, std_acc)
    
    # Comparar modelos con validación cruzada
    print("\n      RESULTADOS DE LA VALIDACIÓN CRUZADA:")
    print("      " + "-"*40)
    for nombre, (acc, std) in resultados_cv.items():
        print(f"      {nombre}: {acc*100:.2f}% ± {std*100:.2f}%")
    
    # Determinar el mejor modelo basado en validación cruzada
    mejor_modelo_cv = max(resultados_cv, key=lambda k: resultados_cv[k][0])
    mejor_acc_cv, mejor_std_cv = resultados_cv[mejor_modelo_cv]
    print(f"\n      ► MEJOR MODELO: {mejor_modelo_cv} con precisión de {mejor_acc_cv*100:.2f}% ± {mejor_std_cv*100:.2f}%")
    
    # Entrenar el mejor modelo en todo el conjunto de entrenamiento
    print(f"\n[5/5] Entrenando el modelo final ({mejor_modelo_cv}) en el conjunto completo...")
    
    # Split para evaluación final
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_encoded, test_size=0.2, random_state=42)
    
    # Crear y entrenar el mejor modelo
    mejor_modelo = modelo_creators[mejor_modelo_cv](vocabulario_size, maxlen, num_classes)
    acc, hist = entrenar_y_evaluar_modelo(mejor_modelo, X_train, y_train, X_test, y_test, label_encoder, f"{mejor_modelo_cv}_Final")
    
    # Cargar el mejor modelo para demostración
    print(f"\nCargando el mejor modelo ({mejor_modelo_cv}) para demostración...")
    mejor_modelo_cargado = load_model(f"models/{mejor_modelo_cv}_Final_final.h5")
    
    # Ejemplo de predicción
    print("\n--- Ejemplo de Predicción ---")
    indice_ejemplo = np.random.randint(0, len(X_test))
    ejemplo_x = X_test[indice_ejemplo:indice_ejemplo+1]
    ejemplo_y = y_test[indice_ejemplo]
    
    prediccion = mejor_modelo_cargado.predict(ejemplo_x)
    clase_predicha = np.argmax(prediccion[0])
    
    print(f"GRD real: {label_encoder.classes_[ejemplo_y]}")
    print(f"GRD predicho: {label_encoder.classes_[clase_predicha]}")
    print(f"Confianza: {prediccion[0][clase_predicha]*100:.2f}%")
    
    # Generar gráfico comparativo de modelos
    plt.figure(figsize=(10, 6))
    nombres = list(resultados_cv.keys())
    medias = [resultados_cv[nombre][0]*100 for nombre in nombres]
    stds = [resultados_cv[nombre][1]*100 for nombre in nombres]
    
    plt.bar(nombres, medias, yerr=stds, capsize=10, color=['blue', 'green', 'orange'])
    plt.axhline(y=np.mean(medias), color='r', linestyle='--', label=f'Media global: {np.mean(medias):.2f}%')
    plt.xlabel('Modelo')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparación de Modelos con Validación Cruzada')
    plt.ylim([min(medias)-max(stds)-5, 100])
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparacion_modelos_cv.png")
    
    # Guardar datos necesarios para predicciones futuras
    print("\n[+] Guardando datos para predicciones futuras...")
    datos_prediccion = {
        "token_diagnosticos": token_diagnosticos,
        "token_procedimientos": token_procedimientos,
        "vocabulario_size": vocabulario_size,
        "maxlen": maxlen,
        "features_diagnosticos": features_diagnosticos,
        "features_procedimientos": features_procedimientos,
        "mejor_modelo": mejor_modelo_cv,
        "clases": label_encoder.classes_.tolist()
    }
    
    with open("models/datos_prediccion.pkl", "wb") as f:
        pickle.dump(datos_prediccion, f)
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO")
    print("="*80)
    print("\nResumen de archivos generados:")
    print("  • Visualizaciones de análisis de datos: distribucion_grd.png, edad_por_grd.png, sexo_por_grd.png")
    print("  • Matrices de confusión: confusion_matrix_*.png")
    print("  • Curvas de aprendizaje: learning_curves_*.png")
    print("  • Resultados de validación cruzada: cv_*.png")
    print("  • Comparación de modelos: comparacion_modelos_cv.png")
    print("  • Modelos guardados en carpeta 'models/'")
    print("  • Datos para predicciones: models/datos_prediccion.pkl")
    
    # Preguntar si desea hacer predicciones
    ejecutar_predicciones()


def predecir_nuevo_caso(diagnosticos, procedimientos, edad, sexo):
    """Predice el GRD para un nuevo caso usando el modelo entrenado"""
    # Cargar datos de predicción
    try:
        with open("models/datos_prediccion.pkl", "rb") as f:
            datos = pickle.load(f)
        
        # Cargar el mejor modelo
        modelo = load_model(f"models/{datos['mejor_modelo']}_Final_final.h5")
        
        # Tokenizar diagnósticos y procedimientos
        tokens = []
        
        # Procesar diagnósticos
        for i in range(len(datos['features_diagnosticos'])):
            if i < len(diagnosticos) and diagnosticos[i] != '':
                token = datos['token_diagnosticos'].get(diagnosticos[i], 0)
            else:
                token = 0
            tokens.append(token)
        
        # Procesar procedimientos
        for i in range(len(datos['features_procedimientos'])):
            if i < len(procedimientos) and procedimientos[i] != '':
                token = datos['token_procedimientos'].get(procedimientos[i], 0)
            else:
                token = 0
            tokens.append(token)
        
        # Añadir edad y sexo
        tokens.append(edad)
        tokens.append(sexo)
        
        # Convertir a array y hacer padding
        X = np.array([tokens])
        X_padded = pad_sequences(X, maxlen=datos['maxlen'], padding='post')
        
        # Hacer predicción
        prediccion = modelo.predict(X_padded, verbose=0)
        clase_predicha = np.argmax(prediccion[0])
        confianza = prediccion[0][clase_predicha] * 100
        
        return {
            "grd": datos['clases'][clase_predicha],
            "confianza": confianza,
            "todas_prob": [(datos['clases'][i], prediccion[0][i] * 100) for i in range(len(datos['clases']))]
        }
    
    except Exception as e:
        print(f"Error al hacer la predicción: {e}")
        return None

def ejecutar_predicciones():
    """Función interactiva para hacer predicciones con nuevos casos"""
    while True:
        print("\n" + "-"*80)
        print("PREDICCIÓN DE GRD CON NUEVOS DATOS")
        print("-"*80)
        print("\n¿Desea realizar una predicción con nuevos datos? (s/n): ", end="")
        respuesta = input().lower()
        
        if respuesta != 's':
            print("\nGracias por usar el sistema de predicción de GRD.")
            break
        
        print("\nIngrese los datos del nuevo caso:")
        
        # Recopilar diagnósticos
        diagnosticos = []
        print("\nDIAGNÓSTICOS (ingrese códigos, deje en blanco para terminar)")
        for i in range(1, 6):  # Limitamos a 5 diagnósticos para simplificar
            diag = input(f"Diagnóstico {i}: ").strip()
            if diag == '':
                break
            diagnosticos.append(diag)
        
        # Recopilar procedimientos
        procedimientos = []
        print("\nPROCEDIMIENTOS (ingrese códigos, deje en blanco para terminar)")
        for i in range(1, 6):  # Limitamos a 5 procedimientos para simplificar
            proc = input(f"Procedimiento {i}: ").strip()
            if proc == '':
                break
            procedimientos.append(proc)
        
        # Recopilar edad y sexo
        try:
            edad = int(input("\nEdad: "))
            sexo_input = input("Sexo (M/F): ").upper()
            sexo = 1 if sexo_input == 'F' else 0
            
            # Hacer predicción
            resultado = predecir_nuevo_caso(diagnosticos, procedimientos, edad, sexo)
            
            if resultado:
                print("\nRESULTADO DE LA PREDICCIÓN:")
                print(f"GRD predicho: {resultado['grd']}")
                print(f"Confianza: {resultado['confianza']:.2f}%")
                
                print("\nProbabilidades para todos los GRD:")
                # Ordenar por probabilidad descendente
                probs_ordenadas = sorted(resultado['todas_prob'], key=lambda x: x[1], reverse=True)
                for grd, prob in probs_ordenadas[:3]:  # Mostrar solo los 3 más probables
                    print(f"  {grd}: {prob:.2f}%")
            
        except ValueError:
            print("Error: Ingrese valores válidos para edad y sexo.")

if __name__ == "__main__":
    main()
