import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def carregar_e_processar():
    # 1. Carregamento (Seu código original)
    path = kagglehub.dataset_download("chitwanmanchanda/fraudulent-transactions-data")
    csv_path = os.path.join(path, "Fraud.csv")
    df = pd.read_csv(csv_path)

    # 2. Pré-processamento (Suas transformações originais)
    le_orig = LabelEncoder()
    le_dest = LabelEncoder()
    le_type = LabelEncoder()

    df['nameOrig_num'] = le_orig.fit_transform(df['nameOrig'])
    df['nameDest_num'] = le_dest.fit_transform(df['nameDest'])
    df['type_num'] = le_type.fit_transform(df['type'])

    # 3. Definição de X e Y (Suas features escolhidas)
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'type_num', 'nameOrig_num', 'nameDest_num']
    target = 'isFraud'

    X = df[features]
    y = df[target]

    # 4. Divisão e Escalonamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test