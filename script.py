from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
import pandas as pd
from src.constants import MODEL_PATH
from src.data import Data
from src.evaluation import Evaluation
from src.explanation import report_model
from src.feature_engineering import FeatureEngineering, FeatureSelection
from src.genetic_algorithm import GeneticAlgorithm
from src.model import Model
from src.model_selection import ModelSelection
from src.preprocessing import DataPreprocessor
from lightgbm import LGBMRegressor

app = Flask(__name__)

model = None
X_train_transformed = None
X_test_transformed = None
y_train = None
y_test = None
genetic_algorithm = None



def initialize_model():
    global model, X_train_transformed, X_test_transformed, y_train, y_test, genetic_algorithm
    data = Data()
    essays = data.load()
    
    data_preprocessor = DataPreprocessor()
    preprocessed_data = data_preprocessor.preprocess(essays)

    feature_selection = FeatureSelection()
    X, y = feature_selection.extract_features_and_labels(preprocessed_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_engineering = FeatureEngineering("one_hot_encoding")
    feature_engineering.fit(X_train)
    X_train_transformed = feature_engineering.transform(X_train)
    X_test_transformed = feature_engineering.transform(X_test)

    model_selection = ModelSelection()
    model_selection.fit(X_train_transformed, y_train, X_test_transformed, y_test)
    
    model = Model(model_class=model_selection.get_best_model_class())
    model.fit(X_train_transformed, y_train)
    model.save(MODEL_PATH)
    
    model = LGBMRegressor(force_row_wise=True)

    # Inicializando o Algoritmo Genético
    genetic_algorithm = GeneticAlgorithm(
        X_train_transformed,
        model,
        population_size=100,
        max_generations=1000,
    )
    genetic_algorithm.fit()


@app.route('/')
def home():
    return render_template('/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500

    data = request.json
    features = data['features']
    
    if not features or not isinstance(features, list):
        return jsonify({'error': 'Invalid input'}), 400

    X_new = pd.DataFrame([features])
    prediction = model.predict(X_new)
    
    return jsonify({'prediction': prediction.tolist()})


# Nova rota para enviar dados ao gráfico
@app.route('/chart-data', methods=['GET'])
def chart_data():
    global genetic_algorithm, model, X_train_transformed, X_test_transformed, y_train, y_test

    if not genetic_algorithm or not model:
        return jsonify({'error': 'Model or genetic algorithm not initialized'}), 500

    # Histórico do algoritmo genético (supondo que existe uma função `get_history` que retorna as pontuações)
    history = genetic_algorithm.get_history()  # Melhores pontuações por geração

    # Avaliação do modelo
    evaluation = Evaluation()
    train_predictions = model.predict(X_train_transformed)
    test_predictions = model.predict(X_test_transformed)

    # Relatório de avaliação para treino e teste
    train_report = evaluation.report(y_train, train_predictions, output_dict=True)
    test_report = evaluation.report(y_test, test_predictions, output_dict=True)

    # Montando os dados para o gráfico
    data = {
        'genetic_algorithm': {
            'generations': list(range(1, len(history) + 1)),
            'scores': history  # Exemplo: melhores scores por geração
        },
        'evaluation': {
            'train_accuracy': train_report['accuracy'],  # Supondo que `accuracy` está no report
            'test_accuracy': test_report['accuracy'],
            'train_precision': train_report.get('precision', 0),  # Ajuste conforme suas métricas
            'test_precision': test_report.get('precision', 0)
        }
    }

    return jsonify(data)


if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
