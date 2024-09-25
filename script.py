from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
import pandas as pd  # Certifique-se de importar pandas
from src.constants import MODEL_PATH
from src.data import Data
from src.evaluation import Evaluation
from src.explanation import report_model
from src.feature_engineering import FeatureEngineering, FeatureSelection
from src.genetic_algorithm import GeneticAlgorithm
from src.model import Model
from src.model_selection import ModelSelection
from src.preprocessing import DataPreprocessor

app = Flask(__name__)

# Inicialização do modelo
model = None

def initialize_model():
    global model
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
    X_test_transformed = feature_engineering.transform(X_test)  # Transformar X_test também

    model_selection = ModelSelection()
    model_selection.fit(X_train_transformed, y_train, X_test_transformed, y_test)  # Agora X_test_transformed está definido
    
    model = Model(model_class=model_selection.get_best_model_class())
    model.fit(X_train_transformed, y_train)
    model.save(MODEL_PATH)

@app.route('/')
def home():
    return render_template('templates/index.html')  # Crie um template HTML para a página inicial

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500

    data = request.json
    features = data['features']
    X_new = pd.DataFrame([features])
    prediction = model.predict(X_new)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
