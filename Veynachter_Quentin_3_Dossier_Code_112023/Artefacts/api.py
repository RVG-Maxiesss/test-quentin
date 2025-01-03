import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request

#On créé l'application Flask
app = Flask(__name__)

#Chemin pour accéder à api.py
dir = os.path.dirname(os.path.abspath(__file__))

#On charge le modèle
model_path = os.path.join(dir, '.', 'model.pkl')
model = joblib.load(model_path)

#Et le scaler
scaler_path = os.path.join(dir, '.', 'scaler.pkl')
scaler = joblib.load(scaler_path)

#On définit une fonction pour la prédiction
@app.route("/predict", methods=['POST'])
def predict():
    #Pour récupérer les données de la requête POST
    data = request.json
    id = data['SK_ID_CURR']

    #On récupère data_final
    data_path = os.path.join(dir, ".", "data_final.parquet")
    df = pd.read_parquet(data_path)
    sample = df[df['SK_ID_CURR'] == id]

    #On supprimes les colonnes ID et Target pour la prédiction
    sample = sample.drop(columns=['TARGET', 'SK_ID_CURR'])

    #On applique le scaler
    sample_scaled = scaler.transform(sample)

    #On prédit la probabilité d'appartenance à la classe 1 (*100 pour l'avoir en pourcentage)
    proba = model.predict_proba(sample_scaled)[:, 1] * 100
    
    #On calcule les valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_scaled)
    
    #On retourne les valeurs SHAP ainsi que la probabilité, sous forme d'item ou liste car c'est nécessaire pour jsonify
    return jsonify({'probability': proba[0],
                    'shap_values': shap_values[0].tolist(),
                    'feature_names': sample.columns.tolist(),
                    'feature_values': sample.values[0].tolist()})

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
