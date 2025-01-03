import os
import sys
import joblib
import pandas as pd
import pytest
from flask import json

#Chemin relatif vers api.py et dashboard.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Veynachter_Quentin_3_Dossier_Code_112023', 'Artefacts')))

#On importe depuis api.py pour commencer
from api import app, dir

#On créé un client de test pour l'application Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

#On teste le chargement du modèle
def test_model_loading():
    model_path = os.path.join(dir, 'model.pkl')
    assert os.path.exists(model_path), f"Le modèle n'a pas été trouvé à {model_path}"
    model = joblib.load(model_path)
    assert model is not None, "Erreur dans le chargement du modèle"

#On teste le chargement du fichier .parquet
def test_parquet_loading():
    data_path = os.path.join(dir, 'data_final.parquet')
    assert os.path.exists(data_path), f"Le fichier .parquet n'a pas été trouvé à {data_path}"
    df = pd.read_parquet(data_path)
    assert not df.empty, "Erreur dans le chargement du fichier .parquet"

#On teste la fonction de prédiction
def test_predict(client):
    data_path = os.path.join(dir, 'data_final.parquet')
    df = pd.read_parquet(data_path)
    sk_id_curr = df.iloc[0]['SK_ID_CURR'] #1er SK_ID_CURR du dataset
    
    #On créé une requête de test à partir du SK_ID_CURR retenu
    with app.test_client() as client:
        response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})
        assert response.status_code == 200, f"Réponse inattendue avec statut {response.status_code}"
        data = json.loads(response.data)
        proba = data['probability']
        assert proba is not None, "La prédiction a échoué"

#On enchaîne avec dashboard.py
from dashboard import get_state, format_value, compute_color

#Simulation d'un état pour st.session_state
mocked_session_state = {'state': {'data_received': False, 'data': None, 'last_sk_id_curr': None}}

#On utilise monkeypatch pour remplacer l'attribut session_state de streamlit dans l'état simulé
@pytest.fixture
def mocked_st(monkeypatch):
    monkeypatch.setattr('streamlit.session_state', mocked_session_state)
    return mocked_session_state

#On teste la fonction compute_color()
@pytest.mark.parametrize('mocked_st', [mocked_session_state], indirect=True)
def test_compute_color(mocked_st):
    assert compute_color(30) == 'green', "Erreur dans la fonction compute_color"
    assert compute_color(70) == 'red', "Erreur dans la fonction compute_color"

#On teste la fonction format_value()
@pytest.mark.parametrize('mocked_st', [mocked_session_state], indirect=True)
def test_format_value(mocked_st):
    assert format_value(1.82) == 1.82, "Erreur dans la fonction format_value"
    assert format_value(7.00) == 7, "Erreur dans la fonction format_value"

#On teste la fonction get_state()
def test_get_state(mocked_st):	
    state = get_state()
    assert isinstance(state, dict), "La fonction get_state doit renvoyer un dictionnaire"
    assert 'data_received' in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'data_received'"
    assert 'data' in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'data'"
    assert 'last_sk_id_curr' in state, "Le dictionnaire renvoyé par get_state doit contenir la clé 'last_sk_id_curr'"
