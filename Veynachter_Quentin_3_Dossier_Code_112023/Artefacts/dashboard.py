import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import streamlit as st

#Chemin pour accéder à dashboard.py
dir = os.path.dirname(os.path.abspath(__file__))

#On récupère data_final
data_path = os.path.join(dir, '.', 'data_final.parquet')
df = pd.read_parquet(data_path)
threshold = 44.5262

#Pour étendre la largeur de la page
st.set_page_config(layout='wide')

#Fonction pour récupérer les états stockés
def get_state():
    if 'state' not in st.session_state:
        st.session_state['state'] = {'data_received': False,
                                     'data': None,
                                     'last_sk_id_curr': None}
    elif ('last_sk_id_curr' not in st.session_state['state']):  #On vérifie si 'last_sk_id_curr' existe
        st.session_state['state']['last_sk_id_curr'] = None  #On l'ajoute si ce n'est pas le cas

    return st.session_state['state']

#Fonction pour formater les valeurs en fonction de leur type
def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val) #Exemple : Remplace 5.0 par 5
        return round(val, 2) #Arrondit les floats à 2 décimales
    return val

#Fonction pour retourner une couleur (vert ou rouge) en fonction du threshold
def compute_color(value):
    if 0 <= value < threshold:
        return "green"
    elif threshold <= value <= 100:
        return "red"

state = get_state()

st.markdown("<h1 style='text-align: center; color: black;'>Probablité de remboursement</h1>", unsafe_allow_html=True)
sk_id_curr = st.text_input('Entrez le SK_ID_CURR :', on_change=lambda: state.update(run=True))

#Style pour le bouton
st.markdown("""
            <style>button {width: 60px !important; white-space: nowrap !important}</style>
            """,
            unsafe_allow_html=True)

if st.button('Run') or state['data_received']:
    #On vérifie si l'ID actuel est différent du dernier ID appelé
    if state['data_received'] and state['last_sk_id_curr'] == sk_id_curr:
        st.success('Données déjà reçues pour ce client')

    #Si l'ID a changé on réinitialise les données
    if state['last_sk_id_curr'] != sk_id_curr:
        state['data_received'] = False
        state['last_sk_id_curr'] = sk_id_curr  #On met à jour le dernier ID

    #Si les données n'ont pas été reçues pour ce client
    if not state['data_received']:
        response = requests.post('http://localhost:5000/predict', json={'SK_ID_CURR': int(sk_id_curr)})
        if response.status_code != 200: #Erreur 200 indique que la requête a été traitée avec succès
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code} - {response.text}")
            st.stop()

        state['data'] = response.json() #Sauvegarde les données reçues
        state['data_received'] = True #Marque les données comme reçues

    data = state['data']

    proba = data['probability']
    shap_values = data['shap_values']
    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values] #Si val est une liste on récupère le premier élément val[0], sinon on récupère val tel quel
    feature_names = data['feature_names']
    feature_values = data['feature_values']

    #On créé un df
    shap_df = pd.DataFrame(list(zip(feature_names,
                                    shap_values,
                                    [format_value(val) for val in feature_values])),
                           columns=['Feature', 'SHAP Value', 'Feature Value'])
    
    #On applique la couleur en fonction de la proba
    color = compute_color(proba)

    #Création d'une jauge pour voir où se positionne le client par rapport au threshold
    jauge_html = f"""
                <div style="position: relative; width: 100%; height: 30px; background: linear-gradient(to right, green {threshold}%, red {threshold}%); border-radius: 15px">
                    <div style="position: absolute; top: 50%; left: {proba}%; transform: translateX(-50%) translateY(-50%);
                    width: 20px; height: 20px; background-color: black; border-radius: 50%; border: 3px solid white">
                </div></div>
                """

    #On affiche
    st.markdown(jauge_html, unsafe_allow_html=True)

    #Un peu d'espace avant la suite
    st.markdown("<br>", unsafe_allow_html=True)

    #Message en fonction de la décision
    message_decision = ('Risque faible' if proba < threshold else 'Risque potentiel')
    st.markdown(f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{message_decision}</div>", unsafe_allow_html=True)

    #Encore un espace
    st.markdown("<br>", unsafe_allow_html=True)

    #On filtre le top 10 features qui réduisent et augmentent le risque
    shap_df_decrease = shap_df[shap_df['SHAP Value'] < 0].sort_values(by='SHAP Value').head(10)
    shap_df_increase = shap_df[shap_df['SHAP Value'] > 0].sort_values(by='SHAP Value', ascending=False).head(10)

    #On créé des subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    #Graphique pour les features qui réduisent le risque
    bars_left = axes[0].barh(shap_df_decrease['Feature'], shap_df_decrease['SHAP Value'], color='lightgreen')
    axes[0].set_xlabel('Feature importance')
    axes[0].set_title('Top 10 features qui réduisent le risque', weight='bold')
    axes[0].invert_yaxis() #Inverse l'axe pour que la feature la plus importante soit en haut
    axes[0].invert_xaxis() #Inverse l'axe pour les barres pointent vers la droite
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3)) #Limite à 3 xticks

    #On retire le cadre autour des axes
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    #On ajoute les feature values en annotations
    for bar, feature_value in zip(bars_left, shap_df_decrease['Feature Value']):
        axes[0].text(bar.get_width() - 0.005, #Hors de la barre
                     bar.get_y() + bar.get_height() / 2, #Centré dans la barre
                     f'{feature_value}',
                     va='center', ha='left', fontsize=10, fontweight='bold', color='black') #Aligné à gauche

    #Graphique pour les features qui augmentent le risque
    bars_right = axes[1].barh(shap_df_increase['Feature'], shap_df_increase['SHAP Value'], color='lightcoral')
    axes[1].set_xlabel('Feature importance')
    axes[1].set_title('Top 10 features qui augmentent le risque', weight='bold')
    axes[1].invert_xaxis() #Inverse l'axe pour les barres pointent vers la gauche
    axes[1].yaxis.tick_right() #Déplace les yticks à droite
    axes[1].yaxis.set_tick_params(left=False) #Désactive les yticks à gauche
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3)) #Limite à 3 xticks

    #On retire le cadre autour des axes
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    #On ajoute les feature values en annotations
    for bar, feature_value in zip(bars_right, shap_df_increase['Feature Value']):
        axes[1].text(bar.get_width() + 0.005, #Hors de la barre
                     bar.get_y() + bar.get_height() / 2, #Centré dans la barre
                     f'{feature_value}',
                     va='center', ha='right', fontsize=10, fontweight='bold', color='black') #Aligné à droite

    st.pyplot(fig)
