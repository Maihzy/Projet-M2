from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, send_file
from mistralai import Mistral
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.data_processing import DataProcessing
from modules.dataset_analyzer import DatasetAnalyzer
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import joblib

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()

API_KEY = os.getenv("API_KEY")
AGENT_ID = "ag:f81b839b:20250724:untitled-agent:1747b6e9"
client = Mistral(api_key=API_KEY)

app = Flask(__name__, 
           template_folder='../WEB/templates', 
           static_folder='../WEB/static')
app.secret_key = 'credit_pour_tous_secret_key'
PROJECT_NAME = "Crédit pour Tous AI"

# Variables globales
current_dataset = None
current_dataset_name = None

MODEL_PATH = "DATA/models/final_model.pkl"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
model_features = model_data["features"]

def get_next_data_filename():
    data_dir = 'DATA'
    os.makedirs(data_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(data_dir) if f.startswith('data_') and f.endswith('.csv')]
    numbers = []
    for f in existing_files:
        try:
            num = int(f.split('_')[1])
            numbers.append(num)
        except:
            continue
    next_num = max(numbers) + 1 if numbers else 1
    date_str = datetime.now().strftime('%Y%m%d')
    return f"data_{next_num}_{date_str}.csv"

@app.route('/')
def index():
    dataset_info = None
    if current_dataset is not None:
        risk_counts = current_dataset['risk_level'].value_counts().to_dict() if 'risk_level' in current_dataset.columns else {}
        dataset_info = {
            'count': len(current_dataset),
            'risk_summary': risk_counts,
            'filename': current_dataset_name
        }
    logging.info("Affichage de la page d'accueil.")
    return render_template(
        'index.html',
        project_name=PROJECT_NAME,
        avatar_url=url_for('static', filename='logo.svg'),
        has_data=current_dataset is not None,
        dataset_info=dataset_info
    )

@app.route('/upload', methods=['POST'])
def upload_data():
    global current_dataset, current_dataset_name
    if 'file' not in request.files:
        logging.warning("Aucun fichier sélectionné lors de l'upload.")
        return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.warning("Nom de fichier vide lors de l'upload.")
        return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
    if not file.filename.endswith('.csv'):
        logging.warning(f"Format non supporté : {file.filename}")
        return jsonify({'success': False, 'error': 'Format non supporté. Utilisez un fichier .csv'}), 400

    temp_filepath = None
    clean_filepath = None
    try:
        temp_dir = 'uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_filepath = os.path.join(temp_dir, file.filename)
        file.save(temp_filepath)
        logging.info(f"Fichier {file.filename} sauvegardé temporairement.")
        raw_data = pd.read_csv(temp_filepath, encoding='utf-8', sep=None, engine='python')
        if len(raw_data) == 0:
            logging.warning("Le fichier CSV ne contient aucune donnée.")
            os.remove(temp_filepath)
            return jsonify({'success': False, 'error': 'Le fichier CSV ne contient aucune donnée'}), 400
        if len(raw_data.columns) < 2:
            logging.warning("Le fichier CSV contient moins de 2 colonnes.")
            os.remove(temp_filepath)
            return jsonify({
                'success': False, 
                'error': f'Le fichier CSV contient seulement {len(raw_data.columns)} colonne(s). Minimum 2 requis.'
            }), 400
        processor = DataProcessing(raw_data)
        processed_data = processor.full_processing()
        clean_filename = get_next_data_filename()
        clean_filepath = os.path.join('DATA', clean_filename)
        processed_data.to_csv(clean_filepath, index=False)
        current_dataset = processed_data
        current_dataset_name = clean_filename
        logging.info(f"Dataset {clean_filename} traité et chargé ({len(processed_data)} lignes).")
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        total = len(processed_data)
        risk_stats = processed_data['risk_level'].value_counts().to_dict()
        success_message = f"Portefeuille analysé avec succès !"
        return jsonify({
            'success': True,
            'message': success_message,
            'filename': clean_filename,
            'num_records': total,
            'risk_summary': risk_stats,
            'original_filename': file.filename,
            'processing_method': 'DataProcessing complet'
        })
    except Exception as e:
        logging.error(f"Erreur lors du traitement du fichier : {str(e)}")
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        if clean_filepath and os.path.exists(clean_filepath):
            os.remove(clean_filepath)
        return jsonify({
            'success': False, 
            'error': f'Erreur lors du traitement: {str(e)}'
        }), 500

def get_model_results(user_message):
    if isinstance(user_message, dict):
        input_data = {feat: user_message.get(feat, None) for feat in model_features}
    else:
        input_data = {feat: None for feat in model_features}
    X = pd.DataFrame([input_data])
    X = X[model_features]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    score = float(model.predict_proba(X)[0][1])
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_vals = shap_values[-1][0]
        else:
            shap_vals = shap_values[0]
        shap_list = [
            {"feature": feat, "shap": float(shap_vals[i])}
            for i, feat in enumerate(model_features)
        ]
    except Exception as e:
        logging.warning(f"Erreur SHAP : {str(e)}")
        shap_list = []
    return {"score": score, "shap": shap_list}

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    logging.info(f"Message reçu du frontend : {user_message}")
    # --- Analyse client spécifique ---
    if "analyse le client" in user_message.lower():
        import re
        match = re.search(r"client\s*(\d+)", user_message.lower())
        if match and current_dataset is not None:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(current_dataset):
                client_data = current_dataset.iloc[idx].to_dict()
                model_results = get_model_results(client_data)
                shap_str = ", ".join([f"{item['feature']} ({item['shap']:.2f})" for item in model_results['shap']])
                tool_message = (
                    f"Analyse du client {idx+1} :\n"
                    f"- Score de défaut : {model_results['score']:.2f}\n"
                    f"- Facteurs SHAP principaux : {shap_str}\n"
                )
                inputs = [
                    {"role": "assistant", "content": tool_message},
                    {"role": "user", "content": user_message}
                ]
                response = client.beta.conversations.start(
                    agent_id=AGENT_ID,
                    inputs=inputs
                )
                bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
                logging.info(f"Réponse chatbot : {bot_reply}")
                return jsonify({"response": bot_reply})

    # --- Effectifs par segment ---
    elif "effectifs" in user_message.lower() or "nombre total de clients" in user_message.lower():
        if current_dataset is not None and 'risk_level' in current_dataset.columns:
            counts = current_dataset['risk_level'].value_counts().to_dict()
            total = len(current_dataset)
            txt = "Effectifs par segment de risque :\n"
            for seg, n in counts.items():
                txt += f"- {seg} : {n} clients\n"
            txt += f"Total clients : {total}\n"
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Proportions par segment ---
    elif "proportion" in user_message.lower() or "pourcentage" in user_message.lower():
        if current_dataset is not None and 'risk_level' in current_dataset.columns:
            counts = current_dataset['risk_level'].value_counts()
            total = len(current_dataset)
            txt = "Proportions par segment de risque :\n"
            for seg, n in counts.items():
                pct = (n / total) * 100
                txt += f"- {seg} : {pct:.1f}%\n"
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Volumes financiers par segment ---
    elif "encours" in user_message.lower() or "volume financier" in user_message.lower():
        if current_dataset is not None and 'risk_level' in current_dataset.columns and 'loan_amnt' in current_dataset.columns:
            txt = "Encours total de prêts par segment :\n"
            for seg, group in current_dataset.groupby('risk_level'):
                total_encours = group['loan_amnt'].sum()
                txt += f"- {seg} : {total_encours:,.0f} €\n"
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Montant moyen par segment ---
    elif "montant moyen" in user_message.lower():
        if current_dataset is not None and 'risk_level' in current_dataset.columns and 'loan_amnt' in current_dataset.columns:
            txt = "Montant moyen emprunté par client et par segment :\n"
            for seg, group in current_dataset.groupby('risk_level'):
                mean_amt = group['loan_amnt'].mean()
                txt += f"- {seg} : {mean_amt:,.2f} €\n"
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Synthèse textuelle ---
    elif "synthèse" in user_message.lower() or "commentaire" in user_message.lower():
        if current_dataset is not None and 'risk_level' in current_dataset.columns and 'loan_amnt' in current_dataset.columns:
            counts = current_dataset['risk_level'].value_counts()
            total = len(current_dataset)
            dominant = counts.idxmax()
            pct_dom = (counts.max() / total) * 100
            txt = (
                f"Le segment dominant est '{dominant}' ({pct_dom:.1f}% des clients). "
                f"Le portefeuille présente {counts.get('Risque Faible',0)} clients à faible risque, "
                f"{counts.get('Risque Modéré',0)} à risque modéré et "
                f"{counts.get('Risque Élevé',0)} à risque élevé. "
                "On observe une tendance stable, sans anomalie majeure. "
                "La croissance du segment faible risque est à surveiller pour optimiser l’acceptation."
            )
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Risque Élevé - Nombre de clients ---
    elif "risque élevé" in user_message.lower() and ("combien" in user_message.lower() or "nombre" in user_message.lower()):
        if current_dataset is not None and 'risk_level' in current_dataset.columns:
            n_eleve = current_dataset['risk_level'].value_counts().get('Risque Élevé', 0)
            total = len(current_dataset)
            txt = (
                f"Il y a {n_eleve} client(s) à risque élevé dans le fichier actuellement chargé "
                f"(sur un total de {total} clients)."
            )
            inputs = [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": user_message}
            ]
            response = client.beta.conversations.start(
                agent_id=AGENT_ID,
                inputs=inputs
            )
            bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
            logging.info(f"Réponse chatbot : {bot_reply}")
            return jsonify({"response": bot_reply})

    # --- Par défaut ---
    response = client.beta.conversations.start(
        agent_id=AGENT_ID,
        inputs=[{"role": "user", "content": user_message}]
    )
    bot_reply = response.outputs[0].content if isinstance(response.outputs, list) else response.outputs.content
    logging.info(f"Réponse chatbot (défaut) : {bot_reply}")
    return jsonify({"response": bot_reply})

@app.route('/reset-dataset', methods=['POST'])
def reset_dataset():
    global current_dataset, current_dataset_name
    current_dataset = None
    current_dataset_name = None
    logging.info("Dataset réinitialisé. Mode conversation libre activé.")
    flash('Dataset supprimé. Mode conversation libre activé.')
    return redirect(url_for('index'))

@app.route('/api/data-info')
def data_info():
    if current_dataset is not None:
        logging.info("Envoi des infos dataset au frontend.")
        return jsonify({
            'has_data': True,
            'num_records': len(current_dataset),
            'filename': current_dataset_name,
            'columns': list(current_dataset.columns),
            'risk_summary': current_dataset['risk_level'].value_counts().to_dict() if 'risk_level' in current_dataset.columns else {}
        })
    logging.info("Aucun dataset chargé, infos non envoyées.")
    return jsonify({'has_data': False})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('DATA', exist_ok=True)
    logging.info("Lancement du serveur Flask.")
    app.run(debug=True, port=5000)
