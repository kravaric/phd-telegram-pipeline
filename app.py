##################################################################
# README: Telegram Data Pipeline with SetFit & Oversampling
# Supports multiple channels, fuzzy matching, n-grams, etc.
# Also saves predictions, confusion matrices, and classification
# reports for each model in separate files, plus an all-model CSV,
# and a pickle of the best model.
#
# 1) Install:
#    pip install streamlit telethon langdetect nltk deep-translator scikit-learn setfit rapidfuzz matplotlib seaborn nest_asyncio
# 2) Run:
#    streamlit run app.py
# 3) Steps in UI:
#    - 1) Collect Data (can provide multiple channels)
#    - 2) Detect Language & Clean
#    - 3) Labeling
#    - 4) Train Models (SetFit + classical ML, oversampling if needed)
##################################################################

import streamlit as st
import asyncio
import pandas as pd
import re
import nltk
from telethon import TelegramClient
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from deep_translator import GoogleTranslator
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from setfit import SetFitModel
from rapidfuzz import fuzz
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio
import joblib
import json

nest_asyncio.apply()

# -----------------------------
# ΒΑΛΕ ΤΑ ΔΙΚΑ ΣΟΥ CREDENTIALS
# -----------------------------
api_id = 1234567  # Βάλε το δικό σου API ID
api_hash = "abcdef1234567890abcdef1234567890"  # Βάλε το δικό σου API Hash

##################################################################
# Streamlit UI
##################################################################
st.title("📊 Telegram Data Pipeline with Multiple Channels, SetFit & Oversampling")

st.subheader("📥 Collect & Filter Messages from Multiple Telegram Channels")

channels_input = st.text_input("Enter channels/groups (comma separated, without @):")
num_messages = st.number_input(
    "Number of Messages to Retrieve:", min_value=10, max_value=10000, value=1000
)
min_word_count = st.number_input(
    "Minimum word count per message:", min_value=1, max_value=100, value=3
)

# Μετατρέπουμε το input σε λίστα
if channels_input:
    channel_list = [c.strip() for c in channels_input.split(",")]
else:
    channel_list = []

def filter_messages(messages, min_words):
    """Φιλτράρει τα μηνύματα με βάση το min_word_count."""
    return [msg for msg in messages if len(msg.split()) >= min_words]

##################################################################
# Συλλογή μηνυμάτων από πολλαπλά κανάλια
##################################################################
async def fetch_telegram_messages_multi(channel_list, num_messages, min_word_count):
    client = TelegramClient("session", api_id, api_hash)
    await client.connect()

    if not await client.is_user_authorized():
        st.error("❌ Unauthorized! Check API credentials.")
        return []

    all_messages = []
    for ch_name in channel_list:
        temp_messages = []
        try:
            async for message in client.iter_messages(ch_name, limit=num_messages):
                if message.text:
                    temp_messages.append(message.text)
        except Exception as e:
            st.warning(f"⚠️ Could not fetch from '{ch_name}'. Error: {e}")
        all_messages.extend(temp_messages)

    # Φιλτράρισμα (min_word_count)
    filtered = [msg for msg in all_messages if len(msg.split()) >= min_word_count]
    return filtered

##################################################################
# 1) Collect Data
##################################################################
if st.button("1) Collect Data"):
    if channel_list:
        data = asyncio.run(fetch_telegram_messages_multi(channel_list, num_messages, min_word_count))
        if data:
            st.success(f"✅ Collected {len(data)} messages from {len(channel_list)} channels!")
            st.session_state["raw_data"] = data
        else:
            st.error("❌ No data collected. Check channels or credentials.")
    else:
        st.warning("Please enter at least one channel.")

##################################################################
# 2) Αναγνώριση γλώσσας & Καθαρισμός
##################################################################
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def clean_text(text, lang):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\\w\\s]", "", text)
    try:
        stop_words = set(stopwords.words(lang))
    except:
        stop_words = set()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    try:
        stemmer = SnowballStemmer(lang)
    except:
        stemmer = None
    lemmatizer = WordNetLemmatizer()
    words = [stemmer.stem(lemmatizer.lemmatize(w)) if stemmer else lemmatizer.lemmatize(w) for w in words]
    return " ".join(words) if words else ""

if "raw_data" in st.session_state and st.button("2) Detect Language & Clean"):
    raw_data = st.session_state["raw_data"]
    df = pd.DataFrame(raw_data, columns=["message"])
    df["lang"] = df["message"].apply(detect_language)
    df["cleaned_message"] = df.apply(lambda row: clean_text(row["message"], row["lang"]), axis=1)
    st.session_state["df_raw"] = df
    st.success("✅ Language detection & cleaning complete!")
    st.write("🔍 Sample Data:", df.head(10))
else:
    st.warning("Collect data first (Step 1).")

##################################################################
# 3) Αυτόματη μετάφραση λέξεων-κλειδιών & Labeling
##################################################################
def translate_keyword(keyword, target_lang):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(keyword)
    except:
        return keyword

def generate_ngrams(text, n=2):
    words = text.split()
    return [" ".join(gram) for gram in ngrams(words, n)]

def label_message(row, user_keywords, threshold=50):
    text_words = set(row["cleaned_message"].split())
    ngram_words = set(generate_ngrams(row["cleaned_message"]))
    translated_keywords = {translate_keyword(kw, row["lang"]) for kw in user_keywords}
    score = sum(
        fuzz.partial_ratio(word, kw)
        for word in text_words.union(ngram_words)
        for kw in translated_keywords
    )
    return 1 if score >= threshold else 0

st.subheader("3) Labeling Messages")
keywords_input = st.text_area("Enter keywords (comma separated):")
if keywords_input:
    user_keywords = [kw.strip().lower() for kw in keywords_input.split(",")]
else:
    user_keywords = []

if "df_raw" in st.session_state and st.button("3) Apply Labeling"):
    df = st.session_state["df_raw"].copy()
    df["label"] = df.apply(lambda row: label_message(row, user_keywords, 50), axis=1)
    st.session_state["df_labeled"] = df
    st.write("📊 Label Distribution:")
    if "label" in df.columns:
        st.write(df["label"].value_counts())
    else:
        st.error("❌ 'label' column not found.")
else:
    st.warning("Detect language & clean first (Step 2), then apply labeling.")

##################################################################
# 4) Εκπαίδευση Μοντέλων (SetFit + Oversampling) + Αποθήκευση
##################################################################
def plot_metrics(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(plt)

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    st.pyplot(plt)

st.subheader("4) Train Machine Learning Models")

if st.button("4) Train Models"):
    if "df_labeled" not in st.session_state:
        st.error("❌ No data available! Please complete previous steps.")
    else:
        df = st.session_state["df_labeled"].copy()
        if "label" not in df.columns or df["label"].nunique() < 2:
            st.error("❌ Not enough labeled data. Check your keywords or add more data.")
        else:
            X_texts = df["cleaned_message"].tolist()
            y = df["label"].values

            # Oversampling αν υπάρχει ανισορροπία
            label_distribution = pd.Series(y).value_counts()
            majority_label = label_distribution.idxmax()
            minority_label = label_distribution.idxmin()
            if label_distribution[minority_label] < label_distribution[majority_label] * 0.5:
                st.warning("⚠️ Dataset is imbalanced! Applying oversampling.")
                X_series = pd.Series(X_texts)
                df_train = pd.DataFrame({"cleaned_text": X_series, "label": y})

                df_minority = df_train[df_train["label"] == minority_label]
                df_majority = df_train[df_train["label"] == majority_label]
                df_minority_upsampled = resample(
                    df_minority,
                    replace=True,
                    n_samples=len(df_majority),
                    random_state=42
                )
                df_train = pd.concat([df_majority, df_minority_upsampled])
                st.success("✅ Oversampling applied!")

                df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
                X_texts = df_train["cleaned_text"].tolist()
                y = df_train["label"].values

            # Δημιουργία TF-IDF για κλασικά μοντέλα
            vectorizer = TfidfVectorizer(max_features=3000)
            X_tfidf = vectorizer.fit_transform(X_texts)

            # Split
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                X_texts, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Neural Network": MLPClassifier(),
                "Naive Bayes": MultinomialNB(),
                "SetFit": SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
            }

            param_grid = {
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                "SVM": {"C": [0.1, 1]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "Neural Network": {"hidden_layer_sizes": [(50,), (100,)]}
            }

            from sklearn.model_selection import GridSearchCV
            best_models = {}
            results = {}
            df_results_list = []  # για αποθήκευση σε model_results.csv

            # Εκπαίδευση
            for name, model in models.items():
                if name == "SetFit":
                    # Εκπαίδευση SetFit με ωμά κείμενα
                    model.fit(X_train_texts, y_train.tolist(), num_epochs=5)
                    best_models[name] = model
                else:
                    # GridSearch αν οριστεί
                    if name in param_grid:
                        grid = GridSearchCV(model, param_grid[name], scoring="f1_weighted", cv=2, n_jobs=-1)
                        grid.fit(X_train_tfidf, y_train)
                        best_models[name] = grid.best_estimator_
                    else:
                        model.fit(X_train_tfidf, y_train)
                        best_models[name] = model

            # Αξιολόγηση μοντέλων + Αποθήκευση
            for name, model in best_models.items():
                if name == "SetFit":
                    y_pred = model.predict(X_test_texts)
                else:
                    y_pred = model.predict(X_test_tfidf)

                rep = classification_report(y_test, y_pred, output_dict=True)
                f1 = rep["weighted avg"]["f1-score"]
                results[name] = f1
                st.write(f"### Model: {name}, F1-score = {f1:.3f}")

                # Οπτικοποίηση
                plot_metrics(y_test, y_pred, name)

                # -----------------------------
                # Αποθήκευση Αποτελεσμάτων
                # -----------------------------

                # 1) Αποθήκευση προβλέψεων
                df_preds = pd.DataFrame({
                    "actual": y_test,
                    "predicted": y_pred
                })
                preds_filename = f"predictions_{name}.csv"
                df_preds.to_csv(preds_filename, index=False)
                st.success(f"✅ Saved predictions for {name} to '{preds_filename}'")

                # 2) Confusion Matrix CSV
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    columns=["Pred_0","Pred_1"],
                    index=["Actual_0","Actual_1"]
                )
                cm_filename = f"confusion_matrix_{name}.csv"
                cm_df.to_csv(cm_filename)
                st.success(f"✅ Confusion matrix for {name} saved to '{cm_filename}'")

                # 3) Classification report JSON
                cr_filename = f"classification_report_{name}.json"
                with open(cr_filename, "w") as jf:
                    json.dump(rep, jf, indent=2)
                st.success(f"✅ Classification report for {name} saved to '{cr_filename}'")

                # 4) Προσθήκη σε μία λίστα για ενιαίο CSV
                df_results_list.append({
                    "model": name,
                    "f1_score": f1
                })

            # Καλύτερο μοντέλο
            best_model = max(results, key=results.get)
            st.success(f"🏆 Best Model: {best_model} (F1-score: {results[best_model]:.3f})")

            # Αποθήκευση όλων των αποτελεσμάτων σε CSV
            df_results = pd.DataFrame(df_results_list)
            df_results.to_csv("model_results.csv", index=False)
            st.success("✅ Saved all model results to 'model_results.csv'")

            # Pickle του καλύτερου μοντέλου
            import joblib
            best_clf = best_models[best_model]
            joblib.dump(best_clf, "best_model.pkl")
            st.success(f"✅ Best model '{best_model}' saved to 'best_model.pkl'")
