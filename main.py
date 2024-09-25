import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import datetime
import concurrent.futures
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import pdist, squareform
import os
import streamlit as st

# Title of the app
st.title("Transaction Anomaly Detection Dashboard Using LLM-FinBert")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

df = None  # Initialize df to None at the beginning

# Load the dataset if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Print current date and time
    now = datetime.datetime.now()
    st.write("Current date and time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

    # Check if required columns are present in the DataFrame
    required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest', 'timestamp', 'creditCard', 'type', 'isFraud']
    
    if all(column in df.columns for column in required_columns):

        features = pd.DataFrame()
        
        # Numerical features
        numerical_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']
        features = df[numerical_columns].copy()

        # Flag large transactions
        large_transaction_threshold = 9000  # Align this with your data generation script
        features['largeAmount'] = (features['amount'] > large_transaction_threshold).astype(int)

        # Calculate balance changes for origin and destination
        features['changebalanceOrig'] = features['newbalanceOrg'] - features['oldbalanceOrg']
        features['changebalanceDest'] = features['newbalanceDest'] - features['oldbalanceDest']

        # Ensure timestamp is a datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort DataFrame to ensure proper ordering
        df = df.sort_values(by=['creditCard', 'timestamp'])

        # Check for duplicates
        duplicates = df[df.duplicated(['creditCard', 'timestamp'], keep=False)]
        if not duplicates.empty:
            st.write("Duplicate entries found:")
            st.write(duplicates)

        # Count the number of transactions for each credit card within a certain time window
        time_window = '1H'  # One hour time window

        # Group by 'creditCard' and ensure a valid rolling operation
        df.set_index('timestamp', inplace=True)

        # Add a rolling count of transactions
        df['countTransac'] = df.groupby('creditCard')['amount'].rolling(time_window).count().reset_index(level=0, drop=True)

        # Reset index after the rolling operation
        df.reset_index(drop=False, inplace=True)

        # One-hot encoding for transaction type
        type_one_hot = pd.get_dummies(df['type'])
        features = pd.concat([features, type_one_hot], axis=1)
        # Include the transaction count feature
        features['countTransac'] = df['countTransac']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, df['isFraud'], test_size=0.5, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle imbalanced dataset
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

        # Train the Random Forest model on the training set
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_smote, y_train_smote)

        # Evaluate the model on the test set
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Initialize FinBERT model and tokenizer
        model_name = "yiyanghkust/finbert-tone"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Function to get embeddings from FinBERT
        def get_single_embedding(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings

        # Function to get embeddings in parallel
        def get_embeddings_parallel(texts):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_single_embedding, text) for text in texts]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

        # Combine the test features with the corresponding part of the original dataframe
        df_test = df.loc[X_test.index]
        st.write("Number of rows in df_test:", len(df_test))

        embeddings_file = 'embeddings_test.npy'

        # Check if embeddings file exists
        if os.path.exists(embeddings_file):
            st.write("Embeddings file FOUND!")
            embeddings_array = np.load(embeddings_file)
        else:
            st.write("Embeddings file NOT FOUND!")
            df_test['combined'] = df_test['type'] + " " + df_test['amount'].astype(str) + " " + \
                                  df_test['oldbalanceOrg'].astype(str) + " " + df_test['newbalanceOrg'].astype(str) + " " + \
                                  df_test['oldbalanceDest'].astype(str) + " " + df_test['newbalanceDest'].astype(str) + " " + \
                                  df_test['creditCard'] + " " + df_test['creditCard'] + " " + df_test['timestamp'].astype(str) + " " + \
                                  X_test['largeAmount'].astype(str) + " " + \
                                  X_test['changebalanceOrig'].astype(str) + " " + \
                                  X_test['changebalanceDest'].astype(str) + " " + \
                                  X_test['countTransac'].astype(str)

            # For one-hot encoded features, add them as strings
            for col in type_one_hot.columns:
                if col in df_test.columns:
                    df_test['combined'] += " " + col + "_" + df_test[col].astype(str)

            combined_texts = df_test['combined'].tolist()
            embeddings_list = get_embeddings_parallel(combined_texts)

            # Flatten each embedding to make it 1D and convert to a 2D array
            embeddings_array = np.array([np.array(embedding).flatten() for embedding in embeddings_list])

            # Save the embeddings to a file for future use
            np.save(embeddings_file, embeddings_array)

        # Normalize the embeddings using StandardScaler
        scaler = StandardScaler()
        try:
            normalized_embeddings = scaler.fit_transform(embeddings_array)
        except ValueError as e:
            st.write("Error during normalization:", e)
            st.write("Shape of embeddings array before normalization:", embeddings_array.shape)

        # Calculate cosine dissimilarity matrix (1 - cosine similarity)
        cosine_dissimilarity_matrix = squareform(pdist(normalized_embeddings, 'cosine'))

        # Find the threshold for anomalies
        mean_dissimilarity = np.mean(cosine_dissimilarity_matrix)
        std_dissimilarity = np.std(cosine_dissimilarity_matrix)
        threshold = mean_dissimilarity + 1.5 * std_dissimilarity

        # Identify indices of the anomalies
        anomaly_indices = np.where(cosine_dissimilarity_matrix > threshold)

        # Since cosine_dissimilarity_matrix is a square matrix, we get pairs of indices
        anomaly_pairs = list(zip(anomaly_indices[0], anomaly_indices[1]))

        # Now we need to map these pairs back to our transactions
        anomaly_transactions = set()
        for i, j in anomaly_pairs:
            if i != j:
                anomaly_transactions.add(i)
                anomaly_transactions.add(j)

        # Create a mapping from new indices to original indices
        index_mapping = df_test.index.tolist()

        # Map the anomaly indices to original indices
        mapped_anomaly_indices = [index_mapping[i] for i in anomaly_transactions]

        # Add an 'embedding_cosine_isAnomaly' column to the DataFrame
        df_test['embedding_cosine_isAnomaly'] = 0
        df_test.loc[mapped_anomaly_indices, 'embedding_cosine_isAnomaly'] = 1

        # Print the threshold value and anomaly count
        st.write("Threshold for anomalies:", threshold)
        st.write("Count of anomalies:", len(anomaly_transactions))

        # Calculate Euclidean distance matrix
        euclidean_distance_matrix = squareform(pdist(normalized_embeddings, 'euclidean'))

        # Find anomalies in the Euclidean distance
        euclidean_mean_distance = np.mean(euclidean_distance_matrix)
        euclidean_std_distance = np.std(euclidean_distance_matrix)
        euclidean_threshold = euclidean_mean_distance + 1.5 * euclidean_std_distance

        # Identify indices of the anomalies based on Euclidean distance
        euclidean_anomaly_indices = np.where(euclidean_distance_matrix > euclidean_threshold)

        # Identify pairs of anomalies
        euclidean_anomaly_pairs = list(zip(euclidean_anomaly_indices[0], euclidean_anomaly_indices[1]))

        # Map these pairs back to our transactions
        euclidean_anomaly_transactions = set()
        for i, j in euclidean_anomaly_pairs:
            if i != j:
                euclidean_anomaly_transactions.add(i)
                euclidean_anomaly_transactions.add(j)

        # Map the Euclidean anomaly indices to original indices
        mapped_euclidean_anomaly_indices = [index_mapping[i] for i in euclidean_anomaly_transactions]

        # Add an 'embedding_euclidean_isAnomaly' column to the DataFrame
        df_test['embedding_euclidean_isAnomaly'] = 0
        df_test.loc[mapped_euclidean_anomaly_indices, 'embedding_euclidean_isAnomaly'] = 1

        # Print the threshold value and anomaly count
        st.write("Threshold for Euclidean distance anomalies:", euclidean_threshold)
        st.write("Count of Euclidean anomalies:", len(euclidean_anomaly_transactions))

        # Display confusion matrix as a heatmap
        st.subheader("Confusion Matrix")
        fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix')
        st.plotly_chart(fig)

        # Display model evaluation metrics
        st.subheader("Model Evaluation Metrics")
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        st.write(metrics)

        # Distribution of fraudulent vs non-fraudulent transactions
        st.subheader("Fraudulent vs Non-Fraudulent Transactions")
        fraud_counts = df['isFraud'].value_counts()
        fig_fraud = px.pie(fraud_counts, values=fraud_counts.values, names=['Non-Fraud', 'Fraud'], 
                            title='Distribution of Transactions', color_discrete_sequence=['#ff9999', '#66b3ff'])
        st.plotly_chart(fig_fraud)

        # Count of transactions per credit card
        st.subheader("Transactions per Credit Card")
        card_counts = df['creditCard'].value_counts()
        fig_card = px.bar(card_counts.head(10), title='Top 10 Credit Cards with Most Transactions',
                          labels={'index': 'Credit Card', 'value': 'Number of Transactions'},
                          color_discrete_sequence=['#ffcc99'])
        st.plotly_chart(fig_card)

        # Histogram of transaction amounts
        st.subheader("Transaction Amount Distribution")
        fig_hist = px.histogram(df, x='amount', title='Distribution of Transaction Amounts',
                                 color_discrete_sequence=['#99ff99'], nbins=30)
        st.plotly_chart(fig_hist)

        # Time series analysis of transactions over time
        st.subheader("Transactions Over Time")
        transactions_over_time = df.groupby(df['timestamp'].dt.date).size().reset_index(name='counts')
        fig_time_series = px.line(transactions_over_time, x='timestamp', y='counts', 
                                   title='Number of Transactions Over Time', markers=True, 
                                   color_discrete_sequence=['#ff6666'])
        st.plotly_chart(fig_time_series)

        # Correlation matrix heatmap
        st.subheader("Correlation Matrix")
        correlation_matrix = features.corr()
        fig_corr = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Blues', title='Feature Correlation Matrix')
        st.plotly_chart(fig_corr)

        # Box plot for transaction amounts by fraud status
        st.subheader("Box Plot of Transaction Amounts by Fraud Status")
        fig_box = px.box(df, x='isFraud', y='amount', title='Transaction Amounts by Fraud Status',
                         color='isFraud', color_discrete_sequence=['#66b3ff', '#ff9999'])
        st.plotly_chart(fig_box)

        # Heatmap for transaction counts by hour and day
        st.subheader("Transaction Counts by Hour and Day")
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        transactions_heatmap = df.groupby(['day_of_week', 'hour']).size().unstack().fillna(0)
        fig_heatmap = px.imshow(transactions_heatmap, color_continuous_scale='Viridis', title='Transaction Counts by Hour and Day')
        st.plotly_chart(fig_heatmap)

        # Feature importance from Random Forest
        st.subheader("Feature Importance")
        importance = rf_model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importance', orientation='h',
                                 color_discrete_sequence=['#66b3ff'])
        st.plotly_chart(fig_importance)

        # Show example anomalous transactions
        st.subheader("Example Anomalous Transactions")
        st.write(df_test[df_test['embedding_cosine_isAnomaly'] == 1].head())

    else:
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
