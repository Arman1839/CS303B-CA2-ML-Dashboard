import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-Learn Imports
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# --- UI Configuration ---
st.set_page_config(page_title="AutoML Flow Dashboard", layout="wide")
st.title("🚀 Professional ML Pipeline Dashboard")
st.markdown("---")

# --- Step 1: Problem Selection ---
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# Horizontal Expansion using Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Data Input", "2. EDA", "3. Cleaning", "4. Feature Selection", 
    "5. Split", "6. Model Selection & Tuning", "7. Training", "8. Metrics"
])

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None

# --- Tab 1: Input Data ---
with tab1:
    st.header("Data Input & PCA Visualization")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write("Data Preview:", df.head())
        
        target_col = st.selectbox("Select Target Feature", df.columns)
        st.session_state.target = target_col
        
        features = st.multiselect("Select Features for PCA visualization", [col for col in df.columns if col != target_col])
        
        if len(features) >= 2:
            st.subheader("PCA Visualization (2D)")
            # Basic imputation for PCA
            temp_df = df[features].dropna()
            temp_df = pd.get_dummies(temp_df) # Handle categorical for PCA
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(temp_df)
            
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            fig = px.scatter(pca_df, x='PC1', y='PC2', title="Overall Data Shape (PCA)")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: EDA ---
with tab2:
    st.header("Exploratory Data Analysis")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("Dataset Description:")
        st.write(df.describe())
        
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        
        # Correlation Heatmap (Numerical only)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.info("Please upload data in Tab 1.")

# --- Tab 3: Data Engineering & Cleaning ---
with tab3:
    st.header("Data Engineering & Outlier Removal")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Missing Value Handling
        st.subheader("Handle Missing Values")
        num_cols = df.select_dtypes(include=[np.number]).columns
        impute_strategy = st.selectbox("Imputation Method for numericals", ["mean", "median", "most_frequent"])
        
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=impute_strategy)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            st.session_state.df = df
            st.success(f"Missing values imputed using {impute_strategy}.")

        # Outlier Detection
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Select Method", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        if outlier_method != "None":
            if st.button("Detect Outliers"):
                outliers = np.zeros(len(df), dtype=bool)
                temp_df = df[num_cols].dropna()
                
                if outlier_method == "Isolation Forest":
                    clf = IsolationForest(random_state=42)
                    preds = clf.fit_predict(temp_df)
                    outliers[temp_df.index] = preds == -1
                
                # Show outlier count
                st.warning(f"Detected {outliers.sum()} outliers.")
                
                if outliers.sum() > 0:
                    remove = st.radio("Do you want to remove these outliers?", ["No", "Yes"])
                    if remove == "Yes":
                        st.session_state.df = df[~outliers]
                        st.success("Outliers removed successfully!")
    else:
        st.info("Please upload data in Tab 1.")

# --- Tab 4: Feature Selection ---
with tab4:
    st.header("Feature Selection")
    if st.session_state.df is not None and st.session_state.target is not None:
        df = st.session_state.df
        target = st.session_state.target
        
        fs_method = st.selectbox("Select Method", ["Variance Threshold", "Correlation", "Information Gain"])
        
        if st.button("Run Feature Selection"):
            num_cols = df.select_dtypes(include=[np.number]).columns
            X = df[num_cols].drop(columns=[target], errors='ignore')
            y = df[target] if target in num_cols else LabelEncoder().fit_transform(df[target].astype(str))
            
            if fs_method == "Variance Threshold":
                selector = VarianceThreshold(threshold=0.1)
                try:
                    selector.fit(X)
                    selected = X.columns[selector.get_support()]
                    st.write("Selected Features based on Variance:", selected.tolist())
                except Exception as e:
                    st.error("Error calculating variance.")
                    
            elif fs_method == "Information Gain":
                score_func = mutual_info_classif if problem_type == "Classification" else mutual_info_regression
                selector = SelectKBest(score_func=score_func, k='all')
                selector.fit(X.fillna(0), y)
                scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_}).sort_values(by='Score', ascending=False)
                st.write(scores)
    else:
        st.info("Please upload data and select a target.")

# --- Tab 5: Data Split ---
with tab5:
    st.header("Train-Test Split")
    if st.session_state.df is not None:
        test_size = st.slider("Select Test Size %", 10, 50, 20) / 100.0
        
        if st.button("Split Data"):
            df = st.session_state.df.dropna()
            # Basic preprocessing to ensure model runs
            df = pd.get_dummies(df, drop_first=True) 
            target = st.session_state.target
            
            # Re-find target in case get_dummies changed its name (not likely for numerical target, but safe)
            actual_target = [col for col in df.columns if target in col][0]
            
            X = df.drop(columns=[actual_target])
            y = df[actual_target]
            
            if problem_type == "Classification":
                y = LabelEncoder().fit_transform(y)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.session_state.split_data = (X_train, X_test, y_train, y_test)
            st.success(f"Data Split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    else:
        st.info("Please complete previous steps.")

# --- Tab 6: Model Selection & Tuning ---
with tab6:
    st.header("Select Model & Hyperparameter Tuning")
    model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "Random Forest", "KMeans (Clustering)"])
    
    tune_mode = st.radio("Hyperparameter Tuning Strategy", ["None", "GridSearch", "RandomSearch"])
    
    st.session_state.model_choice = model_choice
    st.session_state.tune_mode = tune_mode
    
    if model_choice == "SVM":
        st.session_state.svm_kernel = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])

# --- Tab 7: Training & Validation ---
with tab7:
    st.header("Model Training & K-Fold Validation")
    k_folds = st.number_input("Enter value for K (K-Fold Validation)", min_value=2, max_value=20, value=5)
    
    if st.button("Train Model"):
        if 'split_data' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.split_data
            model_choice = st.session_state.model_choice
            
            # Instantiate model
            if problem_type == "Classification":
                if model_choice == "Linear/Logistic Regression": model = LogisticRegression()
                elif model_choice == "SVM": model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'))
                elif model_choice == "Random Forest": model = RandomForestClassifier()
                elif model_choice == "KMeans (Clustering)": model = KMeans(n_clusters=len(np.unique(y_train))) # Unsupervised mapped
            else:
                if model_choice == "Linear/Logistic Regression": model = LinearRegression()
                elif model_choice == "SVM": model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
                elif model_choice == "Random Forest": model = RandomForestRegressor()
                elif model_choice == "KMeans (Clustering)": st.error("KMeans is not for regression"); model = None
            
            if model is not None:
                # K-Fold Validation
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                scoring = 'accuracy' if problem_type == "Classification" else 'neg_mean_squared_error'
                
                with st.spinner("Running K-Fold Validation..."):
                    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                
                st.write(f"K-Fold Results ({scoring}):", cv_results)
                st.write(f"Mean Score: **{np.mean(cv_results):.4f}**")
                
                # Hyperparameter Tuning Handling
                if st.session_state.tune_mode != "None" and model_choice == "Random Forest":
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
                    st.info("Applying tuning for Random Forest...")
                    if st.session_state.tune_mode == "GridSearch":
                        search = GridSearchCV(model, param_grid, cv=3)
                    else:
                        search = RandomizedSearchCV(model, param_grid, cv=3)
                    
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.write("Best Parameters:", search.best_params_)
                else:
                    model.fit(X_train, y_train)
                
                st.session_state.trained_model = model
                st.success("Model trained successfully! Move to Metrics tab.")
        else:
            st.error("Please split data in Tab 5 first.")

# --- Tab 8: Metrics ---
with tab8:
    st.header("Performance Metrics (Overfitting Check)")
    if 'trained_model' in st.session_state and 'split_data' in st.session_state:
        model = st.session_state.trained_model
        X_train, X_test, y_train, y_test = st.session_state.split_data
        
        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        if problem_type == "Classification":
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            col1.metric("Training Accuracy", f"{train_acc:.4f}")
            col2.metric("Testing Accuracy", f"{test_acc:.4f}")
            
            st.text("Classification Report (Test Data):")
            st.text(classification_report(y_test, test_preds))
            
            if train_acc - test_acc > 0.15:
                st.warning("⚠️ High difference between Train and Test accuracy. The model might be OVERFITTING.")
            elif train_acc < 0.60 and test_acc < 0.60:
                st.error("⚠️ Low accuracy on both sets. The model might be UNDERFITTING.")
            else:
                st.success("✅ Model appears to generalize well.")
                
        else: # Regression
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            col1.metric("Training R-Squared", f"{train_r2:.4f}")
            col2.metric("Testing R-Squared", f"{test_r2:.4f}")
            
            st.write(f"Test MSE: {mean_squared_error(y_test, test_preds):.4f}")
            
            if train_r2 - test_r2 > 0.15:
                st.warning("⚠️ High difference between Train and Test R2 score. The model might be OVERFITTING.")
            elif train_r2 < 0.50 and test_r2 < 0.50:
                st.error("⚠️ Low R2 scores on both sets. The model might be UNDERFITTING.")
            else:
                st.success("✅ Model appears to generalize well.")
    else:
        st.info("Train a model first in Tab 7.")