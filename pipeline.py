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

# Horizontal Expansion using Tabs (Satisfies Step-2 Prompt)
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
        
        # --- Advanced Data Pre-processing (Unit 1 & 2 Fix) ---
        # This converts "5000 mAh Battery" to 5000.0 (Numerical)
        if 'battery' in df.columns:
            if df['battery'].dtype == 'object':
                df['battery'] = pd.to_numeric(df['battery'].str.extract('(\d+)', expand=False), errors='coerce')
                st.success("✅ 'battery' column converted to numerical values for mathematical modeling.")

        st.session_state.df = df
        st.write("Data Preview:", df.head())
        
        target_col = st.selectbox("Select Target Feature", df.columns)
        st.session_state.target = target_col
        
        features = st.multiselect("Select Features for PCA visualization", [col for col in df.columns if col != target_col])
        
        if len(features) >= 2:
            st.subheader("PCA Visualization (2D)")
            # Unsupervised Learning demonstration (Unit 2)
            temp_df = df[features].dropna()
            temp_df = pd.get_dummies(temp_df) 
            
            if not temp_df.empty:
                pca = PCA(n_components=2)
                components = pca.fit_transform(temp_df)
                
                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                fig = px.scatter(pca_df, x='PC1', y='PC2', title="Overall Data Shape (PCA)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough clean data for PCA. Please handle missing values in Tab 3.")

# --- Tab 2: EDA ---
with tab2:
    st.header("Exploratory Data Analysis")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("Dataset Description:")
        st.write(df.describe())
        
        st.write("Missing Values Count:")
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

# --- Tab 3: Data Engineering & Cleaning (Unit 1 & 2) ---
with tab3:
    st.header("Data Engineering & Outlier Removal")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Missing Value Handling
        st.subheader("Handle Missing Values (Imputation)")
        num_cols = df.select_dtypes(include=[np.number]).columns
        impute_strategy = st.selectbox("Imputation Method for numericals", ["mean", "median", "most_frequent"])
        
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=impute_strategy)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            st.session_state.df = df
            st.success(f"Missing values imputed using {impute_strategy}.")

        # Outlier Detection (Unit 2)
        st.subheader("Outlier Detection & Removal")
        outlier_method = st.selectbox("Select Method", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        if outlier_method != "None":
            if st.button("Detect Outliers"):
                outliers = np.zeros(len(df), dtype=bool)
                # Ensure numerical only for detection
                temp_df_num = df[num_cols].dropna()
                
                if outlier_method == "Isolation Forest":
                    clf = IsolationForest(random_state=42)
                    preds = clf.fit_predict(temp_df_num)
                    outliers[temp_df_num.index] = (preds == -1)
                elif outlier_method == "IQR":
                    Q1 = temp_df_num.quantile(0.25)
                    Q3 = temp_df_num.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[temp_df_num.index] = ((temp_df_num < (Q1 - 1.5 * IQR)) | (temp_df_num > (Q3 + 1.5 * IQR))).any(axis=1)
                
                st.warning(f"Detected {outliers.sum()} outliers.")
                
                if outliers.sum() > 0:
                    if st.button("Confirm: Delete Outliers from Dataset"):
                        st.session_state.df = df[~outliers]
                        st.success("Outliers removed successfully!")
    else:
        st.info("Please upload data in Tab 1.")

# --- Tab 4: Feature Selection (Unit 2) ---
with tab4:
    st.header("Feature Selection")
    if st.session_state.df is not None and st.session_state.target is not None:
        df = st.session_state.df
        target = st.session_state.target
        
        fs_method = st.selectbox("Select Selection Method", ["Variance Threshold", "Information Gain"])
        
        if st.button("Run Feature Selection"):
            num_cols = df.select_dtypes(include=[np.number]).columns
            X = df[num_cols].drop(columns=[target], errors='ignore').fillna(0)
            y = df[target] if target in num_cols else LabelEncoder().fit_transform(df[target].astype(str))
            
            if fs_method == "Variance Threshold":
                selector = VarianceThreshold(threshold=0.1)
                selector.fit(X)
                selected = X.columns[selector.get_support()]
                st.write("Selected Features based on Variance:", selected.tolist())
                    
            elif fs_method == "Information Gain":
                score_func = mutual_info_classif if problem_type == "Classification" else mutual_info_regression
                selector = SelectKBest(score_func=score_func, k='all')
                selector.fit(X, y)
                scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_}).sort_values(by='Score', ascending=False)
                st.write("Feature Importance Scores:")
                st.write(scores)
    else:
        st.info("Please upload data and select a target.")

# --- Tab 5: Data Split ---
with tab5:
    st.header("Train-Test Split")
    if st.session_state.df is not None:
        test_size = st.slider("Select Test Size %", 10, 50, 20) / 100.0
        
        if st.button("Split Data"):
            # Ensure model-ready format
            df_cleaned = st.session_state.df.dropna()
            df_encoded = pd.get_dummies(df_cleaned, drop_first=True) 
            target = st.session_state.target
            
            # Match target in encoded dataframe
            actual_target = [col for col in df_encoded.columns if target in col][0]
            
            X = df_encoded.drop(columns=[actual_target])
            y = df_encoded[actual_target]
            
            if problem_type == "Classification":
                y = LabelEncoder().fit_transform(y)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.session_state.split_data = (X_train, X_test, y_train, y_test)
            st.success(f"Split Complete: {len(X_train)} train samples, {len(X_test)} test samples.")
    else:
        st.info("Please upload data in Tab 1.")

# --- Tab 6: Model Selection & Tuning ---
with tab6:
    st.header("Select Model & Hyperparameter Tuning Strategy")
    model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "Random Forest", "KMeans (Clustering)"])
    tune_mode = st.radio("AutoML Tuning Mode", ["None", "GridSearch", "RandomSearch"])
    
    st.session_state.model_choice = model_choice
    st.session_state.tune_mode = tune_mode
    
    if model_choice == "SVM":
        st.session_state.svm_kernel = st.selectbox("Select SVM Kernel", ["linear", "rbf", "poly"])

# --- Tab 7: Training & Validation (Unit 1 & 3) ---
with tab7:
    st.header("Model Training & K-Fold Validation")
    k_folds = st.number_input("Value for K (Cross-Validation)", min_value=2, max_value=10, value=5)
    
    if st.button("Train Model Now"):
        if 'split_data' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.split_data
            model_choice = st.session_state.model_choice
            
            # Instantiate
            if problem_type == "Classification":
                if model_choice == "Linear/Logistic Regression": model = LogisticRegression(max_iter=1000)
                elif model_choice == "SVM": model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'))
                elif model_choice == "Random Forest": model = RandomForestClassifier()
                elif model_choice == "KMeans (Clustering)": model = KMeans(n_clusters=len(np.unique(y_train)))
            else:
                if model_choice == "Linear/Logistic Regression": model = LinearRegression()
                elif model_choice == "SVM": model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
                elif model_choice == "Random Forest": model = RandomForestRegressor()
                elif model_choice == "KMeans (Clustering)": st.error("KMeans not for regression"); model = None
            
            if model:
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                scoring = 'accuracy' if problem_type == "Classification" else 'neg_mean_squared_error'
                
                with st.spinner("Validating..."):
                    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                st.write(f"K-Fold Scores ({scoring}):", cv_results)
                st.write(f"Mean Score: **{np.mean(cv_results):.4f}**")
                
                # Hyperparameter Tuning (Unit 3)
                if st.session_state.tune_mode != "None" and model_choice == "Random Forest":
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
                    st.info(f"Running {st.session_state.tune_mode}...")
                    search = GridSearchCV(model, param_grid, cv=3) if st.session_state.tune_mode == "GridSearch" else RandomizedSearchCV(model, param_grid, cv=3)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.write("Best Hyperparameters:", search.best_params_)
                else:
                    model.fit(X_train, y_train)
                
                st.session_state.trained_model = model
                st.success("Training Successful!")
        else:
            st.error("Split your data in Tab 5 first.")

# --- Tab 8: Metrics (Unit 1 Evaluation) ---
with tab8:
    st.header("Performance Metrics & Error Analysis")
    if 'trained_model' in st.session_state and 'split_data' in st.session_state:
        model = st.session_state.trained_model
        X_train, X_test, y_train, y_test = st.session_state.split_data
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        c1, c2 = st.columns(2)
        if problem_type == "Classification":
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            c1.metric("Train Accuracy", f"{train_acc:.4f}")
            c2.metric("Test Accuracy", f"{test_acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, test_preds))
            if train_acc - test_acc > 0.15: st.warning("⚠️ High variance detected: Model is OVERFITTING.")
        else:
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            c1.metric("Train R2", f"{train_r2:.4f}")
            c2.metric("Test R2", f"{test_r2:.4f}")
            if train_r2 - test_r2 > 0.15: st.warning("⚠️ Model is OVERFITTING.")
    else:
        st.info("Train a model in Tab 7 to see metrics.")
