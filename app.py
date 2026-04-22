import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Food Waste Analysis Project", layout="wide")

st.title("🍽 Food Waste Analysis Project")
st.subheader("Working Model for Data Cleaning, Analysis, Visualization, and Prediction")

st.write("Upload your Excel or CSV dataset to begin analysis.")

uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])

def clean_column_name(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")

def find_column(possible_names, columns):
    for p in possible_names:
        for c in columns:
            if p in c:
                return c
    return None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully.")

        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]

        st.write("### Preview of Dataset")
        st.dataframe(df.head())

        st.write("### Dataset Information")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == "object":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

        st.write("### Missing Values After Cleaning")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

        # Detect important columns
        age_col = find_column(["age"], df.columns)
        gender_col = find_column(["gender", "sex"], df.columns)
        waste_type_col = find_column(["food_waste_type", "waste_type", "food_type", "waste_category"], df.columns)
        frequency_col = find_column(["frequency", "how_often", "waste_frequency"], df.columns)
        quantity_col = find_column(["quantity", "amount", "waste_amount", "food_waste_quantity"], df.columns)
        reason_col = find_column(["reason", "cause", "why"], df.columns)

        st.write("### Detected Columns")
        st.write("Age Column:", age_col)
        st.write("Gender Column:", gender_col)
        st.write("Waste Type Column:", waste_type_col)
        st.write("Frequency Column:", frequency_col)
        st.write("Quantity Column:", quantity_col)
        st.write("Reason Column:", reason_col)

        # ---------------------------
        # VISUALIZATIONS
        # ---------------------------
        st.write("## 📊 Data Visualizations")

        col1, col2 = st.columns(2)

        if waste_type_col:
            with col1:
                st.write("### Food Waste Distribution")
                fig, ax = plt.subplots()
                df[waste_type_col].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Food Waste Type")
                ax.set_ylabel("Count")
                ax.set_title("Food Waste Distribution")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.write("### Food Waste Percentage")
                fig, ax = plt.subplots()
                df[waste_type_col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Food Waste Percentage")
                st.pyplot(fig)

        if frequency_col:
            st.write("### Food Waste Frequency")
            fig, ax = plt.subplots()
            df[frequency_col].value_counts().plot(kind="line", marker="o", ax=ax)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Count")
            ax.set_title("Food Waste Frequency Trend")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        if age_col and waste_type_col:
            st.write("### Age-wise Food Waste Analysis")
            fig, ax = plt.subplots()
            df.groupby(age_col)[waste_type_col].count().plot(kind="bar", ax=ax)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.set_title("Age-wise Food Waste Analysis")
            st.pyplot(fig)

        if gender_col and waste_type_col:
            st.write("### Gender-wise Food Waste Analysis")
            fig, ax = plt.subplots()
            df.groupby(gender_col)[waste_type_col].count().plot(kind="bar", ax=ax)
            ax.set_xlabel("Gender")
            ax.set_ylabel("Count")
            ax.set_title("Gender-wise Food Waste Analysis")
            st.pyplot(fig)

        if reason_col:
            st.write("### Top Reasons for Food Waste")
            fig, ax = plt.subplots()
            df[reason_col].value_counts().head(10).plot(kind="bar", ax=ax)
            ax.set_xlabel("Reason")
            ax.set_ylabel("Count")
            ax.set_title("Top Reasons")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ---------------------------
        # MACHINE LEARNING MODEL
        # ---------------------------
        st.write("## 🤖 Food Waste Prediction Model")

        model_df = df.copy()

        # Create target column automatically if not available
        target_col = find_column(["waste_risk", "target", "label"], model_df.columns)

        if target_col is None:
            if quantity_col and quantity_col in model_df.columns:
                model_df[quantity_col] = pd.to_numeric(model_df[quantity_col], errors="coerce")
                threshold = model_df[quantity_col].median()
                model_df["waste_risk"] = np.where(model_df[quantity_col] > threshold, "High", "Low")
                target_col = "waste_risk"
            elif frequency_col and frequency_col in model_df.columns:
                high_words = ["daily", "often", "frequently", "high", "always"]
                model_df["waste_risk"] = model_df[frequency_col].astype(str).apply(
                    lambda x: "High" if any(word in x.lower() for word in high_words) else "Low"
                )
                target_col = "waste_risk"

        if target_col:
            st.write("Target Column Used:", target_col)

            # Encode categorical columns
            label_encoders = {}
            for col in model_df.columns:
                if model_df[col].dtype == "object":
                    le = LabelEncoder()
                    model_df[col] = le.fit_transform(model_df[col].astype(str))
                    label_encoders[col] = le

            X = model_df.drop(columns=[target_col])
            y = model_df[target_col]

            if len(X.columns) > 0 and len(model_df) > 5:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"Model trained successfully. Accuracy: {acc:.2f}")

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

                st.write("### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.write("## 🔍 Predict New Entry")

                input_data = {}
                input_cols = X.columns.tolist()

                cols_ui = st.columns(2)
                idx = 0

                for col in input_cols:
                    original_col = col
                    if original_col in label_encoders:
                        options = list(label_encoders[original_col].classes_)
                        selected = cols_ui[idx % 2].selectbox(f"{original_col}", options)
                        input_data[original_col] = label_encoders[original_col].transform([selected])[0]
                    else:
                        min_val = float(X[original_col].min())
                        max_val = float(X[original_col].max())
                        mean_val = float(X[original_col].mean())
                        input_data[original_col] = cols_ui[idx % 2].number_input(
                            f"{original_col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val
                        )
                    idx += 1

                if st.button("Predict Food Waste Risk"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]

                    if target_col in label_encoders:
                        prediction_label = label_encoders[target_col].inverse_transform([prediction])[0]
                    else:
                        prediction_label = prediction

                    st.success(f"Predicted Food Waste Risk: {prediction_label}")

            else:
                st.warning("Not enough usable columns/data for model training.")
        else:
            st.warning("Could not create or detect a target column for prediction.")

        # Download cleaned data
        st.write("## 📥 Download Cleaned Dataset")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_food_waste_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")