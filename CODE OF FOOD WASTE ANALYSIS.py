import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================================================
# FOOD WASTE ANALYSIS PROJECT
# Ready-to-run code for Excel dataset cleaning + analysis
# =========================================================

# ---------- 1. FILE PATH ----------
# Put your Excel file name here
FILE_PATH = "Food_waste_analysis.xlsx"   

# Output folder
OUTPUT_FOLDER = "food_waste_output"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)


# ---------- 2. LOAD DATA ----------
try:
    df = pd.read_excel(FILE_PATH)
    print("Excel file loaded successfully.")
except Exception as e:
    print("Error loading file:", e)
    raise

print("\nOriginal Dataset Preview:")
print(df.head())

print("\nColumn Names in Excel:")
print(list(df.columns))


# ---------- 3. STANDARDIZE COLUMN NAMES ----------
def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")

df.columns = [clean_column_name(col) for col in df.columns]

print("\nCleaned Column Names:")
print(list(df.columns))


# ---------- 4. REMOVE DUPLICATES ----------
before_dup = len(df)
df = df.drop_duplicates()
after_dup = len(df)
print(f"\nDuplicates removed: {before_dup - after_dup}")


# ---------- 5. HANDLE MISSING VALUES ----------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill object/text columns with mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna("Unknown")

# Fill numeric columns with median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ---------- 6. TRY TO IDENTIFY IMPORTANT COLUMNS ----------
# You can manually edit these if needed
def find_column(possible_names, columns):
    for p in possible_names:
        for c in columns:
            if p in c:
                return c
    return None

age_col = find_column(["age"], df.columns)
gender_col = find_column(["gender", "sex"], df.columns)
waste_type_col = find_column(["food_waste_type", "waste_type", "type_of_food_waste", "food_type", "waste_category"], df.columns)
frequency_col = find_column(["frequency", "how_often", "waste_frequency"], df.columns)
quantity_col = find_column(["quantity", "amount", "waste_amount", "food_waste_quantity"], df.columns)
reason_col = find_column(["reason", "cause", "why"], df.columns)

print("\nDetected Columns:")
print("Age Column:", age_col)
print("Gender Column:", gender_col)
print("Waste Type Column:", waste_type_col)
print("Frequency Column:", frequency_col)
print("Quantity Column:", quantity_col)
print("Reason Column:", reason_col)


# ---------- 7. CLEAN TEXT DATA ----------
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()


# ---------- 8. SAVE CLEANED DATA ----------
cleaned_file = f"{OUTPUT_FOLDER}/cleaned_food_waste_data.xlsx"
df.to_excel(cleaned_file, index=False)
print(f"\nCleaned data saved to: {cleaned_file}")


# ---------- 9. BASIC STATISTICAL SUMMARY ----------
summary_file = f"{OUTPUT_FOLDER}/statistical_summary.xlsx"
df.describe(include="all").to_excel(summary_file)
print(f"Statistical summary saved to: {summary_file}")


# ---------- 10. BAR GRAPH: FOOD WASTE DISTRIBUTION ----------
if waste_type_col is not None:
    plt.figure(figsize=(10, 6))
    df[waste_type_col].value_counts().plot(kind="bar")
    plt.title("Food Waste Distribution by Category")
    plt.xlabel("Food Waste Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_1_food_waste_distribution_bar.png")
    plt.show()
else:
    print("\nFood waste type column not found for bar graph.")


# ---------- 11. PIE CHART: FOOD WASTE PERCENTAGE ----------
if waste_type_col is not None:
    plt.figure(figsize=(8, 8))
    df[waste_type_col].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Percentage of Food Waste Types")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_2_food_waste_percentage_pie.png")
    plt.show()
else:
    print("\nFood waste type column not found for pie chart.")


# ---------- 12. LINE GRAPH: FOOD WASTE FREQUENCY ----------
if frequency_col is not None:
    freq_counts = df[frequency_col].value_counts()

    plt.figure(figsize=(10, 6))
    freq_counts.plot(kind="line", marker="o")
    plt.title("Food Waste Frequency Trend")
    plt.xlabel("Frequency")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_3_food_waste_frequency_line.png")
    plt.show()
else:
    print("\nFrequency column not found for line graph.")


# ---------- 13. AGE-WISE ANALYSIS ----------
if age_col is not None and waste_type_col is not None:
    plt.figure(figsize=(10, 6))
    age_waste = df.groupby(age_col)[waste_type_col].count()
    age_waste.plot(kind="bar")
    plt.title("Age-wise Food Waste Analysis")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_4_age_wise_food_waste_analysis.png")
    plt.show()
else:
    print("\nAge column or waste type column not found for age-wise analysis.")


# ---------- 14. GENDER-WISE ANALYSIS ----------
if gender_col is not None and waste_type_col is not None:
    plt.figure(figsize=(10, 6))
    gender_waste = df.groupby(gender_col)[waste_type_col].count()
    gender_waste.plot(kind="bar")
    plt.title("Gender-wise Food Waste Analysis")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_5_gender_wise_food_waste_analysis.png")
    plt.show()
else:
    print("\nGender column or waste type column not found for gender-wise analysis.")


# ---------- 15. REASON ANALYSIS ----------
if reason_col is not None:
    plt.figure(figsize=(10, 6))
    df[reason_col].value_counts().head(10).plot(kind="bar")
    plt.title("Top Reasons for Food Waste")
    plt.xlabel("Reason")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_6_top_reasons_for_food_waste.png")
    plt.show()
else:
    print("\nReason column not found for reason analysis.")


# ---------- 16. QUANTITY ANALYSIS ----------
if quantity_col is not None:
    # Try converting quantity to numeric if possible
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
    df[quantity_col] = df[quantity_col].fillna(df[quantity_col].median())

    plt.figure(figsize=(10, 6))
    plt.hist(df[quantity_col], bins=10, edgecolor="black")
    plt.title("Distribution of Food Waste Quantity")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/figure_7_food_waste_quantity_histogram.png")
    plt.show()
else:
    print("\nQuantity column not found for quantity analysis.")


# ---------- 17. CREATE FREQUENCY TABLES ----------
if waste_type_col is not None:
    waste_type_table = df[waste_type_col].value_counts().reset_index()
    waste_type_table.columns = ["Food Waste Type", "Count"]
    waste_type_table.to_excel(f"{OUTPUT_FOLDER}/table_food_waste_type_frequency.xlsx", index=False)
    print("Food waste type frequency table saved.")

if frequency_col is not None:
    frequency_table = df[frequency_col].value_counts().reset_index()
    frequency_table.columns = ["Frequency", "Count"]
    frequency_table.to_excel(f"{OUTPUT_FOLDER}/table_frequency_distribution.xlsx", index=False)
    print("Frequency distribution table saved.")

if reason_col is not None:
    reason_table = df[reason_col].value_counts().reset_index()
    reason_table.columns = ["Reason", "Count"]
    reason_table.to_excel(f"{OUTPUT_FOLDER}/table_reason_distribution.xlsx", index=False)
    print("Reason distribution table saved.")


# ---------- 18. FINAL REPORT TEXT ----------
print("\n========== PROJECT ANALYSIS COMPLETE ==========")
print("Generated files are saved inside folder:", OUTPUT_FOLDER)
print("1. Cleaned Excel dataset")
print("2. Statistical summary")
print("3. Bar chart")
print("4. Pie chart")
print("5. Line graph")
print("6. Age-wise analysis chart")
print("7. Gender-wise analysis chart")
print("8. Reasons analysis chart")
print("9. Quantity histogram")
print("10. Frequency tables")
print("==============================================")