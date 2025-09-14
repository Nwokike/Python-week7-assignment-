# --- Task 1: Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Make plots look nicer
sns.set(style="whitegrid")

# --- Task 1: Load and Explore the Dataset ---
try:
    # Example: using Iris dataset from seaborn
    df = sns.load_dataset("iris")  
    # If using your own dataset, replace with:
    # df = pd.read_csv("your_dataset.csv")
    
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the dataset path.")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Check structure
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()   # Or df.fillna(value, inplace=True)

# --- Task 2: Basic Data Analysis ---
print("\nBasic Statistics:")
print(df.describe())

# Example grouping: mean petal length per species
grouped = df.groupby("species")["petal_length"].mean()
print("\nAverage Petal Length per Species:")
print(grouped)

# --- Task 3: Data Visualization ---

# 1. Line Chart (trend over index, here we simulate with petal_length over sample index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal_length"], label="Petal Length", color="blue")
plt.title("Line Chart: Petal Length Over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8,5))
grouped.plot(kind="bar", color="orange")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df["sepal_length"], bins=15, color="green", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
plt.scatter(df["sepal_length"], df["petal_length"], alpha=0.7, c="red")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# --- Findings/Observations ---
print("\nObservations:")
print("- The Iris dataset has no missing values and 3 species categories.")
print("- Versicolor species tends to have intermediate petal lengths.")
print("- Sepal length distribution shows a normal-like spread.")
print("- Scatter plot shows a positive correlation between sepal length and petal length.")
