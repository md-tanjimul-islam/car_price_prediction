"""
eda.py
------
Exploratory data analysis. Prints a summary and saves charts to reports/figures/.

Run from the project root:   python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.preprocess import ROOT, clean_data, engineer_features, get_xy, load_data

FIG_DIR = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")


def save(name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close()
    print(f"  saved reports/figures/{name}.png")


def main():
    raw = load_data()
    print("== RAW DATA ==")
    print(raw.shape, "| missing values:", int(raw.isna().sum().sum()),
          "| duplicates:", int(raw.duplicated().sum()))
    print(raw.describe().T.round(2))

    df = clean_data(raw, verbose=True)
    X, y = get_xy(df)
    feats = engineer_features(X)
    df = df.join(feats[["Car_Age", "Brand", "Vehicle_Type"]])

    print("\n== CATEGORY COUNTS ==")
    for c in ["Fuel_Type", "Selling_type", "Transmission", "Owner", "Vehicle_Type", "Brand"]:
        print(df[c].value_counts().to_dict())

    print("\n== MEDIAN SELLING PRICE (lakh) ==")
    for c in ["Fuel_Type", "Transmission", "Selling_type", "Vehicle_Type"]:
        print(df.groupby(c)["Selling_Price"].median().round(2).to_dict())

    print("\nSaving charts...")
    # 1. target distribution (raw vs log)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=df, x="Selling_Price", bins=30, kde=True, ax=ax[0]).set_title("Selling price (lakh)")
    sns.histplot(data=df, x=np.log1p(df["Selling_Price"]), bins=30, kde=True, ax=ax[1]).set_title("log(1 + price)")
    save("01_target_distribution")

    # 2. numeric distributions
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, c in zip(ax, ["Present_Price", "Driven_kms", "Car_Age"]):
        sns.histplot(data=df, x=c, bins=30, kde=True, ax=a).set_title(c)
    save("02_numeric_distributions")

    # 3. outliers
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, c in zip(ax, ["Selling_Price", "Present_Price", "Driven_kms"]):
        sns.boxplot(y=df[c], ax=a).set_title(c)
    save("03_outliers_boxplots")

    # 4. categorical vs price
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, c in zip(ax.ravel(), ["Fuel_Type", "Selling_type", "Transmission",
                                 "Owner", "Vehicle_Type", "Brand"]):
        sns.boxplot(data=df, x=c, y="Selling_Price", ax=a)
        a.set_title(f"Price by {c}")
        a.tick_params(axis="x", rotation=45)
    save("04_categorical_vs_price")

    # 5. numeric vs price
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c in zip(ax, ["Present_Price", "Car_Age", "Driven_kms"]):
        sns.scatterplot(data=df, x=c, y="Selling_Price", hue="Vehicle_Type", ax=a)
        a.set_title(f"{c} vs Selling_Price")
    save("05_numeric_vs_price")

    # 6. log-log relationship (why a log target works well)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=np.log1p(df["Present_Price"]), y=np.log1p(df["Selling_Price"]),
                    hue=df["Car_Age"], palette="viridis")
    plt.title("log(Present price) vs log(Selling price)")
    save("06_log_log_relationship")

    # 7. correlation heatmap
    plt.figure(figsize=(7, 5.5))
    num = df[["Selling_Price", "Present_Price", "Driven_kms", "Car_Age", "Owner"]]
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    save("07_correlation_heatmap")

    # 8. average price by brand
    plt.figure(figsize=(8, 4.5))
    order = df.groupby("Brand")["Selling_Price"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="Brand", y="Selling_Price", order=order, estimator="mean", errorbar=None)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average selling price by brand (lakh)")
    save("08_price_by_brand")

    print("\nEDA done.")


if __name__ == "__main__":
    main()