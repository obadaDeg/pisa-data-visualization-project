import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from scipy import stats
import warnings


os.makedirs("outputs/plots", exist_ok=True)


student_df = pd.read_csv("dataset/pisa.csv")
print(f"Dataset shape: {student_df.shape}")


student_df.head()


student_df.describe().to_csv("outputs/pisa_describe.csv")
student_df.describe().info()


def memory_usage(pandas_obj):
    """Calculate memory usage of a pandas object in MB"""
    if isinstance(pandas_obj, pd.DataFrame):
        usage_bytes = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_bytes = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_bytes / (1024 * 1024)
    return usage_mb


def optimize_floats(df):
    """Optimize float dtypes by downcasting to float32 where possible"""
    float_cols = df.select_dtypes(include=["float64"]).columns

    for col in float_cols:

        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def optimize_ints(df):
    """Optimize integer dtypes by downcasting to smallest possible integer type"""
    int_cols = df.select_dtypes(include=["int64"]).columns

    for col in int_cols:

        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def optimize_categorical(df, categorical_threshold=0.5, excluded_cols=None):
    """Convert columns with low cardinality to categorical type"""
    if excluded_cols is None:
        excluded_cols = []

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        if col not in excluded_cols:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < categorical_threshold:
                df[col] = df[col].astype("category")

    int_cols = df.select_dtypes(include=["int"]).columns
    for col in int_cols:
        if col not in excluded_cols:
            num_unique_values = len(df[col].unique())
            if num_unique_values < 50:
                df[col] = df[col].astype("category")

    return df


def optimize_known_pisa_columns(df):
    """Apply specific optimizations for known PISA data columns"""

    single_value_cols = ["CYC"]
    for col in single_value_cols:
        if col in df.columns:

            df[col] = df[col].astype("category")

    categorical_cols = [
        "CNT",
        "CNTRYID",
        "SUBNATIO",
        "LANGTEST_QQQ",
        "LANGTEST_COG",
        "LANGTEST_PAQ",
        "ISCEDP",
        "COBN_S",
        "COBN_M",
        "COBN_F",
        "LANGN",
        "REGION",
        "OECD",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    pv_cols = [col for col in df.columns if col.startswith("PV")]
    for col in pv_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


student_df = optimize_floats(student_df)
student_df = optimize_ints(student_df)
student_df = optimize_categorical(student_df)
student_df = optimize_known_pisa_columns(student_df)


print(f"Memory usage after optimization: {memory_usage(student_df)} MB")


student_df.info()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


os.makedirs("outputs/cleaning", exist_ok=True)


def check_duplicates(df, id_column="CNTSTUID"):
    """
    Check and remove duplicate student IDs

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    id_column : str, default='CNTSTUID'
        Column containing unique student identifiers

    Returns:
    --------
    pandas DataFrame
        DataFrame with duplicates removed
    """
    print(f"\nChecking for duplicate student IDs in column '{id_column}'...")
    if id_column in df.columns:
        duplicate_ids = df[id_column].duplicated().sum()
        if duplicate_ids > 0:
            print(f"Found {duplicate_ids} duplicate student IDs. Removing duplicates.")
            df = df.drop_duplicates(subset=id_column, keep="first")
        else:
            print("No duplicate student IDs found.")
    else:
        print(f"Warning: {id_column} column not found. Skipping duplicate check.")

    return df


def analyze_missing_values(df, key_variables):
    """
    Analyze and report missing values in key variables

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    key_variables : list
        List of key variables to analyze

    Returns:
    --------
    pandas DataFrame
        DataFrame with missing value report
    """
    print("\nAnalyzing missing values in key variables...")

    existing_vars = [col for col in key_variables if col in df.columns]

    missing_counts = df[existing_vars].isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    missing_report = pd.DataFrame(
        {"Missing Count": missing_counts, "Missing Percentage": missing_percentages}
    ).sort_values("Missing Percentage", ascending=False)

    print("Missing values in key variables:")
    print(missing_report[missing_report["Missing Count"] > 0])

    missing_report.to_csv("outputs/cleaning/missing_values_report.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df[existing_vars].isnull(), cbar=False, yticklabels=False, cmap="viridis"
    )
    plt.title("Missing Values in Key Variables")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("outputs/cleaning/missing_values_heatmap.png")
    plt.close()

    return existing_vars, missing_report


def handle_missing_performance(df, perf_variables):
    """
    Remove students with missing performance data

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    perf_variables : list
        List of performance variables

    Returns:
    --------
    pandas DataFrame
        DataFrame with rows containing missing performance removed
    """
    print("\nHandling missing performance data...")

    existing_perf_vars = [col for col in perf_variables if col in df.columns]

    if not existing_perf_vars:
        print("No performance variables found. Skipping.")
        return df

    missing_perf = df[existing_perf_vars].isnull().any(axis=1)
    if missing_perf.sum() > 0:
        print(
            f"Removing {missing_perf.sum()} students with missing performance scores."
        )
        df = df.drop(df[missing_perf].index)
    else:
        print("No missing performance data found.")

    return df


def remove_excessive_missing(df, key_variables, threshold=0.5):
    """
    Remove rows with excessive missing values

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    key_variables : list
        List of key variables to check
    threshold : float, default=0.5
        Minimum proportion of key variables required

    Returns:
    --------
    pandas DataFrame
        DataFrame with rows containing excessive missing values removed
    """
    print("\nRemoving rows with excessive missing values...")

    existing_vars = [col for col in key_variables if col in df.columns]

    key_vars_present = df[existing_vars].count(axis=1)
    min_vars_required = len(existing_vars) * threshold
    excessive_missing = key_vars_present < min_vars_required

    if excessive_missing.sum() > 0:
        print(
            f"Removing {excessive_missing.sum()} students with more than {int((1-threshold)*100)}% of key variables missing."
        )
        df = df.drop(df[excessive_missing].index)
    else:
        print("No rows with excessive missing values found.")

    return df


def impute_demographic_variables(df, demo_variables):
    """
    Impute missing values in demographic variables

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    demo_variables : list
        List of demographic variables to impute

    Returns:
    --------
    pandas DataFrame
        DataFrame with imputed demographic variables
    """
    print("\nImputing missing values in demographic variables...")

    existing_demo_vars = [col for col in demo_variables if col in df.columns]

    for col in existing_demo_vars:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:

            if df[col].dtype == "category" or df[col].nunique() < 10:
                most_frequent = df[col].mode()[0]
                df[col] = df[col].fillna(most_frequent)
                print(
                    f"Filled {missing_count} missing values in {col} with most frequent value: {most_frequent}"
                )
            else:

                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(
                    f"Filled {missing_count} missing values in {col} with median: {median_val}"
                )

    return df


def impute_ses_variables(df, ses_variables):
    """
    Impute missing values in SES variables using KNN or median

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    ses_variables : list
        List of SES variables

    Returns:
    --------
    pandas DataFrame
        DataFrame with imputed SES variables
    """
    print("\nImputing missing values in SES variables...")

    if "ESCS" not in df.columns:
        print("ESCS variable not found. Skipping SES imputation.")
        return df

    missing_count = df["ESCS"].isnull().sum()
    if missing_count == 0:
        print("No missing values in ESCS. Skipping SES imputation.")
        return df

    other_ses_vars = [
        col for col in ses_variables if col in df.columns and col != "ESCS"
    ]

    if len(other_ses_vars) > 0:

        try:

            impute_cols = other_ses_vars + ["ESCS"]
            impute_df = df[impute_cols].copy()

            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(impute_df)

            df.loc[df["ESCS"].isnull(), "ESCS"] = imputed_values[
                df["ESCS"].isnull(), -1
            ]
            print(f"Imputed {missing_count} missing ESCS values using KNN imputation.")
        except Exception as e:

            print(f"KNN imputation failed: {e}")
            median_escs = df["ESCS"].median()
            df["ESCS"] = df["ESCS"].fillna(median_escs)
            print(
                f"Imputed {missing_count} missing ESCS values with median: {median_escs}"
            )
    else:

        median_escs = df["ESCS"].median()
        df["ESCS"] = df["ESCS"].fillna(median_escs)
        print(f"Imputed {missing_count} missing ESCS values with median: {median_escs}")

    return df


def impute_school_variables(df, school_variables):
    """
    Impute missing values in school variables with median

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    school_variables : list
        List of school variables

    Returns:
    --------
    pandas DataFrame
        DataFrame with imputed school variables
    """
    print("\nImputing missing values in school variables...")

    existing_school_vars = [col for col in school_variables if col in df.columns]

    for col in existing_school_vars:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Filled {missing_count} missing values in {col} with median.")

    return df


def identify_continuous_variables(df, all_variables):
    """
    Identify continuous variables in the dataset

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    all_variables : list
        List of variables to check

    Returns:
    --------
    list
        List of identified continuous variables
    """
    continuous_vars = []

    for col in all_variables:
        if col in df.columns:
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                if df[col].nunique() > 10:
                    continuous_vars.append(col)

    return continuous_vars


def handle_outliers(df, continuous_vars, perf_vars):
    """
    Detect and handle outliers in continuous variables

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    continuous_vars : list
        List of continuous variables to check for outliers
    perf_vars : list
        List of performance variables to exclude

    Returns:
    --------
    pandas DataFrame, DataFrame
        DataFrame with handled outliers and outlier report
    """
    print("\nDetecting and handling outliers in key continuous variables...")

    outlier_report = {}

    for col in continuous_vars:

        if col in perf_vars:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        iqr_lower_bound = Q1 - 1.5 * IQR
        iqr_upper_bound = Q3 + 1.5 * IQR

        mean_val = df[col].mean()
        std_val = df[col].std()
        z_lower_bound = mean_val - 2.5 * std_val
        z_upper_bound = mean_val + 2.5 * std_val

        lower_bound = max(iqr_lower_bound, z_lower_bound)
        upper_bound = min(iqr_upper_bound, z_upper_bound)

        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        all_outliers = lower_outliers | upper_outliers
        outlier_count = all_outliers.sum()

        if outlier_count > 0:
            outlier_report[col] = {
                "Count": outlier_count,
                "Percentage": (outlier_count / len(df)) * 100,
                "Min Outlier": df.loc[all_outliers, col].min(),
                "Max Outlier": df.loc[all_outliers, col].max(),
                "IQR Range": f"[{iqr_lower_bound:.2f}, {iqr_upper_bound:.2f}]",
                "Z-score Range": f"[{z_lower_bound:.2f}, {z_upper_bound:.2f}]",
                "Final Range": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
            }

            if upper_outliers.sum() > 0:
                print(
                    f"Capping {upper_outliers.sum()} upper outliers in {col} at {upper_bound:.2f}"
                )
                df.loc[upper_outliers, col] = upper_bound

            if lower_outliers.sum() > 0:
                print(
                    f"Capping {lower_outliers.sum()} lower outliers in {col} at {lower_bound:.2f}"
                )
                df.loc[lower_outliers, col] = lower_bound

    outlier_df = pd.DataFrame(outlier_report).T

    if not outlier_df.empty:
        print("\nOutliers detected and handled:")
        print(outlier_df)
        outlier_df.to_csv("outputs/cleaning/outlier_report.csv")
    else:
        print("No outliers detected in continuous variables.")

    return df, outlier_df


def handle_age_outliers(df):
    """
    Handle age outliers specific to PISA target population

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset

    Returns:
    --------
    pandas DataFrame
        DataFrame with age outliers removed
    """
    print("\nHandling age outliers...")

    if "AGE" not in df.columns:
        print("AGE variable not found. Skipping age outlier handling.")
        return df

    age_outliers = (df["AGE"] < 14) | (df["AGE"] > 16.5)

    if age_outliers.sum() > 0:
        print(
            f"Removing {age_outliers.sum()} students outside the target age range (14-16.5)."
        )
        df = df.drop(df[age_outliers].index)
    else:
        print("No age outliers found.")

    return df


def standardize_categorical_vars(df, cat_vars):
    """
    Standardize categorical variables and handle rare categories

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    cat_vars : list
        List of categorical variables to standardize

    Returns:
    --------
    pandas DataFrame
        DataFrame with standardized categorical variables
    """
    print("\nStandardizing categorical variables...")

    existing_cat_vars = [col for col in cat_vars if col in df.columns]

    for col in existing_cat_vars:
        if df[col].nunique() < 50:

            val_counts = df[col].value_counts()

            rare_cats = val_counts[val_counts / len(df) < 0.001].index.tolist()

            if rare_cats:
                print(
                    f"Column {col}: Consolidating {len(rare_cats)} rare categories into 'Other'"
                )

                if df[col].dtype != "category":
                    df[col] = df[col].astype("category")

                if "Other" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(["Other"])

                df.loc[df[col].isin(rare_cats), col] = "Other"

    return df


def create_derived_variables(df):
    """
    Create derived variables that might be useful for analysis

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset

    Returns:
    --------
    pandas DataFrame
        DataFrame with added derived variables
    """
    print("\nCreating derived variables...")

    if "AGE" in df.columns:
        df["AGE_GROUP"] = pd.cut(
            df["AGE"],
            bins=[0, 12, 14, 16, 18, 100],
            labels=["Under 12", "12-14", "14-16", "16-18", "Over 18"],
        )
        print("Created AGE_GROUP variable.")

    if "ESCS" in df.columns:
        df["ESCS_QUINTILE"] = pd.qcut(
            df["ESCS"], q=5, labels=["Lowest", "Low", "Middle", "High", "Highest"]
        )
        print("Created ESCS_QUINTILE variable.")

    for pv in ["PV1MATH", "PV1READ", "PV1SCIE"]:
        if pv in df.columns:

            var_name = f"{pv}_LEVEL"

            if pv == "PV1MATH":
                bins = [0, 358, 420, 482, 545, 607, 669, 1000]
            elif pv == "PV1READ":
                bins = [0, 335, 407, 480, 553, 626, 698, 1000]
            else:
                bins = [0, 335, 410, 484, 559, 633, 708, 1000]

            df[var_name] = pd.cut(
                df[pv],
                bins=bins,
                labels=[
                    "Below 1",
                    "Level 1",
                    "Level 2",
                    "Level 3",
                    "Level 4",
                    "Level 5",
                    "Level 6",
                ],
            )
            print(f"Created {var_name} variable.")

            high_cutoff = bins[4]
            df[f"{pv}_HIGH"] = (df[pv] >= high_cutoff).astype(int)
            print(f"Created {pv}_HIGH indicator for scores ≥ {high_cutoff} (Level 4+).")

    return df


def check_extreme_performance(df, perf_vars):
    """
    Check for extreme performance values and remove them

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset
    perf_vars : list
        List of performance variables to check

    Returns:
    --------
    pandas DataFrame
        DataFrame with extreme performance values removed
    """
    print("\nChecking for extreme performance values...")

    existing_perf_vars = [col for col in perf_vars if col in df.columns]

    for pv in existing_perf_vars:

        extreme_scores = (df[pv] < 100) | (df[pv] > 900)

        if extreme_scores.sum() > 0:
            print(f"WARNING: Found {extreme_scores.sum()} extreme values in {pv}.")
            print(f"Removing {extreme_scores.sum()} rows with extreme {pv} values.")
            df = df.drop(df[extreme_scores].index)

    return df


def check_implausible_relationships(df):
    """
    Check for implausible relationships between ESCS and performance

    Parameters:
    -----------
    df : pandas DataFrame
        The PISA dataset

    Returns:
    --------
    pandas DataFrame
        DataFrame with flagged implausible relationships
    """
    print("\nChecking for implausible relationships...")

    if "ESCS" in df.columns and "PV1MATH" in df.columns:

        X = df[["ESCS"]]
        y = df["PV1MATH"]
        reg = LinearRegression().fit(X, y)
        df["MATH_RESIDUAL"] = y - reg.predict(X)

        residual_std = df["MATH_RESIDUAL"].std()
        extreme_residuals = np.abs(df["MATH_RESIDUAL"]) > 3 * residual_std

        if extreme_residuals.sum() > 0:
            print(
                f"Flagged {extreme_residuals.sum()} rows with extreme ESCS-Math relationships."
            )

            df["EXTREME_RESIDUAL"] = extreme_residuals.astype(int)
            print("Added 'EXTREME_RESIDUAL' flag for these observations.")
    else:
        print("ESCS or PV1MATH variables not found. Skipping relationship check.")

    return df


def create_data_reports(df, continuous_vars, cat_vars, orig_size):
    """
    Create final data quality reports

    Parameters:
    -----------
    df : pandas DataFrame
        The cleaned PISA dataset
    continuous_vars : list
        List of continuous variables
    cat_vars : list
        List of categorical variables
    orig_size : int
        Original size of the dataset before cleaning

    Returns:
    --------
    dict
        Dictionary with cleaning summary statistics
    """
    print("\nCreating final dataset reports...")

    rows_removed = orig_size - len(df)

    key_demographics = ["AGE", "GRADE", "ST004D01T", "IMMIG"]
    key_performance = ["PV1MATH", "PV1READ", "PV1SCIE"]
    key_ses = ["ESCS", "HOMEPOS", "HISCED"]
    key_school = ["SCHSUST", "DISCLIM", "TEACHSUP", "BELONG"]

    all_key_vars = key_demographics + key_performance + key_ses + key_school
    existing_key_vars = [col for col in all_key_vars if col in df.columns]

    cleaning_report = {
        "Original rows": orig_size,
        "Rows after cleaning": len(df),
        "Rows removed": rows_removed,
        "Percentage removed": (rows_removed / orig_size) * 100,
        "Missing values after": df[existing_key_vars].isnull().sum().sum(),
        "Variables modified": len(existing_key_vars),
        "New variables created": sum(
            [
                "AGE_GROUP" in df.columns,
                "ESCS_QUINTILE" in df.columns,
                "PV1MATH_LEVEL" in df.columns,
                "PV1READ_LEVEL" in df.columns,
                "PV1SCIE_LEVEL" in df.columns,
                "PV1MATH_HIGH" in df.columns,
                "PV1READ_HIGH" in df.columns,
                "PV1SCIE_HIGH" in df.columns,
            ]
        ),
    }

    print("\nCleaning summary:")
    for key, value in cleaning_report.items():
        print(f"{key}: {value}")

    pd.DataFrame([cleaning_report]).to_csv(
        "outputs/cleaning/cleaning_summary.csv", index=False
    )

    df.head().to_csv("outputs/cleaning/sample_head.csv")

    existing_cont_vars = [col for col in continuous_vars if col in df.columns]
    if existing_cont_vars:
        df[existing_cont_vars].describe().to_csv(
            "outputs/cleaning/continuous_vars_summary.csv"
        )

    cat_vars_extended = cat_vars + [
        "AGE_GROUP",
        "ESCS_QUINTILE",
        "PV1MATH_LEVEL",
        "PV1READ_LEVEL",
        "PV1SCIE_LEVEL",
    ]
    cat_vars_extended = [v for v in cat_vars_extended if v in df.columns]

    for var in cat_vars_extended:
        if var in df.columns:
            df[var].value_counts().to_csv(f"outputs/cleaning/{var}_distribution.csv")

    return cleaning_report


def clean_pisa_data(df, save_path=None, inplace=True):
    """
    Comprehensive data cleaning for PISA dataset by calling modular functions

    Parameters:
    -----------
    df : pandas DataFrame
        The raw PISA dataset
    save_path : str, optional
        Path to save the cleaned dataset
    inplace : bool, default=True
        Whether to modify the original DataFrame or return a copy

    Returns:
    --------
    pandas DataFrame
        The cleaned PISA dataset (same as input if inplace=True)
    """
    if not inplace:
        df = df.copy()

    print(f"Starting data cleaning process on DataFrame with shape: {df.shape}")
    original_size = len(df)

    key_demographics = ["AGE", "GRADE", "ST004D01T", "IMMIG"]
    key_performance = ["PV1MATH", "PV1READ", "PV1SCIE"]
    key_ses = ["ESCS", "HOMEPOS", "HISCED"]
    key_school = ["SCHSUST", "DISCLIM", "TEACHSUP", "BELONG"]
    all_key_vars = key_demographics + key_performance + key_ses + key_school

    df = check_duplicates(df)

    existing_key_vars, missing_report = analyze_missing_values(df, all_key_vars)

    df = handle_missing_performance(df, key_performance)

    df = remove_excessive_missing(df, existing_key_vars)

    df = impute_demographic_variables(df, key_demographics)

    df = impute_ses_variables(df, key_ses)

    df = impute_school_variables(df, key_school)

    continuous_vars = identify_continuous_variables(df, existing_key_vars)

    df, outlier_report = handle_outliers(df, continuous_vars, key_performance)

    df = handle_age_outliers(df)

    cat_vars = [
        col
        for col in existing_key_vars
        if col not in continuous_vars and df[col].dtype != "category"
    ]
    df = standardize_categorical_vars(df, cat_vars)

    df = create_derived_variables(df)

    df = check_extreme_performance(df, key_performance)

    df = check_implausible_relationships(df)

    cleaning_report = create_data_reports(df, continuous_vars, cat_vars, original_size)

    if save_path:
        print(f"\nSaving cleaned dataset to {save_path}")
        df.to_csv(save_path, index=False)

    print("\nData cleaning completed successfully.")
    return df


student_df = clean_pisa_data(student_df, save_path="outputs/pisa_cleaned.csv")


def plot_performance_distributions(df):
    """Plot histograms of PISA performance scores"""
    performance_vars = ["PV1MATH", "PV1READ", "PV1SCIE"]
    titles = ["Mathematics", "Reading", "Science"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (var, title) in enumerate(zip(performance_vars, titles)):
        sns.histplot(df[var], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {title} Scores")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")

        mean_val = df[var].mean()
        axes[i].axvline(mean_val, color="red", linestyle="--")
        axes[i].text(
            mean_val + 10,
            0.8 * axes[i].get_ylim()[1],
            f"Mean: {mean_val:.1f}",
            color="red",
        )

    plt.tight_layout()
    plt.savefig("outputs/plots/performance_distributions.png")
    plt.show()


plot_performance_distributions(student_df)


def plot_escs_distribution(df):
    """Plot histogram of ESCS index"""
    plt.figure(figsize=(12, 6))

    sns.histplot(df["ESCS"], kde=True)
    plt.title("Distribution of Economic, Social and Cultural Status (ESCS) Index")
    plt.xlabel("ESCS Index")
    plt.ylabel("Frequency")

    mean_val = df["ESCS"].mean()
    median_val = df["ESCS"].median()

    plt.axvline(mean_val, color="red", linestyle="--")
    plt.axvline(median_val, color="green", linestyle="-.")

    plt.text(mean_val + 0.1, 0.8 * plt.ylim()[1], f"Mean: {mean_val:.2f}", color="red")
    plt.text(
        median_val + 0.1,
        0.7 * plt.ylim()[1],
        f"Median: {median_val:.2f}",
        color="green",
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/escs_distribution.png")
    plt.show()


plot_escs_distribution(student_df)


student_df["gender"] = student_df["ST004D01T"].map({1: "Male", 2: "Female"})


def plot_gender_distribution(df):
    """Plot count of students by gender"""
    plt.figure(figsize=(8, 6))

    sns.countplot(x="gender", data=df)
    plt.title("Distribution of Students by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")

    for p in plt.gca().patches:
        plt.gca().annotate(
            f"{p.get_height():,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig("outputs/plots/gender_distribution.png")
    plt.show()


plot_gender_distribution(student_df)


def plot_top_countries_math(df, top_n=15):
    """Plot bar chart of top countries by math performance"""

    country_math = df.groupby("CNT")["PV1MATH"].mean().sort_values(ascending=False)

    top_countries = country_math.head(top_n)

    plt.figure(figsize=(12, 8))

    plt.barh(y=range(len(top_countries)), width=top_countries.values, color="steelblue")

    plt.yticks(range(len(top_countries)), top_countries.index)

    plt.title(f"Top {top_n} Countries by Mathematics Performance")
    plt.xlabel("Average Math Score")
    plt.ylabel("Country")

    global_avg = df["PV1MATH"].mean()
    plt.axvline(global_avg, color="red", linestyle="--")
    plt.text(
        global_avg + 5,
        len(top_countries) / 2,
        f"Global Average: {global_avg:.1f}",
        color="red",
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/top_countries_math.png")
    plt.show()


plot_top_countries_math(student_df)


def plot_escs_math_relationship(df, sample_size=5000):
    """Plot scatter plot of ESCS vs Math performance"""

    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df

    plt.figure(figsize=(12, 8))

    sns.regplot(
        x="ESCS",
        y="PV1MATH",
        data=sample_df,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
    )

    plt.title("Relationship between Socioeconomic Status and Math Performance")
    plt.xlabel("ESCS (Economic, Social and Cultural Status Index)")
    plt.ylabel("Math Score")

    corr = df["ESCS"].corr(df["PV1MATH"])
    plt.annotate(
        f"Correlation: {corr:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/escs_math_relationship.png")
    plt.show()


plot_escs_math_relationship(student_df)


def plot_performance_by_gender(df):
    """Plot box plots of performance by gender"""

    perf_vars = ["PV1MATH", "PV1READ", "PV1SCIE"]
    perf_titles = ["Mathematics", "Reading", "Science"]

    plt.figure(figsize=(15, 6))

    for i, (var, title) in enumerate(zip(perf_vars, perf_titles)):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x="gender", y=var, data=df)
        plt.title(f"{title} Performance by Gender")
        plt.xlabel("Gender")
        plt.ylabel(f"{title} Score")

    plt.tight_layout()
    plt.savefig("outputs/plots/performance_by_gender.png")
    plt.show()


plot_performance_by_gender(student_df)


def plot_performance_by_immigrant(df):
    """Plot box plots of performance by immigrant status"""

    plt.figure(figsize=(15, 6))

    for i, (var, title) in enumerate(
        zip(["PV1MATH", "PV1READ", "PV1SCIE"], ["Mathematics", "Reading", "Science"])
    ):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x="immigrant_status", y=var, data=df)
        plt.title(f"{title} Performance by Immigrant Status")
        plt.xlabel("Immigrant Status")
        plt.ylabel(f"{title} Score")

    plt.tight_layout()
    plt.savefig("outputs/plots/performance_by_immigrant.png")
    plt.show()


if "immigrant_status" in student_df.columns:
    plot_performance_by_immigrant(student_df)
else:
    print("Immigrant status data not available in the dataset.")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap of key variables"""

    key_vars = [
        "PV1MATH",
        "PV1READ",
        "PV1SCIE",
        "ESCS",
        "HOMEPOS",
        "HISCED",
        "DISCLIM",
        "TEACHSUP",
        "BELONG",
    ]

    available_vars = [var for var in key_vars if var in df.columns]

    corr_matrix = df[available_vars].corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )

    plt.title("Correlation Matrix of Key PISA Variables")
    plt.tight_layout()
    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.show()


plot_correlation_heatmap(student_df)


def plot_gender_gap_by_country(df, top_n=10):
    """Plot gender performance gaps across top countries"""

    country_gender_perf = (
        df.groupby(["CNT", "gender"])[["PV1MATH", "PV1READ", "PV1SCIE"]]
        .mean()
        .reset_index()
    )

    wide_format = country_gender_perf.pivot(
        index="CNT", columns="gender", values=["PV1MATH", "PV1READ", "PV1SCIE"]
    )

    gender_gaps = pd.DataFrame(
        {
            "Math Gap": wide_format[("PV1MATH", "Male")]
            - wide_format[("PV1MATH", "Female")],
            "Reading Gap": wide_format[("PV1READ", "Male")]
            - wide_format[("PV1READ", "Female")],
            "Science Gap": wide_format[("PV1SCIE", "Male")]
            - wide_format[("PV1SCIE", "Female")],
        }
    )

    reading_gap_countries = gender_gaps.sort_values("Reading Gap").index
    top_female_adv = reading_gap_countries[:top_n]
    top_male_adv = reading_gap_countries[-top_n:]
    highlighted_countries = pd.Index(top_female_adv.tolist() + top_male_adv.tolist())

    plot_data = country_gender_perf[
        country_gender_perf["CNT"].isin(highlighted_countries)
    ]

    g = sns.FacetGrid(plot_data, col="CNT", col_wrap=5, height=3, aspect=1.2)
    g.map_dataframe(sns.barplot, x="gender", y="PV1READ")
    g.set_axis_labels("Gender", "Reading Score")
    g.set_titles("{col_name}")
    g.fig.suptitle(
        "Reading Performance by Gender Across Countries with Largest Gaps",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/gender_gap_by_country.png")
    plt.show()


plot_gender_gap_by_country(student_df)


def plot_overall_gender_difference(df):
    """Plot overall gender differences across all subjects"""

    gender_perf = (
        df.groupby("gender")[["PV1MATH", "PV1READ", "PV1SCIE"]].mean().reset_index()
    )

    plot_data = pd.melt(
        gender_perf,
        id_vars=["gender"],
        value_vars=["PV1MATH", "PV1READ", "PV1SCIE"],
        var_name="Subject",
        value_name="Score",
    )

    subject_map = {"PV1MATH": "Mathematics", "PV1READ": "Reading", "PV1SCIE": "Science"}
    plot_data["Subject"] = plot_data["Subject"].map(subject_map)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        x="Subject", y="Score", hue="gender", data=plot_data, palette="Set1"
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for subject in subject_map.values():
        male_score = plot_data[
            (plot_data["Subject"] == subject) & (plot_data["gender"] == "Male")
        ]["Score"].values[0]
        female_score = plot_data[
            (plot_data["Subject"] == subject) & (plot_data["gender"] == "Female")
        ]["Score"].values[0]
        gap = male_score - female_score

        x_pos = list(subject_map.values()).index(subject)
        y_pos = (male_score + female_score) / 2

        gap_text = f"{gap:.1f}" if gap >= 0 else f"{gap:.1f}"
        plt.annotate(
            gap_text,
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos + 20),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="->", color="black" if gap >= 0 else "red"),
            fontweight="bold",
        )

    plt.title("Overall Gender Differences in Academic Performance", fontsize=14)
    plt.ylabel("Average Score", fontsize=12)
    plt.xlabel("", fontsize=12)

    plt.figtext(
        0.5,
        0.01,
        "Note: Arrows show the gender gap (Male score - Female score)",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.legend(title="Gender")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("outputs/plots/overall_gender_difference.png")
    plt.show()


plot_overall_gender_difference(student_df)


def plot_escs_math_by_immigrant(df, sample_size=10000):
    """Plot relationship between ESCS and Math by immigrant status"""

    if "immigrant_status" not in df.columns:
        print("Immigrant status not available")
        return

    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df

    g = sns.FacetGrid(sample_df, col="immigrant_status", height=4, aspect=1.2)
    g.map_dataframe(sns.regplot, x="ESCS", y="PV1MATH", scatter_kws={"alpha": 0.3})

    for i, immigrant_status in enumerate(
        sorted(df["immigrant_status"].dropna().unique())
    ):
        subset = df[df["immigrant_status"] == immigrant_status]
        corr = subset["ESCS"].corr(subset["PV1MATH"])
        g.axes[0, i].annotate(
            f"Corr: {corr:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    g.set_axis_labels("ESCS Index", "Math Score")
    g.set_titles("Immigrant Status: {col_name}")
    g.fig.suptitle(
        "Relationship Between Socioeconomic Status and Math Performance\nBy Immigrant Status",
        fontsize=16,
        y=1.05,
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/escs_math_by_immigrant.png")
    plt.show()


if "immigrant_status" in student_df.columns:
    plot_escs_math_by_immigrant(student_df)
else:
    print("Immigrant status variable not available in the dataset.")


def plot_performance_escs_climate(df, sample_size=5000):
    """Plot scatter plot with multiple encodings for performance, ESCS, and school climate"""
    if "DISCLIM" not in df.columns:
        print("School climate variable DISCLIM not found. Skipping visualization.")
        return

    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df

    sample_df["ESCS_quartile"] = pd.qcut(
        sample_df["ESCS"],
        4,
        labels=["Bottom 25%", "Lower middle", "Upper middle", "Top 25%"],
    )

    plt.figure(figsize=(12, 8))

    scatter = sns.scatterplot(
        data=sample_df,
        x="DISCLIM",
        y="PV1MATH",
        hue="ESCS_quartile",
        size="PV1READ",
        sizes=(20, 200),
        alpha=0.6,
    )

    plt.title(
        "Relationship Between School Discipline Climate, Math Performance, and Socioeconomic Status"
    )
    plt.xlabel("School Discipline Climate Index (DISCLIM)")
    plt.ylabel("Math Performance")

    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(
        handles, labels, title="ESCS Quartile", loc="upper left", bbox_to_anchor=(1, 1)
    )

    plt.annotate(
        "Point size represents reading performance",
        xy=(0.05, 0.05),
        xycoords="figure fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/performance_escs_climate.png")
    plt.show()


plot_performance_escs_climate(student_df)


get_ipython().system(
    'jupyter nbconvert --to script "Part_I_pisa_exploration.ipynb" --output-dir="outputs/scripts"'
)
get_ipython().system(
    'jupyter nbconvert --to html "Part_I_pisa_exploration.ipynb" --output-dir="outputs/html"'
)
