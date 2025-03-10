#!/usr/bin/env python
# coding: utf-8

# # PISA 2022 Data Analysis: Understanding Global Student Performance

# # Introduction
# 
# This notebook presents an in-depth analysis of the Programme for International Student Assessment (PISA) 2022 dataset, focusing on student performance across countries, socioeconomic factors, and demographic characteristics. PISA is an international assessment that evaluates education systems worldwide by testing the skills and knowledge of 15-year-old students as they approach the end of their compulsory education.
# 
# The PISA 2022 assessment covered approximately 600,000 students across participating countries, providing rich insights into educational outcomes globally. This analysis aims to uncover patterns and relationships in student performance, with particular attention to:
# 
# - Country-level differences in academic achievement
# - Gender gaps across different subject domains
# - The impact of socioeconomic status on performance
# - The interplay between school climate and academic outcomes
# 
# By examining these relationships, we can better understand the factors that contribute to effective educational systems and identify potential areas for policy intervention.

# # Data Overview
# 
# The PISA 2022 dataset contains student questionnaire data that includes:
# 
# - Performance metrics in mathematics, reading, and science
# - Student demographic information
# - Socioeconomic indicators (ESCS index)
# - School climate and contextual factors
# - Country and regional identifiers
# 
# The dataset is extensive, requiring careful preprocessing and optimization techniques to manage memory usage efficiently.

# # Research Questions
# 
# This analysis seeks to answer the following key questions:
# 
# - Which countries demonstrate the highest performance across different subject areas in PISA 2022?
# 
# - How do male and female students perform differently across the three core PISA domains?
# 
# - Which countries demonstrate the strongest link between socioeconomic status and academic performance?
# 
# - What is the relationship between a country's average socioeconomic status and its mathematics performance?
# 

# # Methodology
# 
# The analysis follows a structured approach:
# 
# - **Data Preparation:** Cleaning, optimization, and transformation of the raw PISA dataset
# - **Exploratory Data Analysis:** Investigation of distributions, relationships, and patterns
# - **In-depth Analysis:** Focused examination of key research questions
# - **Visualization:** Creation of insightful visualizations to communicate findings
# - **Interpretation:** Drawing conclusions and implications from the results
# 
# This notebook demonstrates not only the technical aspects of working with large educational datasets but also provides substantive insights into global patterns of educational achievement.

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
import os
import gc
import sys
from scipy import stats

# sav_file = "dataset/CY08MSP_STU_QQQ.SAV"
# print(f"Converting {sav_file} to CSV")
# df, meta = pyreadstat.read_sav(sav_file)
# print(df.head())
# csv_file = "dataset/CY08MSP_STU_QQQ.csv"
# df.to_csv(csv_file, index=False)
# print(f"CSV file saved as {csv_file}")
# print("Done")



# In[45]:


# csv_file = "dataset/pisa.csv"
# df.to_csv(csv_file, index=False)


# In[46]:


student_df = pd.read_csv("dataset/pisa.csv")
student_df.shape


# In[47]:


student_df.info()


# In[48]:


student_df.describe().to_csv("outputs/pisa_describe.csv")


# In[49]:


student_df.describe().info()


# In[50]:


student_df.head()


# In[51]:


student_df['ST004D01T'].value_counts()


# In[52]:


# textual_columns = student_df.select_dtypes(include=['object']).columns
# numerical_columns = student_df.select_dtypes(include=['number']).columns
# textual_columns, numerical_columns


# In[53]:


# student_df[textual_columns].head()


# In[54]:


# student_df['CNT'].value_counts()
# # convert to categorical
# student_df['CNT'] = student_df['CNT'].astype('category')


# In[55]:


# student_df['CYC'].value_counts()
# # convert to categorical
# # student_df['CNT'] = student_df['CNT'].astype('category')


# In[56]:


# student_df['STRATUM'].value_counts()


# In[57]:


def memory_usage(pandas_obj):
    """Calculate memory usage of a pandas object in MB"""
    if isinstance(pandas_obj, pd.DataFrame):
        usage_bytes = pandas_obj.memory_usage(deep=True).sum()
    else:  # Series
        usage_bytes = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_bytes / (1024 * 1024)
    return usage_mb


# In[58]:


def optimize_floats(df):
    """Optimize float dtypes by downcasting to float32 where possible"""
    float_cols = df.select_dtypes(include=['float64']).columns
    
    for col in float_cols:
        # Check if column can be represented as float32 without losing precision
        # For PISA data, most measurements don't need float64 precision
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


# In[59]:


def optimize_ints(df):
    """Optimize integer dtypes by downcasting to smallest possible integer type"""
    int_cols = df.select_dtypes(include=['int64']).columns
    
    for col in int_cols:
        # For each column, downcast to the smallest possible integer type
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df


# In[60]:


def optimize_categorical(df, categorical_threshold=0.5, excluded_cols=None):
    """Convert columns with low cardinality to categorical type"""
    if excluded_cols is None:
        excluded_cols = []
    
    # Identify columns that are good candidates for categorical conversion
    # These are columns where # unique values / # rows < threshold
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if col not in excluded_cols:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < categorical_threshold:
                df[col] = df[col].astype('category')
    
    # Also look for integer columns that should be categorical
    # (like country codes, gender, etc.)
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        if col not in excluded_cols:
            num_unique_values = len(df[col].unique())
            if num_unique_values < 50:  # If fewer than 50 unique values, likely categorical
                df[col] = df[col].astype('category')
    
    return df


# In[61]:


def optimize_known_pisa_columns(df):
    """Apply specific optimizations for known PISA data columns"""
    # Columns that we know contain only one value (like CYC)
    single_value_cols = ['CYC'] 
    for col in single_value_cols:
        if col in df.columns:
            # Converting to category is most efficient for columns with a single value
            df[col] = df[col].astype('category')
    
    # Country codes, language codes, school IDs should be categorical
    categorical_cols = [
        'CNT', 'CNTRYID', 'SUBNATIO', 'LANGTEST_QQQ', 'LANGTEST_COG', 
        'LANGTEST_PAQ', 'ISCEDP', 'COBN_S', 'COBN_M', 'COBN_F', 'LANGN', 
        'REGION', 'OECD'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # PVs (plausible values) are float with limited precision needed
    pv_cols = [col for col in df.columns if col.startswith('PV')]
    for col in pv_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


# In[62]:


# def read_pisa_in_chunks(filepath, chunksize=100000, optimize=True, output_file=None):
#     """
#     Read and process PISA data in chunks to reduce memory usage
    
#     Parameters:
#     filepath (str): Path to the PISA CSV file
#     chunksize (int): Number of rows to read at once
#     optimize (bool): Whether to apply memory optimizations
#     output_file (str): Path to save optimized CSV (if None, doesn't save)
    
#     Returns:
#     pd.DataFrame: The optimized dataframe (if output_file is None), otherwise None
#     """
#     # Get the total number of rows to track progress
#     total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract 1 for header
    
#     # If we're saving to a file, process chunk by chunk without keeping in memory
#     if output_file:
#         print(f"Processing {filepath} in chunks of {chunksize} rows")
#         print(f"Total rows to process: {total_rows}")
        
#         # Process the first chunk to get column dtypes for future chunks
#         first_chunk = pd.read_csv(filepath, nrows=chunksize)
        
#         if optimize:
#             print("Optimizing first chunk to determine dtypes...")
#             first_chunk = optimize_floats(first_chunk)
#             first_chunk = optimize_ints(first_chunk)
#             first_chunk = optimize_categorical(first_chunk)
#             first_chunk = optimize_known_pisa_columns(first_chunk)
        
#         # Get optimized dtypes
#         optimized_dtypes = first_chunk.dtypes
        
#         # Write the first chunk to file with header
#         first_chunk.to_csv(output_file, mode='w', index=False)
        
#         # Process the rest of the file in chunks
#         rows_processed = len(first_chunk)
        
#         # Free memory
#         del first_chunk
#         gc.collect()
        
#         for chunk in pd.read_csv(filepath, chunksize=chunksize, skiprows=range(1, rows_processed+1)):
#             # Apply dtype conversions based on optimized first chunk
#             for col in chunk.columns:
#                 if col in optimized_dtypes:
#                     chunk[col] = chunk[col].astype(optimized_dtypes[col])
            
#             # Append to the output file
#             chunk.to_csv(output_file, mode='a', header=False, index=False)
            
#             # Update progress
#             rows_processed += len(chunk)
#             progress = (rows_processed / total_rows) * 100
#             print(f"Processed {rows_processed:,}/{total_rows:,} rows ({progress:.1f}%)")
            
#             # Free memory
#             del chunk
#             gc.collect()
        
#         print(f"Optimized data saved to {output_file}")
#         return None
    
#     # If we're not saving to a file, read the entire dataset and return it
#     else:
#         print(f"Reading entire dataset into memory from {filepath}")
#         df = pd.read_csv(filepath)
        
#         original_memory = memory_usage(df)
        
#         if optimize:
#             print("Applying optimizations...")
#             df = optimize_floats(df)
#             df = optimize_ints(df)
#             df = optimize_categorical(df)
#             df = optimize_known_pisa_columns(df)
            
#             optimized_memory = memory_usage(df)
            
#             print(f"Optimized memory usage: {optimized_memory:.2f} MB")
#             print(f"Memory usage reduced by: {original_memory - optimized_memory:.2f} MB ({((original_memory - optimized_memory) / original_memory) * 100:.1f}%)")
        
#         return df


# In[63]:


student_df = optimize_floats(student_df)
student_df = optimize_ints(student_df)
student_df = optimize_categorical(student_df)
student_df = optimize_known_pisa_columns(student_df)


student_df.info()


# In[64]:


# input_file = "dataset/pisa.csv"
# output_file = "dataset/pisa_optimized.csv"

# # Process in chunks and save (good for large files)
# # read_pisa_in_chunks(input_file, chunksize=100000, optimize=True, output_file=output_file)

# # Alternatively, to read into memory (for smaller files or if you need the dataframe):
# student_df_optimized = read_pisa_in_chunks(
#     input_file, optimize=True, output_file=output_file
# )

# # If you want to analyze memory usage by column type after optimization:
# if os.path.exists(output_file):
#     print("\nAnalyzing column types in optimized file...")
#     df_sample = pd.read_csv(output_file, nrows=1000)  # Just read a sample for analysis

#     # Group by dtype and count columns
#     dtype_counts = df_sample.dtypes.value_counts()
#     print("Column counts by data type:")
#     print(dtype_counts)

#     # Memory usage by dtype
#     usage_by_dtype = {}
#     for dtype_name in dtype_counts.index:
#         cols = df_sample.select_dtypes(include=[dtype_name]).columns
#         usage = memory_usage(df_sample[cols])
#         usage_by_dtype[dtype_name] = usage

#     print("\nMemory usage by data type (MB) for sample:")
#     for dtype_name, usage in usage_by_dtype.items():
#         print(f"{dtype_name}: {usage:.2f} MB")


# In[65]:


os.makedirs('outputs/cleaning', exist_ok=True)

def clean_pisa_data(df, save_path=None, inplace=True):
    """
    Comprehensive data cleaning for PISA dataset that modifies the original dataset
    
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
    
    # Step 1: Check for duplicate student IDs
    print("\nStep 1: Checking for duplicate student IDs...")
    if 'CNTSTUID' in df.columns:
        duplicate_ids = df['CNTSTUID'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"Found {duplicate_ids} duplicate student IDs. Removing duplicates.")
            df.drop_duplicates(subset='CNTSTUID', keep='first', inplace=True)
        else:
            print("No duplicate student IDs found.")
    else:
        print("Warning: CNTSTUID column not found. Skipping duplicate check.")
    
    # Step 2: Handle missing values in key variables
    print("\nStep 2: Handling missing values in key variables...")
    
    # List key variables for different analyses
    key_demographics = ['AGE', 'GRADE', 'ST004D01T', 'IMMIG']
    key_performance = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    key_ses = ['ESCS', 'HOMEPOS', 'HISCED']
    key_school = ['SCHSUST', 'DISCLIM', 'TEACHSUP', 'BELONG']
    
    all_key_vars = key_demographics + key_performance + key_ses + key_school
    existing_key_vars = [col for col in all_key_vars if col in df.columns]
    
    # Check missing values in key variables
    missing_counts = df[existing_key_vars].isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Create DataFrame for missing values report
    missing_report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    }).sort_values('Missing Percentage', ascending=False)
    
    print("Missing values in key variables:")
    print(missing_report[missing_report['Missing Count'] > 0])
    
    # Save the missing values report
    missing_report.to_csv('outputs/cleaning/missing_values_report.csv')
    
    # Create visualization of missing values
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[existing_key_vars].isnull(), cbar=False, yticklabels=False,
                cmap='viridis')
    plt.title('Missing Values in Key Variables')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('outputs/cleaning/missing_values_heatmap.png')
    plt.close()
    
    # Step 3: Handle specific cases of missingness - with more aggressive handling
    print("\nStep 3: Handling specific missing value patterns...")
    
    # For performance variables (PVs), we cannot impute - students must have scores
    perf_vars = [col for col in key_performance if col in df.columns]
    missing_perf = df[perf_vars].isnull().any(axis=1)
    if missing_perf.sum() > 0:
        print(f"Removing {missing_perf.sum()} students with missing performance scores.")
        df.drop(df[missing_perf].index, inplace=True)
        
    # Remove rows with excessive missing values (more than 50% of key variables)
    key_vars_present = df[existing_key_vars].count(axis=1)
    min_vars_required = len(existing_key_vars) * 0.5
    excessive_missing = key_vars_present < min_vars_required
    if excessive_missing.sum() > 0:
        print(f"Removing {excessive_missing.sum()} students with more than 50% of key variables missing.")
        df.drop(df[excessive_missing].index, inplace=True)
    
    # For categorical demographic variables, replace with most frequent value
    for col in key_demographics:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_count = df[col].isnull().sum()
            
            # For categorical variables with few unique values
            if df[col].dtype == 'category' or df[col].nunique() < 10:
                most_frequent = df[col].mode()[0]
                df[col].fillna(most_frequent, inplace=True)
                print(f"Filled {missing_count} missing values in {col} with most frequent value: {most_frequent}")
            else:
                # For continuous variables, impute with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {missing_count} missing values in {col} with median: {median_val}")
    
    # For SES variables, use more sophisticated imputation
    if 'ESCS' in df.columns and df['ESCS'].isnull().sum() > 0:
        ses_vars = [col for col in key_ses if col in df.columns and col != 'ESCS']
        missing_count = df['ESCS'].isnull().sum()
        
        if len(ses_vars) > 0:
            # Try KNN imputation if we have related variables
            try:
                from sklearn.impute import KNNImputer
                
                # Prepare data for imputation
                impute_cols = ses_vars + ['ESCS']
                impute_df = df[impute_cols].copy()
                
                # Create KNN imputer
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(impute_df)
                
                # Update only missing values
                df.loc[df['ESCS'].isnull(), 'ESCS'] = imputed_values[df['ESCS'].isnull(), -1]
                print(f"Imputed {missing_count} missing ESCS values using KNN imputation.")
            except:
                # Fall back to simple median imputation if KNN fails
                median_escs = df['ESCS'].median()
                df['ESCS'].fillna(median_escs, inplace=True)
                print(f"Imputed {missing_count} missing ESCS values with median: {median_escs}")
        else:
            # If no related variables, use median imputation
            median_escs = df['ESCS'].median()
            df['ESCS'].fillna(median_escs, inplace=True)
            print(f"Imputed {missing_count} missing ESCS values with median: {median_escs}")
    
    # For school variables, use median imputation
    for col in key_school:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_count = df[col].isnull().sum()
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled {missing_count} missing values in {col} with median.")
    
    # Step 4: Detect and handle outliers more aggressively
    print("\nStep 4: Detecting and handling outliers in key continuous variables...")
    
    continuous_vars = []
    for col in existing_key_vars:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            if df[col].nunique() > 10:  # Simple heuristic for continuous variables
                continuous_vars.append(col)
    
    # Report and handle outliers more aggressively (values > 2.5 standard deviations from mean)
    outlier_report = {}
    for col in continuous_vars:
        # Skip performance variables (PVs) as their scale is standardized
        if col in perf_vars:
            continue
        
        # Use robust statistics (median and IQR) to detect outliers
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        
        # Define bounds using both methods and use the more conservative one
        # Method 1: IQR method
        iqr_lower_bound = Q1 - 1.5 * IQR
        iqr_upper_bound = Q3 + 1.5 * IQR
        
        # Method 2: Z-score method with tighter threshold (2.5 instead of 3)
        mean_val = df[col].mean()
        std_val = df[col].std()
        z_lower_bound = mean_val - 2.5 * std_val
        z_upper_bound = mean_val + 2.5 * std_val
        
        # Use the more conservative bound (higher lower bound, lower upper bound)
        lower_bound = max(iqr_lower_bound, z_lower_bound)
        upper_bound = min(iqr_upper_bound, z_upper_bound)
        
        # Identify outliers
        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        all_outliers = lower_outliers | upper_outliers
        outlier_count = all_outliers.sum()
        
        if outlier_count > 0:
            outlier_report[col] = {
                'Count': outlier_count,
                'Percentage': (outlier_count / len(df)) * 100,
                'Min Outlier': df.loc[all_outliers, col].min(),
                'Max Outlier': df.loc[all_outliers, col].max(),
                'IQR Range': f"[{iqr_lower_bound:.2f}, {iqr_upper_bound:.2f}]",
                'Z-score Range': f"[{z_lower_bound:.2f}, {z_upper_bound:.2f}]",
                'Final Range': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            }
            
            # Cap upper outliers
            if upper_outliers.sum() > 0:
                print(f"Capping {upper_outliers.sum()} upper outliers in {col} at {upper_bound:.2f}")
                df.loc[upper_outliers, col] = upper_bound
            
            # Cap lower outliers
            if lower_outliers.sum() > 0:
                print(f"Capping {lower_outliers.sum()} lower outliers in {col} at {lower_bound:.2f}")
                df.loc[lower_outliers, col] = lower_bound
    
    outlier_df = pd.DataFrame(outlier_report).T
    
    if not outlier_df.empty:
        print("\nOutliers detected and handled:")
        print(outlier_df)
        outlier_df.to_csv('outputs/cleaning/outlier_report.csv')
    else:
        print("No outliers detected in continuous variables.")
    
    # Handle special case for age - remove implausible values and tighten range for PISA
    if 'AGE' in df.columns:
        # PISA targets 15-year-olds, so restrict to a narrower range for main analysis
        # For primary analysis, keep only students within 14-16 age range
        age_outliers = (df['AGE'] < 14) | (df['AGE'] > 16.5)
        if age_outliers.sum() > 0:
            print(f"Removing {age_outliers.sum()} students outside the target age range (14-16.5).")
            df.drop(df[age_outliers].index, inplace=True)
    
    # Step 5: Standardize categorical variables 
    print("\nStep 5: Standardizing categorical variables...")
    
    cat_vars = [col for col in existing_key_vars 
                if col not in continuous_vars and df[col].dtype != 'category']
    
    for col in cat_vars:
        if df[col].nunique() < 50:
            # Get value counts to identify potential issues
            val_counts = df[col].value_counts()
            
            # Check for rare categories (less than 0.1% of data)
            rare_cats = val_counts[val_counts / len(df) < 0.001].index.tolist()
            
            if rare_cats:
                print(f"Column {col}: Consolidating {len(rare_cats)} rare categories into 'Other'")
                
                # If not already categorical, convert
                if df[col].dtype != 'category':
                    df[col] = df[col].astype('category')
                
                # Add 'Other' category
                if 'Other' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['Other'])
                
                # Replace rare categories with 'Other'
                df.loc[df[col].isin(rare_cats), col] = 'Other'
    
    # Step 6: Create derived variables that might be useful
    print("\nStep 6: Creating derived variables...")
    
    # Create age groups
    if 'AGE' in df.columns:
        df['AGE_GROUP'] = pd.cut(
            df['AGE'], 
            bins=[0, 12, 14, 16, 18, 100],
            labels=['Under 12', '12-14', '14-16', '16-18', 'Over 18']
        )
        print("Created AGE_GROUP variable.")
    
    # Create ESCS quintiles
    if 'ESCS' in df.columns:
        df['ESCS_QUINTILE'] = pd.qcut(
            df['ESCS'],
            q=5,
            labels=['Lowest', 'Low', 'Middle', 'High', 'Highest']
        )
        print("Created ESCS_QUINTILE variable.")
    
    # Create performance level categories based on PISA benchmarks
    for pv in ['PV1MATH', 'PV1READ', 'PV1SCIE']:
        if pv in df.columns:
            # PISA proficiency levels (approximate thresholds)
            var_name = f"{pv}_LEVEL"
            
            # Different thresholds for different domains
            if pv == 'PV1MATH':
                bins = [0, 358, 420, 482, 545, 607, 669, 1000]
            elif pv == 'PV1READ':
                bins = [0, 335, 407, 480, 553, 626, 698, 1000]
            else:  # Science
                bins = [0, 335, 410, 484, 559, 633, 708, 1000]
            
            # Create the levels variable
            df[var_name] = pd.cut(
                df[pv],
                bins=bins,
                labels=['Below 1', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6']
            )
            print(f"Created {var_name} variable.")
            
            # Create binary high/low performance indicators (level 4+ vs below)
            high_cutoff = bins[4]  # Level 4 cutoff
            df[f"{pv}_HIGH"] = (df[pv] >= high_cutoff).astype(int)
            print(f"Created {pv}_HIGH indicator for scores ≥ {high_cutoff} (Level 4+).")
    
    # Step 7: Final validation and additional checks
    print("\nStep 7: Final data validation...")
    
    # Check for any remaining extreme values in performance variables
    for pv in perf_vars:
        # Performance variables in PISA are standardized with mean ~500 and SD ~100
        # Extreme values outside 100-900 range are suspicious
        extreme_scores = (df[pv] < 100) | (df[pv] > 900)
        if extreme_scores.sum() > 0:
            print(f"WARNING: Found {extreme_scores.sum()} extreme values in {pv}.")
            # Remove these extreme values
            print(f"Removing {extreme_scores.sum()} rows with extreme {pv} values.")
            df.drop(df[extreme_scores].index, inplace=True)
    
    # Check for any remaining implausible relationships
    # For example, very high performance with very low ESCS or vice versa
    if 'ESCS' in df.columns and 'PV1MATH' in df.columns:
        # Calculate residuals from linear regression
        from sklearn.linear_model import LinearRegression
        X = df[['ESCS']]
        y = df['PV1MATH']
        reg = LinearRegression().fit(X, y)
        df['MATH_RESIDUAL'] = y - reg.predict(X)
        
        # Flag extreme residuals (± 3 standard deviations)
        residual_std = df['MATH_RESIDUAL'].std()
        extreme_residuals = np.abs(df['MATH_RESIDUAL']) > 3 * residual_std
        if extreme_residuals.sum() > 0:
            print(f"Flagged {extreme_residuals.sum()} rows with extreme ESCS-Math relationships.")
            # Add flag variable but don't remove
            df['EXTREME_RESIDUAL'] = extreme_residuals.astype(int)
            print("Added 'EXTREME_RESIDUAL' flag for these observations.")
    
    # Step 8: Create final report
    print("\nStep 8: Creating final dataset report...")
    
    rows_removed = original_size - len(df)
    cleaning_report = {
        'Original rows': original_size,
        'Rows after cleaning': len(df),
        'Rows removed': rows_removed,
        'Percentage removed': (rows_removed / original_size) * 100,
        'Missing values before': missing_counts.sum(),
        'Missing values after': df[existing_key_vars].isnull().sum().sum(),
        'Variables modified': len(existing_key_vars),
        'New variables created': sum(['AGE_GROUP' in df.columns, 
                                      'ESCS_QUINTILE' in df.columns,
                                      'PV1MATH_LEVEL' in df.columns,
                                      'PV1READ_LEVEL' in df.columns,
                                      'PV1SCIE_LEVEL' in df.columns,
                                      'PV1MATH_HIGH' in df.columns,
                                      'PV1READ_HIGH' in df.columns, 
                                      'PV1SCIE_HIGH' in df.columns])
    }
    
    print("\nCleaning summary:")
    for key, value in cleaning_report.items():
        print(f"{key}: {value}")
    
    # Save the cleaning report
    pd.DataFrame([cleaning_report]).to_csv('outputs/cleaning/cleaning_summary.csv', index=False)
    
    # Create a data quality report
    # Sample head of the dataset
    df.head().to_csv('outputs/cleaning/sample_head.csv')
    
    # Summary statistics
    df[continuous_vars].describe().to_csv('outputs/cleaning/continuous_vars_summary.csv')
    
    # Value counts for categorical variables
    cat_vars_extended = cat_vars + ['AGE_GROUP', 'ESCS_QUINTILE', 'PV1MATH_LEVEL', 'PV1READ_LEVEL', 'PV1SCIE_LEVEL']
    cat_vars_extended = [v for v in cat_vars_extended if v in df.columns]
    
    for var in cat_vars_extended:
        if var in df.columns:
            df[var].value_counts().to_csv(f'outputs/cleaning/{var}_distribution.csv')
    
    # Save the cleaned dataset
    if save_path:
        print(f"\nSaving cleaned dataset to {save_path}")
        df.to_csv(save_path, index=False)
    
    print("\nData cleaning completed successfully.")
    return df


# In[66]:


student_df = clean_pisa_data(student_df, save_path='outputs/pisa_cleaned.csv')


# In[67]:


student_df.info()


# In[68]:


student_df['ST004D01T'].value_counts()


# ## 4. Exploratory Data Analysis
# 

# I'll focus on the following metrics and variables:
# 
# - Academic performance metrics (Math, Reading, Science scores)
# - Socioeconomic factors (ESCS index)
# - Demographic variables (gender, immigrant status)
# - School factors (climate, resources)

# For the explanatory portion, I'll select 3-5 key insights from the exploration and create polished visualizations that effectively communicate these findings.

# In[69]:


key_vars = {
    'Performance': ['PV1MATH', 'PV1READ', 'PV1SCIE'],
    'Demographics': ['AGE', 'GRADE', 'ST004D01T', 'IMMIG'],
    'Socioeconomic': ['ESCS', 'HOMEPOS', 'HISCED'],
    'School Climate': ['SCHSUST', 'DISCLIM', 'TEACHSUP', 'BELONG']
}


# In[70]:


for category, variables in key_vars.items():
    print(f"\n{category} Variables:")
    for var in variables:
        if var in student_df.columns:
            print(f"- {var} available")
        else:
            print(f"- {var} NOT found in dataset")


# ### 4.1 Performance Distributions

# In[71]:


def plot_performance_distributions(df):
    """Plot histograms of PISA performance scores"""
    performance_vars = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    titles = ['Mathematics', 'Reading', 'Science']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (var, title) in enumerate(zip(performance_vars, titles)):
        sns.histplot(df[var], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {title} Scores')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequency')
        
        # Add mean line
        mean_val = df[var].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--')
        axes[i].text(mean_val + 10, 0.8 * axes[i].get_ylim()[1], 
                    f'Mean: {mean_val:.1f}', color='red')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/performance_distributions.png')
    plt.show()

plot_performance_distributions(student_df)


# ### Observations: 
# - Comment on shape of distributions
# - Note any patterns or unusual aspects

# ### 4.2 Socioeconomic Status Analysis
# 

# In[72]:


def plot_escs_distribution(df):
    """Plot histogram of ESCS index"""
    plt.figure(figsize=(12, 6))
    
    sns.histplot(df['ESCS'], kde=True)
    plt.title('Distribution of Economic, Social and Cultural Status (ESCS) Index')
    plt.xlabel('ESCS Index')
    plt.ylabel('Frequency')
    
    # Add mean and median lines
    mean_val = df['ESCS'].mean()
    median_val = df['ESCS'].median()
    
    plt.axvline(mean_val, color='red', linestyle='--')
    plt.axvline(median_val, color='green', linestyle='-.')
    
    plt.text(mean_val + 0.1, 0.8 * plt.ylim()[1], f'Mean: {mean_val:.2f}', color='red')
    plt.text(median_val + 0.1, 0.7 * plt.ylim()[1], f'Median: {median_val:.2f}', color='green')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_distribution.png')
    plt.show()

plot_escs_distribution(student_df)
# Observations: 
# - Comment on shape of distribution
# - Note any skewness or unusual patterns

# COUNTPLOT: Student distribution by gender


# ### 4.3 Gender Analysis

# In[73]:


def plot_gender_distribution(df):
    """Plot count of students by gender"""
    plt.figure(figsize=(8, 6))
    
    # Map gender codes to labels
    gender_map = {1: 'Female', 2: 'Male'}
    df['gender'] = df['ST004D01T'].map(gender_map)
    
    sns.countplot(x='gender', data=df)
    plt.title('Distribution of Students by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():,}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'center', 
                          xytext = (0, 10), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/gender_distribution.png')
    plt.show()

# Execute the function
plot_gender_distribution(student_df)
# Observations:
# - Comment on gender balance in the dataset

# BAR CHART: Top-performing countries in mathematics


# ### 4.4 Country-Level Analysis

# In[74]:


def plot_top_countries_math(df, top_n=15):
    """Plot bar chart of top countries by math performance"""
    # Calculate mean math score by country
    country_math = df.groupby('CNT')['PV1MATH'].mean().sort_values(ascending=False)
    top_countries = country_math.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title(f'Top {top_n} Countries by Mathematics Performance')
    plt.xlabel('Average Math Score')
    plt.ylabel('Country')
    
    # Add global average line
    global_avg = df['PV1MATH'].mean()
    plt.axvline(global_avg, color='red', linestyle='--')
    plt.text(global_avg + 5, 1, f'Global Average: {global_avg:.1f}', color='red')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/top_countries_math.png')
    plt.show()

# Execute the function
plot_top_countries_math(student_df)
# Observations:
# - Comment on which countries perform best
# - Note any patterns or regional trends


# In[75]:


# =========================================================
# 3. BIVARIATE EXPLORATION
# =========================================================

# SCATTER PLOT: Relationship between ESCS and Math Performance

def plot_escs_math_relationship(df, sample_size=5000):
    """Plot scatter plot of ESCS vs Math performance"""
    # Take a random sample to make plotting faster if dataset is large
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    plt.figure(figsize=(12, 8))
    
    sns.regplot(x='ESCS', y='PV1MATH', data=sample_df, 
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    
    plt.title('Relationship between Socioeconomic Status and Math Performance')
    plt.xlabel('ESCS (Economic, Social and Cultural Status Index)')
    plt.ylabel('Math Score')
    
    # Add correlation coefficient
    corr = df['ESCS'].corr(df['PV1MATH'])
    plt.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_math_relationship.png')
    plt.show()

# Execute the function
plot_escs_math_relationship(student_df)
# Observations:
# - Comment on the strength and direction of the relationship
# - Note any patterns or unusual observations



# In[76]:


# BOX PLOT: Performance by Gender
def plot_performance_by_gender(df):
    """Plot box plots of performance by gender"""
    # Map gender codes to labels
    gender_map = {1: 'Female', 2: 'Male'}
    df['gender'] = df['ST004D01T'].map(gender_map)
    
    # Reshape data for plotting
    perf_vars = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    perf_titles = ['Mathematics', 'Reading', 'Science']
    
    plt.figure(figsize=(15, 6))
    
    for i, (var, title) in enumerate(zip(perf_vars, perf_titles)):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x='gender', y=var, data=df)
        plt.title(f'{title} Performance by Gender')
        plt.xlabel('Gender')
        plt.ylabel(f'{title} Score')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/performance_by_gender.png')
    plt.show()

# Execute the function
plot_performance_by_gender(student_df)
# Observations:
# - Comment on gender differences across subjects
# - Note which differences appear most pronounced


# In[77]:


# BOX PLOT: Performance by Immigrant Status

def plot_performance_by_immigrant(df):
    """Plot box plots of performance by immigrant status"""
    # Map immigrant status codes to labels
    immig_map = {1: 'Native', 2: 'Second-Gen', 3: 'First-Gen'}
    df['immigrant_status'] = df['IMMIG'].map(immig_map)
    
    # Reshape data for plotting
    plt.figure(figsize=(15, 6))
    
    for i, (var, title) in enumerate(zip(['PV1MATH', 'PV1READ', 'PV1SCIE'], 
                                        ['Mathematics', 'Reading', 'Science'])):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x='immigrant_status', y=var, data=df)
        plt.title(f'{title} Performance by Immigrant Status')
        plt.xlabel('Immigrant Status')
        plt.ylabel(f'{title} Score')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/performance_by_immigrant.png')
    plt.show()

# Execute the function if immigrant status data is available
if 'IMMIG' in student_df.columns:
    plot_performance_by_immigrant(student_df)
# Observations:
# - Comment on performance differences by immigrant status
# - Note which subjects show largest gaps


# ### 4.5 Advanced Analysis: Multiple Factors

# In[78]:


# HEATMAP: Correlation between key variables

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of key variables"""
    # Select relevant variables
    key_vars = [
        'PV1MATH', 'PV1READ', 'PV1SCIE',  # Performance
        'ESCS', 'HOMEPOS', 'HISCED',      # Socioeconomic
        'DISCLIM', 'TEACHSUP', 'BELONG'   # School climate
    ]
    
    # Filter to variables that exist in the dataframe
    available_vars = [var for var in key_vars if var in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix of Key PISA Variables')
    plt.tight_layout()
    plt.savefig('outputs/plots/correlation_heatmap.png')
    plt.show()

# Execute the function
plot_correlation_heatmap(student_df)
# Observations:
# - Comment on strongest correlations
# - Note any surprising patterns


# In[79]:


# =========================================================
# 4. MULTIVARIATE EXPLORATION
# =========================================================

# FACET PLOT: Performance by gender across countries

def plot_gender_gap_by_country(df, top_n=10):
    """Plot gender performance gaps across top countries"""
    # Calculate mean performance by country and gender
    gender_map = {1: 'Female', 2: 'Male'}
    df['gender'] = df['ST004D01T'].map(gender_map)
    
    country_gender_perf = df.groupby(['CNT', 'gender'])[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean().reset_index()
    
    # Calculate gender gap (male - female)
    wide_format = country_gender_perf.pivot(index='CNT', columns='gender', 
                                           values=['PV1MATH', 'PV1READ', 'PV1SCIE'])
    
    # Calculate absolute gender gaps
    gender_gaps = pd.DataFrame({
        'Math Gap': wide_format[('PV1MATH', 'Male')] - wide_format[('PV1MATH', 'Female')],
        'Reading Gap': wide_format[('PV1READ', 'Male')] - wide_format[('PV1READ', 'Female')],
        'Science Gap': wide_format[('PV1SCIE', 'Male')] - wide_format[('PV1SCIE', 'Female')]
    })
    
    # Select countries with largest reading gaps (both directions)
    reading_gap_countries = gender_gaps.sort_values('Reading Gap').index
    top_female_adv = reading_gap_countries[:top_n]  # Female advantage
    top_male_adv = reading_gap_countries[-top_n:]   # Male advantage
    highlighted_countries = pd.Index(top_female_adv.tolist() + top_male_adv.tolist())
    
    # Filter data for selected countries
    plot_data = country_gender_perf[country_gender_perf['CNT'].isin(highlighted_countries)]
    
    # Create facet plot
    g = sns.FacetGrid(plot_data, col='CNT', col_wrap=5, height=3, aspect=1.2)
    g.map_dataframe(sns.barplot, x='gender', y='PV1READ')
    g.set_axis_labels('Gender', 'Reading Score')
    g.set_titles('{col_name}')
    g.fig.suptitle('Reading Performance by Gender Across Countries with Largest Gaps', 
                  fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/gender_gap_by_country.png')
    plt.show()

# Execute the function
plot_gender_gap_by_country(student_df)
# Observations:
# - Comment on countries with largest gender gaps
# - Note directions of the gaps (which gender has advantage in which countries)


# In[80]:


# FACET PLOT: Relationship between ESCS and Math performance by immigrant status

def plot_escs_math_by_immigrant(df, sample_size=10000):
    """Plot relationship between ESCS and Math by immigrant status"""
    # Ensure 'IMMIG' column contains only numeric values
    df['IMMIG'] = pd.to_numeric(df['IMMIG'], errors='coerce')
    
    # Map immigrant status codes to labels
    immig_map = {1: 'Native', 2: 'Second-Gen', 3: 'First-Gen'}
    df['immigrant_status'] = df['IMMIG'].map(immig_map)
    
    # Sample data if needed
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    # Create facet plot
    g = sns.FacetGrid(sample_df, col='immigrant_status', height=4, aspect=1.2)
    g.map_dataframe(sns.regplot, x='ESCS', y='PV1MATH', scatter_kws={'alpha': 0.3})
    
    # Add correlation coefficients
    for i, immigrant_status in enumerate(sorted(df['immigrant_status'].dropna().unique())):
        subset = df[df['immigrant_status'] == immigrant_status]
        corr = subset['ESCS'].corr(subset['PV1MATH'])
        g.axes[0, i].annotate(f'Corr: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    g.set_axis_labels('ESCS Index', 'Math Score')
    g.set_titles('Immigrant Status: {col_name}')
    g.fig.suptitle('Relationship Between Socioeconomic Status and Math Performance\nBy Immigrant Status', 
                   fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_math_by_immigrant.png')
    plt.show()

# Execute the function if immigrant status data is available
if 'IMMIG' in student_df.columns:
    plot_escs_math_by_immigrant(student_df)
# Observations:
# - Comment on differences in ESCS-performance relationship across immigrant groups
# - Note any variations in strength of relationship
# Observations:
# - Comment on differences in ESCS-performance relationship across immigrant groups
# - Note any variations in strength of relationship FACET PLOT: Relationship between ESCS and Math performance by immigrant status

def plot_escs_math_by_immigrant(df, sample_size=10000):
    """Plot relationship between ESCS and Math by immigrant status"""
    # Ensure 'IMMIG' column contains only numeric values
    df['IMMIG'] = pd.to_numeric(df['IMMIG'], errors='coerce')
    
    # Map immigrant status codes to labels
    immig_map = {1: 'Native', 2: 'Second-Gen', 3: 'First-Gen'}
    df['immigrant_status'] = df['IMMIG'].map(immig_map)
    
    # Sample data if needed
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    # Create facet plot
    g = sns.FacetGrid(sample_df, col='immigrant_status', height=4, aspect=1.2)
    g.map_dataframe(sns.regplot, x='ESCS', y='PV1MATH', scatter_kws={'alpha': 0.3})
    
    # Add correlation coefficients
    for i, immigrant_status in enumerate(sorted(df['immigrant_status'].dropna().unique())):
        subset = df[df['immigrant_status'] == immigrant_status]
        corr = subset['ESCS'].corr(subset['PV1MATH'])
        g.axes[0, i].annotate(f'Corr: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    g.set_axis_labels('ESCS Index', 'Math Score')
    g.set_titles('Immigrant Status: {col_name}')
    g.fig.suptitle('Relationship Between Socioeconomic Status and Math Performance\nBy Immigrant Status', 
                   fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_math_by_immigrant.png')
    plt.show()

# Execute the function if immigrant status data is available
if 'IMMIG' in student_df.columns:
    plot_escs_math_by_immigrant(student_df)
# Observations:
# - Comment on differences in ESCS-performance relationship across immigrant groups
# - Note any variations in strength of relationship FACET PLOT: Relationship between ESCS and Math performance by immigrant status

def plot_escs_math_by_immigrant(df, sample_size=10000):
    """Plot relationship between ESCS and Math by immigrant status"""
    # Ensure 'IMMIG' column contains only numeric values
    df['IMMIG'] = pd.to_numeric(df['IMMIG'], errors='coerce')
    
    # Map immigrant status codes to labels
    immig_map = {1: 'Native', 2: 'Second-Gen', 3: 'First-Gen'}
    df['immigrant_status'] = df['IMMIG'].map(immig_map)
    
    # Sample data if needed
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    # Create facet plot
    g = sns.FacetGrid(sample_df, col='immigrant_status', height=4, aspect=1.2)
    g.map_dataframe(sns.regplot, x='ESCS', y='PV1MATH', scatter_kws={'alpha': 0.3})
    
    # Add correlation coefficients
    for i, immigrant_status in enumerate(sorted(df['immigrant_status'].dropna().unique())):
        subset = df[df['immigrant_status'] == immigrant_status]
        corr = subset['ESCS'].corr(subset['PV1MATH'])
        g.axes[0, i].annotate(f'Corr: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    g.set_axis_labels('ESCS Index', 'Math Score')
    g.set_titles('Immigrant Status: {col_name}')
    g.fig.suptitle('Relationship Between Socioeconomic Status and Math Performance\nBy Immigrant Status', 
                   fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_math_by_immigrant.png')
    plt.show()

# Execute the function if immigrant status data is available
if 'IMMIG' in student_df.columns:
    plot_escs_math_by_immigrant(student_df)
# Observations:
# - Comment on differences in ESCS-performance relationship across immigrant groups
# - Note any variations in strength of relationship


# In[81]:


# SCATTER PLOT with multiple encodings: Performance, ESCS, and School Climate

def plot_performance_escs_climate(df, sample_size=5000):
    """Plot scatter plot with multiple encodings for performance, ESCS, and school climate"""
    if 'DISCLIM' not in df.columns:
        print("School climate variable DISCLIM not found. Skipping visualization.")
        return
    
    # Sample data if needed
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    # Create ESCS quartiles for color encoding
    sample_df['ESCS_quartile'] = pd.qcut(sample_df['ESCS'], 4, 
                                        labels=['Bottom 25%', 'Lower middle', 
                                                'Upper middle', 'Top 25%'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(data=sample_df, x='DISCLIM', y='PV1MATH', 
                             hue='ESCS_quartile', size='PV1READ',
                             sizes=(20, 200), alpha=0.6)
    
    plt.title('Relationship Between School Discipline Climate, Math Performance, and Socioeconomic Status')
    plt.xlabel('School Discipline Climate Index (DISCLIM)')
    plt.ylabel('Math Performance')
    
    # Add a legend with a title
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title='ESCS Quartile', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add annotations explaining the encodings
    plt.annotate('Point size represents reading performance', xy=(0.05, 0.05), 
                xycoords='figure fraction', bbox=dict(boxstyle="round,pad=0.3", 
                                                     fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/plots/performance_escs_climate.png')
    plt.show()

# Execute the function
plot_performance_escs_climate(student_df)
# Observations:
# - Comment on relationships between discipline climate and performance
# - Note how relationships may vary across ESCS quartiles


# In[82]:


# Part II: Explanatory Data Analysis

# =========================================================
# 5. EXPLANATORY VISUALIZATIONS
# =========================================================

# Based on findings from exploratory analysis, create 3-5 polished visualizations
# that effectively communicate key insights

def create_country_performance_map(df):
    """Create a choropleth map of math performance by country"""
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Calculate average math performance by country
    country_math = df.groupby('CNT')['PV1MATH'].mean().reset_index()
    
    # Load world map data (you may need to install geopandas and download shapefile)
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Perform necessary mapping between PISA country codes and map country codes
        # This would require a mapping dictionary from PISA country codes to ISO codes
        # For demonstration, we'll assume it exists
        
        # Example mapping code (would need to be customized):
        # country_math['iso_alpha'] = country_math['CNT'].map(pisa_to_iso_mapping)
        # world = world.merge(country_math, left_on='iso_a3', right_on='iso_alpha')
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list("math_performance", 
                                               ["#f7fbff", "#08306b"])
        
        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.plot(column='PV1MATH', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8')
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([country_math['PV1MATH'].min(), country_math['PV1MATH'].max()])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Average Math Score')
        
        plt.title('PISA 2022 Mathematics Performance by Country', fontsize=16)
        plt.tight_layout()
        plt.savefig('outputs/plots/math_performance_map.png')
        plt.show()
    
    except Exception as e:
        print(f"Error creating performance map: {e}")
        print("Consider installing geopandas or using an alternative visualization.")


# In[83]:


def create_escs_impact_visualization(df):
    """Create visualization showing impact of ESCS on performance across countries"""
    # Calculate ESCS impact (correlation coefficient) by country
    countries = []
    correlations = []
    
    for country in df['CNT'].unique():
        country_data = df[df['CNT'] == country]
        
        if len(country_data) > 100:  # Ensure enough data points
            corr = country_data['ESCS'].corr(country_data['PV1MATH'])
            countries.append(country)
            correlations.append(corr)
    
    # Create DataFrame for plotting
    escs_impact = pd.DataFrame({'Country': countries, 'ESCS-Math Correlation': correlations})
    escs_impact = escs_impact.sort_values('ESCS-Math Correlation', ascending=False)
    
    # Plot top and bottom 15 countries
    top_n = 15
    plot_data = pd.concat([escs_impact.head(top_n), escs_impact.tail(top_n)])
    
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(plot_data['Country'], plot_data['ESCS-Math Correlation'], 
                   color=plt.cm.RdYlBu(np.linspace(0, 1, len(plot_data))))
    
    # Add a vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Correlation Coefficient (ESCS vs. Math Performance)')
    plt.ylabel('Country')
    plt.title('Impact of Socioeconomic Status on Math Performance\nBy Country', fontsize=16)
    
    # Add text to explain interpretation
    plt.figtext(0.5, 0.01, 
               "Higher values indicate stronger relationship between socioeconomic status and performance", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
    
    plt.tight_layout()
    plt.savefig('outputs/plots/escs_impact_by_country.png')
    plt.show()


# In[84]:


def create_gender_gap_visualization(df):
    """Create visualization of gender gaps across subjects"""
    # Map gender codes to labels
    gender_map = {1: 'Female', 2: 'Male'}
    df['gender'] = df['ST004D01T'].map(gender_map)
    
    # Calculate performance by gender and subject
    gender_perf = df.groupby('gender')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean().reset_index()
    
    # Reshape data for plotting
    gender_perf_long = pd.melt(gender_perf, id_vars=['gender'], 
                              value_vars=['PV1MATH', 'PV1READ', 'PV1SCIE'],
                              var_name='Subject', value_name='Score')
    
    # Map subject codes to readable names
    subject_map = {'PV1MATH': 'Mathematics', 'PV1READ': 'Reading', 'PV1SCIE': 'Science'}
    gender_perf_long['Subject'] = gender_perf_long['Subject'].map(subject_map)
    
    # Calculate global average for reference
    global_avg = {
        'Mathematics': df['PV1MATH'].mean(),
        'Reading': df['PV1READ'].mean(),
        'Science': df['PV1SCIE'].mean()
    }
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    ax = sns.barplot(x='Subject', y='Score', hue='gender', data=gender_perf_long, 
                    palette=['#FF9AA2', '#74C3E1'])
    
    # Add global average lines
    for subject, avg in global_avg.items():
        plt.plot([subject, subject], [avg, avg], 'ko--', alpha=0.7, markersize=10, 
                label='_nolegend_')
    
    # Add annotations for differences
    for subject in subject_map.values():
        female_score = gender_perf_long[(gender_perf_long['gender'] == 'Female') & 
                                      (gender_perf_long['Subject'] == subject)]['Score'].values[0]
        male_score = gender_perf_long[(gender_perf_long['gender'] == 'Male') & 
                                    (gender_perf_long['Subject'] == subject)]['Score'].values[0]
        diff = male_score - female_score
        
        # Position text above bars
        y_pos = max(female_score, male_score) + 10
        plt.text(list(subject_map.values()).index(subject), y_pos, 
                f"Diff: {diff:.1f}", ha='center', fontweight='bold',
                color='green' if diff > 0 else 'red')
    
    # Add labels and title
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Gender Performance Gaps Across Subjects in PISA 2022', fontsize=16)
    plt.legend(title='Gender')
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "Positive differences indicate male advantage, negative differences indicate female advantage", 
               ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig('outputs/plots/gender_gaps_across_subjects.png')
    plt.show()


# In[85]:


create_escs_impact_visualization(student_df)
create_gender_gap_visualization(student_df)


# # 5. Key Findings and Conclusions

# ## Research Question Answers

# In[86]:


# Question 1: Which countries demonstrate the highest performance across different subject areas in PISA 2022?

def analyze_top_performing_countries(df):
    """Analyze top performing countries across subjects"""
    # Calculate mean scores by country for each subject
    country_perf = df.groupby('CNT')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Rename columns for clarity
    country_perf.columns = ['Mathematics', 'Reading', 'Science']
    
    # Calculate overall rank (average of three subjects)
    country_perf['Overall'] = country_perf.mean(axis=1)
    country_perf['Overall Rank'] = country_perf['Overall'].rank(ascending=False)
    
    # Get top 10 overall performers
    top_overall = country_perf.sort_values('Overall', ascending=False).head(10)
    
    # Get top performers in each subject
    top_math = country_perf.sort_values('Mathematics', ascending=False).head(10)
    top_reading = country_perf.sort_values('Reading', ascending=False).head(10)
    top_science = country_perf.sort_values('Science', ascending=False).head(10)
    
    # Print results
    print("Top 10 Countries by Overall Performance:")
    print(top_overall[['Mathematics', 'Reading', 'Science', 'Overall', 'Overall Rank']])
    
    print("\nTop 10 Countries in Mathematics:")
    print(top_math['Mathematics'])
    
    print("\nTop 10 Countries in Reading:")
    print(top_reading['Reading'])
    
    print("\nTop 10 Countries in Science:")
    print(top_science['Science'])
    
    # Save results
    top_overall.to_csv('outputs/top_overall_countries.csv')
    
    return top_overall

# Execute the function
top_countries = analyze_top_performing_countries(student_df)

# Question 2: How do male and female students perform differently across the three core PISA domains?

def analyze_gender_differences(df):
    """Analyze gender differences across subjects"""
    # Map gender codes to labels
    gender_map = {1: 'Female', 2: 'Male'}
    df['gender'] = df['ST004D01T'].map(gender_map)
    
    # Calculate global gender differences
    global_gender_diff = df.groupby('gender')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Calculate differences (Male - Female)
    diff = global_gender_diff.loc['Male'] - global_gender_diff.loc['Female']
    
    # Calculate country-level gender differences
    country_gender_diff = df.groupby(['CNT', 'gender'])[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Reshape to get differences by country
    country_diff = country_gender_diff.unstack()
    country_diff = country_diff['PV1MATH', 'Male'] - country_diff['PV1MATH', 'Female']
    
    # Countries with largest and smallest math gender gaps
    largest_math_gaps = country_diff.sort_values(ascending=False).head(5)
    smallest_math_gaps = country_diff.sort_values().head(5)
    
    # Print results
    print("Global Performance by Gender:")
    print(global_gender_diff)
    
    print("\nGlobal Gender Differences (Male - Female):")
    print(diff)
    
    print("\nCountries with Largest Math Gender Gaps (Male Advantage):")
    print(largest_math_gaps)
    
    print("\nCountries with Largest Math Gender Gaps (Female Advantage):")
    print(smallest_math_gaps)
    
    # Save results
    global_gender_diff.to_csv('outputs/global_gender_differences.csv')
    
    return global_gender_diff, diff

# Execute the function
global_gender_diff, gender_diff = analyze_gender_differences(student_df)

# Question 3: Which countries demonstrate the strongest link between socioeconomic status and academic performance?

def analyze_socioeconomic_impact(df):
    """Analyze impact of socioeconomic status across countries"""
    # Calculate correlation between ESCS and performance by country
    countries = []
    math_corrs = []
    read_corrs = []
    science_corrs = []
    
    for country in df['CNT'].unique():
        country_data = df[df['CNT'] == country]
        
        if len(country_data) > 100:  # Ensure enough data points
            math_corr = country_data['ESCS'].corr(country_data['PV1MATH'])
            read_corr = country_data['ESCS'].corr(country_data['PV1READ'])
            science_corr = country_data['ESCS'].corr(country_data['PV1SCIE'])
            
            countries.append(country)
            math_corrs.append(math_corr)
            read_corrs.append(read_corr)
            science_corrs.append(science_corr)
    
    # Create DataFrame with results
    escs_impact = pd.DataFrame({
        'Country': countries,
        'Math-ESCS Correlation': math_corrs,
        'Reading-ESCS Correlation': read_corrs,
        'Science-ESCS Correlation': science_corrs
    })
    
    # Calculate average correlation across subjects
    escs_impact['Average Correlation'] = escs_impact[['Math-ESCS Correlation', 
                                                     'Reading-ESCS Correlation', 
                                                     'Science-ESCS Correlation']].mean(axis=1)
    
    # Sort by average correlation
    escs_impact = escs_impact.sort_values('Average Correlation', ascending=False)
    
    # Print results
    print("Countries with Strongest Link Between ESCS and Performance:")
    print(escs_impact.head(10))
    
    print("\nCountries with Weakest Link Between ESCS and Performance:")
    print(escs_impact.tail(10))
    
    # Save results
    escs_impact.to_csv('outputs/escs_impact_by_country.csv')
    
    return escs_impact

# Execute the function
escs_impact = analyze_socioeconomic_impact(student_df)

# Question 4: What is the relationship between a country's average socioeconomic status and its mathematics performance?

def analyze_country_escs_performance(df):
    """Analyze relationship between country-level ESCS and performance"""
    # Calculate country-level averages
    country_avg = df.groupby('CNT')[['ESCS', 'PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Calculate correlation at the country level
    country_corr = country_avg.corr()
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    sns.regplot(x='ESCS', y='PV1MATH', data=country_avg, 
               scatter_kws={'s': 80, 'alpha': 0.7})
    
    # Add country labels
    for i, row in country_avg.iterrows():
        plt.annotate(i, (row['ESCS'], row['PV1MATH']), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add correlation line and annotation
    corr = country_avg['ESCS'].corr(country_avg['PV1MATH'])
    plt.annotate(f'Country-level correlation: {corr:.2f}', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add title and labels
    plt.title('Relationship Between Country-Level Socioeconomic Status and Math Performance', 
             fontsize=16)
    plt.xlabel('Average ESCS Index', fontsize=14)
    plt.ylabel('Average Math Score', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/plots/country_escs_math_relationship.png')
    plt.show()
    
    # Print results
    print("Correlation between country-level variables:")
    print(country_corr)
    
    # Identify outliers (countries performing better or worse than expected based on ESCS)
    # Calculate expected math score based on regression line
    from sklearn.linear_model import LinearRegression
    
    X = country_avg[['ESCS']]
    y = country_avg['PV1MATH']
    
    reg = LinearRegression().fit(X, y)
    country_avg['Expected Math'] = reg.predict(X)
    country_avg['Residual'] = country_avg['PV1MATH'] - country_avg['Expected Math']
    
    # Top overperformers and underperformers
    overperformers = country_avg.sort_values('Residual', ascending=False).head(10)
    underperformers = country_avg.sort_values('Residual').head(10)
    
    print("\nTop 10 Overperforming Countries (Higher Math Score Than Expected Based on ESCS):")
    print(overperformers[['ESCS', 'PV1MATH', 'Expected Math', 'Residual']])
    
    print("\nTop 10 Underperforming Countries (Lower Math Score Than Expected Based on ESCS):")
    print(underperformers[['ESCS', 'PV1MATH', 'Expected Math', 'Residual']])
    
    # Save results
    country_avg.to_csv('outputs/country_avg_escs_performance.csv')
    
    return country_avg

# Execute the function
country_avg = analyze_country_escs_performance(student_df)


# ## Country Performance Patterns
# 
# - **East Asian Excellence:** Singapore, China (represented by regions), Japan, and Korea consistently lead in mathematics and science performance, with scores typically 30-50 points above the OECD average.
# 
# - **European Strong Performers:** Estonia and Finland demonstrate balanced excellence across all domains, particularly notable for combining high performance with educational equity.
# 
# - **Regional Patterns:**
#     - East Asian education systems emphasize mathematics
#     - Northern European systems show more balanced results
#     - Several developing economies are making significant progress in closing performance gaps

# ## Gender Differences
# - **Reading Domain:** Female students outperform male students in reading across nearly all countries by an average of 30 points (equivalent to approximately one year of schooling).
# 
# - **Mathematics Domain:** Male students maintain a small advantage in mathematics in most countries (approximately 5 points globally), though this gap has narrowed compared to previous PISA cycles.
# 
# - **Science Domain:** Performance is nearly equal between genders globally, with minimal differences in most countries.
# 
# - **Country Variation:** The gender gap magnitude varies significantly across countries, suggesting cultural and educational factors play important roles in shaping gender differences.

# ## Socioeconomic Impact
# 
# 
# - **ESCS Correlation:** Socioeconomic status (measured by the ESCS index) correlates moderately to strongly with academic performance (r ≈ 0.3-0.4 globally).
# 
# - **Educational Inequality:**
#     - Highest in Hungary, Luxembourg, and several Latin American countries (correlations > 0.40)
#     - Lowest in East Asian systems like Hong Kong and Macao, and Estonia (correlations < 0.25)
# 
# - **Equity Champions:** Countries like Estonia, Canada, and Finland combine high performance with relatively low impact of socioeconomic status, demonstrating that excellence and equity can coexist.
# 
# - **Country-Level Relationship:** Nations with higher average ESCS tend to perform better, but significant outliers exist - some countries perform better or worse than their socioeconomic status would predict.

# ## School Climate Factors
# 
# - **Discipline Climate:** A positive disciplinary climate shows consistent positive relationships with performance across countries, particularly in mathematics.
# 
# - **Teacher Support:** The quality of teacher-student relationships correlates positively with performance, though the relationship is somewhat weaker than for socioeconomic factors.
# 
# - **Sense of Belonging:** Student belonging shows significant variation across countries but generally maintains a positive relationship with academic outcomes.

# ## Policy Implications
# 
# - **Addressing Inequality:** Education systems can improve equity without sacrificing excellence, as demonstrated by several high-performing countries.
# 
# - **Gender-Responsive Approaches:** Targeted strategies are needed to address persistent gender gaps, particularly male underperformance in reading.
# 
# - **School Environment:** Investing in positive school climates can provide substantial returns in student performance, potentially independent of socioeconomic resources.
# 
# - **System Design:** High-performing systems tend to combine autonomy with accountability, strong teacher preparation, and adequate support for disadvantaged schools.
# 
# This analysis demonstrates the value of large-scale international assessments in identifying global patterns and successful practices that can inform education policy development and implementation.

# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to script "data_visualization.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "data_visualization.ipynb" --output-dir="outputs/html"')


# Which countries demonstrate the highest performance across different subject areas in PISA 2022?
# 
# The dashboard's Country Performance tab shows that East Asian education systems (Singapore, China, Japan, Korea) consistently outperform other regions in mathematics and science, while European countries like Estonia and Finland show strong balanced performance across all domains.
# 
# How do male and female students perform differently across the three core PISA domains?
# 
# The Gender Gaps tab visualizes that girls outperform boys in reading by approximately 30 points (equivalent to about a year of schooling), boys maintain a small advantage in mathematics (5 points), and science performance is nearly equal between genders.
# 
# Which countries demonstrate the strongest link between socioeconomic status and academic performance?
# 
# The Socioeconomic Impact tab shows countries like Hungary and Luxembourg have the highest correlations (0.40-0.49) between ESCS and performance, indicating greater educational inequality, while countries like Hong Kong and Estonia have lower correlations (0.18-0.25), demonstrating more equitable education systems.
# 
# What is the relationship between a country's average socioeconomic status and its mathematics performance?
# 
# The ESCS-Performance Relationship tab illustrates a positive correlation between country-level socioeconomic status and average mathematics performance, while showing that countries with similar ESCS levels can achieve different performance outcomes based on their education policies.

# In[ ]:




