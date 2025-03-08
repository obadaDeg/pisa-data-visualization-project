#!/usr/bin/env python
# coding: utf-8

# In[10]:


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



# In[11]:


# csv_file = "dataset/pisa.csv"
# df.to_csv(csv_file, index=False)


# In[12]:


student_df = pd.read_csv("dataset/pisa.csv")
student_df.shape


# In[13]:


student_df.info()


# In[14]:


student_df.describe().to_csv("outputs/pisa_describe.csv")


# In[15]:


student_df.describe().info()


# In[16]:


student_df.head()


# In[17]:


# textual_columns = student_df.select_dtypes(include=['object']).columns
# numerical_columns = student_df.select_dtypes(include=['number']).columns
# textual_columns, numerical_columns


# In[18]:


# student_df[textual_columns].head()


# In[19]:


# student_df['CNT'].value_counts()
# # convert to categorical
# student_df['CNT'] = student_df['CNT'].astype('category')


# In[20]:


# student_df['CYC'].value_counts()
# # convert to categorical
# # student_df['CNT'] = student_df['CNT'].astype('category')


# In[21]:


# student_df['STRATUM'].value_counts()


# In[22]:


def memory_usage(pandas_obj):
    """Calculate memory usage of a pandas object in MB"""
    if isinstance(pandas_obj, pd.DataFrame):
        usage_bytes = pandas_obj.memory_usage(deep=True).sum()
    else:  # Series
        usage_bytes = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_bytes / (1024 * 1024)
    return usage_mb


# In[23]:


def optimize_floats(df):
    """Optimize float dtypes by downcasting to float32 where possible"""
    float_cols = df.select_dtypes(include=['float64']).columns
    
    for col in float_cols:
        # Check if column can be represented as float32 without losing precision
        # For PISA data, most measurements don't need float64 precision
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


# In[24]:


def optimize_ints(df):
    """Optimize integer dtypes by downcasting to smallest possible integer type"""
    int_cols = df.select_dtypes(include=['int64']).columns
    
    for col in int_cols:
        # For each column, downcast to the smallest possible integer type
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df


# In[25]:


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


# In[26]:


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


# In[27]:


def read_pisa_in_chunks(filepath, chunksize=100000, optimize=True, output_file=None):
    """
    Read and process PISA data in chunks to reduce memory usage
    
    Parameters:
    filepath (str): Path to the PISA CSV file
    chunksize (int): Number of rows to read at once
    optimize (bool): Whether to apply memory optimizations
    output_file (str): Path to save optimized CSV (if None, doesn't save)
    
    Returns:
    pd.DataFrame: The optimized dataframe (if output_file is None), otherwise None
    """
    # Get the total number of rows to track progress
    total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract 1 for header
    
    # If we're saving to a file, process chunk by chunk without keeping in memory
    if output_file:
        print(f"Processing {filepath} in chunks of {chunksize} rows")
        print(f"Total rows to process: {total_rows}")
        
        # Process the first chunk to get column dtypes for future chunks
        first_chunk = pd.read_csv(filepath, nrows=chunksize)
        
        if optimize:
            print("Optimizing first chunk to determine dtypes...")
            first_chunk = optimize_floats(first_chunk)
            first_chunk = optimize_ints(first_chunk)
            first_chunk = optimize_categorical(first_chunk)
            first_chunk = optimize_known_pisa_columns(first_chunk)
        
        # Get optimized dtypes
        optimized_dtypes = first_chunk.dtypes
        
        # Write the first chunk to file with header
        first_chunk.to_csv(output_file, mode='w', index=False)
        
        # Process the rest of the file in chunks
        rows_processed = len(first_chunk)
        
        # Free memory
        del first_chunk
        gc.collect()
        
        for chunk in pd.read_csv(filepath, chunksize=chunksize, skiprows=range(1, rows_processed+1)):
            # Apply dtype conversions based on optimized first chunk
            for col in chunk.columns:
                if col in optimized_dtypes:
                    chunk[col] = chunk[col].astype(optimized_dtypes[col])
            
            # Append to the output file
            chunk.to_csv(output_file, mode='a', header=False, index=False)
            
            # Update progress
            rows_processed += len(chunk)
            progress = (rows_processed / total_rows) * 100
            print(f"Processed {rows_processed:,}/{total_rows:,} rows ({progress:.1f}%)")
            
            # Free memory
            del chunk
            gc.collect()
        
        print(f"Optimized data saved to {output_file}")
        return None
    
    # If we're not saving to a file, read the entire dataset and return it
    else:
        print(f"Reading entire dataset into memory from {filepath}")
        df = pd.read_csv(filepath)
        
        original_memory = memory_usage(df)
        
        if optimize:
            print("Applying optimizations...")
            df = optimize_floats(df)
            df = optimize_ints(df)
            df = optimize_categorical(df)
            df = optimize_known_pisa_columns(df)
            
            optimized_memory = memory_usage(df)
            
            print(f"Optimized memory usage: {optimized_memory:.2f} MB")
            print(f"Memory usage reduced by: {original_memory - optimized_memory:.2f} MB ({((original_memory - optimized_memory) / original_memory) * 100:.1f}%)")
        
        return df


# In[28]:


student_df = optimize_floats(student_df)
student_df = optimize_ints(student_df)
student_df = optimize_categorical(student_df)
student_df = optimize_known_pisa_columns(student_df)


student_df.info()


# In[29]:


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


# In[30]:


os.makedirs('outputs/cleaning', exist_ok=True)

def clean_pisa_data(df, save_path=None):
    """
    Comprehensive data cleaning for PISA dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        The raw PISA dataset
    save_path : str, optional
        Path to save the cleaned dataset
        
    Returns:
    --------
    pandas DataFrame
        The cleaned PISA dataset
    """
    print(f"Starting data cleaning process on DataFrame with shape: {df.shape}")
    original_size = len(df)
    
    # Step 1: Check for duplicate student IDs
    print("\nStep 1: Checking for duplicate student IDs...")
    if 'CNTSTUID' in df.columns:
        duplicate_ids = df['CNTSTUID'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"Found {duplicate_ids} duplicate student IDs. Keeping first occurrence.")
            df = df.drop_duplicates(subset='CNTSTUID', keep='first')
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
    
    # Step 3: Handle specific cases of missingness
    print("\nStep 3: Handling specific missing value patterns...")
    
    # For performance variables (PVs), we cannot impute - students must have scores
    perf_vars = [col for col in key_performance if col in df.columns]
    missing_perf = df[perf_vars].isnull().any(axis=1)
    if missing_perf.sum() > 0:
        print(f"Removing {missing_perf.sum()} students with missing performance scores.")
        df = df[~missing_perf]
    
    # For demographic variables, we'll keep track of missingness with flags
    for col in key_demographics:
        if col in df.columns and df[col].isnull().sum() > 0:
            # Create a missing flag
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            
            # For categorical variables, create a "Missing" category
            if df[col].dtype == 'category' or df[col].nunique() < 10:
                if df[col].dtype != 'category':
                    df[col] = df[col].astype('category')
                df[col] = df[col].cat.add_categories(['Missing'])
                df.loc[df[col].isnull(), col] = 'Missing'
            else:
                # For continuous variables, impute with median
                df[col] = df[col].fillna(df[col].median())
    
    # For SES variables, we'll try to impute ESCS if missing but HOMEPOS and HISCED are available
    if 'ESCS' in df.columns and df['ESCS'].isnull().sum() > 0:
        ses_vars = [col for col in key_ses if col in df.columns and col != 'ESCS']
        if len(ses_vars) > 0:
            # Create missing flag
            df['ESCS_missing'] = df['ESCS'].isnull().astype(int)
            
            # Simple imputation: Use correlation with available SES variables
            from sklearn.impute import SimpleImputer
            
            # Only impute if the missing rate is reasonable (<30%)
            missing_rate = df['ESCS'].isnull().mean()
            if missing_rate < 0.3:
                print(f"Imputing ESCS for {df['ESCS'].isnull().sum()} students.")
                
                # Prepare data for imputation
                impute_df = df[ses_vars + ['ESCS']].copy()
                
                # Create imputer
                imputer = SimpleImputer(strategy='median')
                imputed = imputer.fit_transform(impute_df)
                
                # Replace only the missing values
                df.loc[df['ESCS'].isnull(), 'ESCS'] = imputed[df['ESCS'].isnull(), -1]
            else:
                print(f"ESCS missing rate too high ({missing_rate:.1%}) for reliable imputation.")
    
    # Step 4: Check for and handle outliers
    print("\nStep 4: Checking for outliers in key continuous variables...")
    
    continuous_vars = []
    for col in existing_key_vars:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            if df[col].nunique() > 10:  # Simple heuristic for continuous variables
                continuous_vars.append(col)
    
    # Report on outliers (values > 3 standard deviations from mean)
    outlier_report = {}
    for col in continuous_vars:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z_scores > 3).sum()
        outlier_report[col] = {
            'Count': outliers,
            'Percentage': (outliers / len(df)) * 100
        }
    
    outlier_df = pd.DataFrame(outlier_report).T
    outlier_df.columns = ['Outlier Count', 'Outlier Percentage']
    outlier_df = outlier_df.sort_values('Outlier Count', ascending=False)
    
    print("Outliers summary (values > 3 std from mean):")
    print(outlier_df[outlier_df['Outlier Count'] > 0])
    
    # Save the outlier report
    outlier_df.to_csv('outputs/cleaning/outlier_report.csv')
    
    # For most PISA variables, outliers are meaningful and should be kept
    # For age, we might want to restrict to appropriate school ages
    if 'AGE' in df.columns:
        age_outliers = (df['AGE'] < 10) | (df['AGE'] > 20)
        if age_outliers.sum() > 0:
            print(f"Removing {age_outliers.sum()} students with implausible ages (<10 or >20).")
            df = df[~age_outliers]
    
    # Step 5: Check and fix inconsistent categorical values
    print("\nStep 5: Checking for inconsistent categorical values...")
    
    cat_vars = [col for col in existing_key_vars 
                if col not in continuous_vars and df[col].dtype != 'category']
    
    for col in cat_vars:
        # Convert to categorical if it has relatively few unique values
        if df[col].nunique() < 50:
            print(f"Converting {col} to categorical type. Has {df[col].nunique()} unique values.")
            df[col] = df[col].astype('category')
    
    # Step 6: Create report on final dataset
    print("\nStep 6: Creating final dataset report...")
    
    rows_removed = original_size - len(df)
    cleaning_report = {
        'Original rows': original_size,
        'Rows after cleaning': len(df),
        'Rows removed': rows_removed,
        'Percentage removed': (rows_removed / original_size) * 100
    }
    
    print("\nCleaning summary:")
    for key, value in cleaning_report.items():
        print(f"{key}: {value}")
    
    # Save the cleaning report
    pd.DataFrame([cleaning_report]).to_csv('outputs/cleaning/cleaning_summary.csv', index=False)
    
    # Step 7: Save the cleaned dataset
    if save_path:
        print(f"\nSaving cleaned dataset to {save_path}")
        df.to_csv(save_path, index=False)
    
    print("\nData cleaning completed successfully.")
    return df


# In[31]:


student_df = clean_pisa_data(student_df, save_path='outputs/cleaning/pisa_cleaned.csv')
# quick checking for cleaning process


# In[32]:


student_df.info()


# In[33]:


os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# ANALYSIS 1: Performance across countries
def analyze_performance(df):
    print("\nAnalyzing performance across countries...")
    # Calculate mean performance for each country (using first PV)
    performance_by_country = df.groupby('CNT')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Sort by math performance
    performance_by_country = performance_by_country.sort_values('PV1MATH', ascending=False)
    
    # Calculate global averages
    global_avg = {
        'Math': df['PV1MATH'].mean(),
        'Reading': df['PV1READ'].mean(),
        'Science': df['PV1SCIE'].mean()
    }
    
    # Save results
    performance_by_country.to_csv('outputs/tables/performance_by_country.csv')
    
    # Create visualizations
    plot_country_performance(performance_by_country)
    
    return performance_by_country, global_avg

# ANALYSIS 2: Socioeconomic status and performance
def analyze_ses_performance(df):
    print("\nAnalyzing relationship between socioeconomic status and performance...")
    # Create ESCS quartiles
    df['ESCS_quartile'] = pd.qcut(df['ESCS'], 4, labels=['Bottom 25%', 'Lower middle', 'Upper middle', 'Top 25%'])
    
    # Calculate performance by ESCS quartile
    performance_by_ses = df.groupby('ESCS_quartile')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Calculate correlation
    ses_math_corr = df[['ESCS', 'PV1MATH']].corr().iloc[0, 1]
    ses_read_corr = df[['ESCS', 'PV1READ']].corr().iloc[0, 1]
    ses_scie_corr = df[['ESCS', 'PV1SCIE']].corr().iloc[0, 1]
    
    correlations = {
        'SES-Math': ses_math_corr,
        'SES-Reading': ses_read_corr,
        'SES-Science': ses_scie_corr
    }
    
    # Calculate SES impact by country
    ses_impact_by_country = df.groupby('CNT').apply(lambda x: x[['ESCS', 'PV1MATH']].corr().iloc[0, 1])
    ses_impact_by_country = ses_impact_by_country.sort_values(ascending=False)
    
    # Save results
    performance_by_ses.to_csv('outputs/tables/performance_by_ses_quartile.csv')
    pd.DataFrame([correlations]).to_csv('outputs/tables/ses_performance_correlations.csv')
    ses_impact_by_country.to_frame('SES-Math Correlation').to_csv('outputs/tables/ses_impact_by_country.csv')
    
    # Create visualizations
    plot_ses_performance(df)
    plot_ses_impact_by_country(ses_impact_by_country)
    
    return performance_by_ses, correlations, ses_impact_by_country

# ANALYSIS 3: Gender differences
def analyze_gender_differences(df):
    print("\nAnalyzing gender differences in performance...")
    # ST004D01T is typically the gender variable
    df['gender'] = df['ST004D01T'].map({1: 'Female', 2: 'Male'})
    
    gender_performance = df.groupby('gender')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Gender gap by country
    gender_gap_by_country = df.groupby(['CNT', 'gender'])[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean().unstack()
    
    # Calculate the gender gap (male - female)
    math_gap = gender_gap_by_country['PV1MATH']['Male'] - gender_gap_by_country['PV1MATH']['Female']
    reading_gap = gender_gap_by_country['PV1READ']['Male'] - gender_gap_by_country['PV1READ']['Female']
    science_gap = gender_gap_by_country['PV1SCIE']['Male'] - gender_gap_by_country['PV1SCIE']['Female']
    
    gender_gaps = pd.DataFrame({
        'Math Gap': math_gap,
        'Reading Gap': reading_gap,
        'Science Gap': science_gap
    })
    
    # Save results
    gender_performance.to_csv('outputs/tables/gender_performance.csv')
    gender_gaps.to_csv('outputs/tables/gender_gaps_by_country.csv')
    
    # Create visualizations
    plot_gender_differences(gender_performance)
    plot_gender_gaps_by_country(gender_gaps)
    
    return gender_performance, gender_gaps

# ANALYSIS 4: School climate and performance
def analyze_school_climate(df):
    print("\nAnalyzing impact of school climate on performance...")
    
    # Select relevant school climate variables
    climate_vars = ['DISCLIM', 'TEACHSUP', 'BELONG', 'FEELSAFE', 'SCHRISK']
    
    # Calculate correlations with performance
    climate_math_corrs = {}
    
    for var in climate_vars:
        if var in df.columns:
            climate_math_corrs[var] = df[[var, 'PV1MATH']].corr().iloc[0, 1]
    
    # Create a climate index (average of standardized climate variables)
    climate_df = df[climate_vars].copy()
    
    # Standardize values
    for var in climate_vars:
        if var in climate_df.columns:
            climate_df[var] = (climate_df[var] - climate_df[var].mean()) / climate_df[var].std()
    
    df['climate_index'] = climate_df.mean(axis=1)
    
    # Calculate performance by climate index quartile
    df['climate_quartile'] = pd.qcut(df['climate_index'], 4, 
                                    labels=['Bottom 25%', 'Lower middle', 'Upper middle', 'Top 25%'])
    
    performance_by_climate = df.groupby('climate_quartile')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Save results
    pd.DataFrame([climate_math_corrs]).T.rename(columns={0:'Math Correlation'}).to_csv(
        'outputs/tables/climate_correlations.csv')
    performance_by_climate.to_csv('outputs/tables/performance_by_climate.csv')
    
    # Create visualizations
    plot_climate_correlations(climate_math_corrs)
    plot_performance_by_climate(performance_by_climate)
    
    return climate_math_corrs, performance_by_climate

# ANALYSIS 5: Immigrant status and performance
def analyze_immigrant_status(df):
    print("\nAnalyzing impact of immigrant status on performance...")
    
    # IMMIG values: 1=Native, 2=Second-generation, 3=First-generation
    df['immigrant_status'] = df['IMMIG'].map({
        1: 'Native', 
        2: 'Second-generation', 
        3: 'First-generation'
    })
    
    # Performance by immigrant status
    immig_performance = df.groupby('immigrant_status')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    
    # Performance gap by country
    immig_gap_by_country = df.groupby(['CNT', 'immigrant_status'])[['PV1MATH']].mean().unstack()
    
    # Calculate the gap (native - immigrant)
    if 'Native' in immig_gap_by_country['PV1MATH'].columns and 'First-generation' in immig_gap_by_country['PV1MATH'].columns:
        math_gap = immig_gap_by_country['PV1MATH']['Native'] - immig_gap_by_country['PV1MATH']['First-generation']
        math_gap = math_gap.sort_values(ascending=False)
        
        # Save results
        immig_performance.to_csv('outputs/tables/performance_by_immigrant_status.csv')
        math_gap.to_frame('Native-Immigrant Math Gap').to_csv('outputs/tables/immigrant_gap_by_country.csv')
        
        # Create visualizations
        plot_immigrant_performance(immig_performance)
        plot_immigrant_gap_by_country(math_gap)
        
        return immig_performance, math_gap
    else:
        print("Warning: Immigrant status categories not found as expected. Check your data.")
        return immig_performance, None

# VISUALIZATION FUNCTIONS

def plot_country_performance(performance_df, top_n=15):
    top_countries = performance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    ax = top_countries.plot(kind='barh', figsize=(12, 8))
    
    plt.title(f'Top {top_n} Countries by Math Performance')
    plt.xlabel('Average Score')
    plt.ylabel('Country')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('outputs/plots/top_countries_performance.png')
    plt.close()

def plot_ses_performance(df, sample_size=5000):
    # Take a random sample to make plotting faster
    sample_df = df.sample(min(sample_size, len(df)))
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x='ESCS', y='PV1MATH', data=sample_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    
    plt.title('Relationship between Socioeconomic Status and Math Performance')
    plt.xlabel('ESCS (Economic, Social and Cultural Status Index)')
    plt.ylabel('Math Score (PV1MATH)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('outputs/plots/ses_math_relationship.png')
    plt.close()

def plot_ses_impact_by_country(ses_impact, top_n=15):
    plt.figure(figsize=(12, 8))
    
    # Plot top and bottom countries by SES impact
    top_countries = ses_impact.head(top_n)
    bottom_countries = ses_impact.tail(top_n)
    
    combined = pd.concat([top_countries, bottom_countries])
    combined.plot(kind='barh', figsize=(12, 8))
    
    plt.title(f'Countries with Highest and Lowest SES-Math Correlation')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Country')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.tight_layout()
    
    plt.savefig('outputs/plots/ses_impact_by_country.png')
    plt.close()

def plot_gender_differences(gender_performance):
    plt.figure(figsize=(10, 6))
    
    gender_performance.plot(kind='bar', figsize=(10, 6))
    
    plt.title('Performance by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('outputs/plots/gender_performance.png')
    plt.close()

def plot_gender_gaps_by_country(gender_gaps, top_n=15):
    plt.figure(figsize=(12, 10))
    
    # Sort by reading gap (typically shows largest gender differences)
    sorted_gaps = gender_gaps.sort_values('Reading Gap')
    
    # Plot countries with largest gaps in favor of females and males
    top_female = sorted_gaps.head(top_n)
    top_male = sorted_gaps.tail(top_n)
    
    # Combine and plot
    combined_gaps = pd.concat([top_female, top_male])
    combined_gaps[['Reading Gap']].plot(kind='barh', figsize=(12, 10), color='purple')
    
    plt.title(f'Countries with Largest Reading Gender Gaps')
    plt.xlabel('Score Difference (Male - Female)')
    plt.ylabel('Country')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.tight_layout()
    
    plt.savefig('outputs/plots/reading_gender_gaps.png')
    plt.close()

def plot_climate_correlations(climate_corrs):
    plt.figure(figsize=(10, 6))
    
    climate_df = pd.DataFrame.from_dict(climate_corrs, orient='index', columns=['Math Correlation'])
    climate_df.plot(kind='bar', figsize=(10, 6))
    
    plt.title('Correlation between School Climate Factors and Math Performance')
    plt.xlabel('School Climate Factor')
    plt.ylabel('Correlation with Math Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('outputs/plots/climate_correlations.png')
    plt.close()

def plot_performance_by_climate(performance_by_climate):
    plt.figure(figsize=(10, 6))
    
    performance_by_climate.plot(kind='bar', figsize=(10, 6))
    
    plt.title('Performance by School Climate Quartiles')
    plt.xlabel('School Climate Quartile')
    plt.ylabel('Average Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('outputs/plots/performance_by_climate.png')
    plt.close()

def plot_immigrant_performance(immig_performance):
    plt.figure(figsize=(10, 6))
    
    immig_performance.plot(kind='bar', figsize=(10, 6))
    
    plt.title('Performance by Immigrant Status')
    plt.xlabel('Immigrant Status')
    plt.ylabel('Average Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('outputs/plots/immigrant_performance.png')
    plt.close()

def plot_immigrant_gap_by_country(math_gap, top_n=15):
    plt.figure(figsize=(12, 8))
    
    # Get countries with largest gaps in both directions
    top_gaps = math_gap.head(top_n)
    bottom_gaps = math_gap.tail(top_n)
    
    combined_gaps = pd.concat([top_gaps, bottom_gaps])
    combined_gaps.plot(kind='barh', figsize=(12, 8))
    
    plt.title(f'Countries with Largest Native-Immigrant Math Gaps')
    plt.xlabel('Score Difference (Native - Immigrant)')
    plt.ylabel('Country')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.tight_layout()
    
    plt.savefig('outputs/plots/immigrant_gap_by_country.png')
    plt.close()

# Run all analyses
def run_all_analyses(df):
    print("Starting PISA data analysis...")
    # Check for required columns
    required_columns = ['CNT', 'PV1MATH', 'PV1READ', 'PV1SCIE', 'ESCS', 'ST004D01T', 'IMMIG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following required columns are missing: {missing_columns}")
        print("Some analyses may not run correctly.")
    
    # Run analyses
    performance_results = analyze_performance(df)
    ses_results = analyze_ses_performance(df)
    gender_results = analyze_gender_differences(df)
    climate_results = analyze_school_climate(df)
    immigrant_results = analyze_immigrant_status(df)
    
    print("\nAnalysis complete! Results saved to 'outputs/tables/' and plots saved to 'outputs/plots/'")
    
    return {
        'performance': performance_results,
        'ses': ses_results,
        'gender': gender_results,
        'climate': climate_results,
        'immigrant': immigrant_results
    }

# Execute the script
if __name__ == "__main__":
    run_all_analyses(student_df)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "data_visualization.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "data_visualization.ipynb" --output-dir="outputs/html"')

