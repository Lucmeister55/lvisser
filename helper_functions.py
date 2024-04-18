import pandas as pd
import pysam
import numpy as np
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import sqlite3
import csv
import glob
import gffpandas.gffpandas as gffpd
from intervaltree import Interval, IntervalTree
import bisect
from tqdm import tqdm
from biomart import BiomartServer
import os
import re
import gseapy as gp
from gseapy.plot import barplot, dotplot
from gseapy import GSEA
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
from scipy.stats import kruskal, mannwhitneyu, ttest_ind, f_oneway, wilcoxon, ranksums, fisher_exact, chi2_contingency, entropy, pearsonr, zscore
import json
import ast
import itertools
import sys
from venn import venn
import contextlib
from imblearn.over_sampling import SMOTE
from upsetplot import from_contents, plot
import warnings
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import multipletests
import shap
import pyranges as pr

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN

def assert_column(df, column_name, df_name=''):
        assert column_name in df.columns, f"'{column_name}' column is missing in {df_name}"
        assert not df[column_name].isnull().all(), f"'{column_name}' column is empty in {df_name}"

def cpg2segment_aggregation_trees(cpg_df, segment_df):

    dataframes = {'cpg_df': ['source_directory', 'sample_id_adj', 'chrom'],
                'segment_df': ['start', 'end', 'segment_id', 'chrom', 'gene_symbol', 'length']}

    for df_name, columns in dataframes.items():
        for column in columns:
            assert_column(eval(df_name), column, df_name)

    # Filter out chromosomes from cpg_df that aren't in segment_df
    cpg_df = cpg_df[cpg_df['chrom'].isin(segment_df['chrom'].unique())]

    # Create an IntervalTree for each group
    segment_trees = {str(seq_id): IntervalTree(Interval(row.start, row.end, row.segment_id) for row in group.itertuples()) 
                     for seq_id, group in segment_df.groupby('chrom')}
    
    print("initializing meth_seg...")

    # Initialize meth_seg DataFrame
    meth_seg = pd.DataFrame(list(itertools.product(cpg_df["sample_id_adj"].unique(), 
                                                   segment_df["segment_id"].unique())),
                            columns=['sample_id_adj', 'segment_id'])
    
    meth_seg = meth_seg.merge(cpg_df[['sample_id_adj', 'source_directory']].drop_duplicates(), on='sample_id_adj', how='left')

    meth_seg = meth_seg.sort_values(['sample_id_adj']).reset_index(drop=True)
    meth_seg["total_methylation"] = 0.0
    meth_seg["positions"] = [[] for _ in range(len(meth_seg))]
    meth_seg["mod_qual_positions"] = [[] for _ in range(len(meth_seg))]
    meth_seg["num_cpgs"] = 0

    # Initialize a dictionary to store the rows
    meth_seg_dict = {(row.segment_id, row.sample_id_adj, row.source_directory): row for _, row in meth_seg.iterrows()}

    # Group the DataFrame by 'source_directory', 'sample_id_adj', and 'chrom'
    grouped = cpg_df.groupby(['source_directory', 'sample_id_adj', 'chrom'])

    for (source_directory, sample_id_adj, chrom), group in tqdm(grouped, desc="Aggregating"):
        tree = segment_trees[chrom]
        for row in group.itertuples():
            intervals = tree[row.ref_position]
            for interval in intervals:
                # Use the dictionary for lookup and update
                key = (interval.data, sample_id_adj, source_directory)
                meth_seg_row = meth_seg_dict[key]
                meth_seg_row.total_methylation += row.mod_qual
                meth_seg_row.num_cpgs += 1  # Increment the number of CpGs for this segment
                meth_seg_row.positions.append(row.ref_position)
                meth_seg_row.mod_qual_positions.append(row.mod_qual)

    # Convert the dictionary back to a DataFrame
    meth_seg = pd.DataFrame(meth_seg_dict.values())

    # Calculate the average of the values in the 'mod_quals_positions' column
    meth_seg['avg_methylation'] = meth_seg['mod_qual_positions'].apply(lambda x: np.mean(x) if x else 0)

    meth_seg = meth_seg.merge(segment_df[['segment_id', 'gene_symbol', 'length', 'chrom']], on='segment_id', how='left').sort_values("segment_id")

    # Convert the lists to strings
    meth_seg['positions'] = meth_seg['positions'].astype(str)
    meth_seg['mod_qual_positions'] = meth_seg['mod_qual_positions'].astype(str)

    # Now you can safely drop duplicates
    meth_seg = meth_seg.drop_duplicates().reset_index(drop=True)

    return meth_seg

def create_fm(meth_seg, metadata):
    dataframes = {
        'meth_seg': ['segment_id', 'source_directory', 'sample_id_adj', 'avg_methylation'],
        'metadata': ['source_directory', 'sample_id_adj', 'Group']
    }

    for df_name, columns in dataframes.items():
        for column in columns:
            assert_column(eval(df_name), column, df_name)

    meth_seg_pivot = meth_seg.pivot_table(index=["sample_id_adj", "source_directory"], columns="segment_id", values="avg_methylation").reset_index()
    # Fill missing values with zeros
    meth_seg_pivot = meth_seg_pivot.fillna(0)
    meth_seg_fm = meth_seg_pivot.merge(metadata[["sample_id_adj", "Group", "tumor_type"]], on = "sample_id_adj", how = "left").drop_duplicates().reset_index(drop = True)
    # Assuming 'Group' is the column you want to move to the front
    group = meth_seg_fm.pop('Group')
    meth_seg_fm.insert(0, 'Group', group)
    meth_seg_fm.sort_values("source_directory", inplace = True)

    # Remove all zero columns
    meth_seg_fm = meth_seg_fm.loc[:, (meth_seg_fm != 0).any(axis=0)]

    return meth_seg_fm

def create_fm_wgbs(df, metadata):
    processed_df = df.select_dtypes(include=[float])
    processed_df['segment_id'] = df['segment_id']

    processed_df = processed_df.set_index('segment_id').dropna()
    processed_df = processed_df[processed_df.index.isin(df["segment_id"])].T

    # Reset the index and rename it to 'sample_id_adj'
    processed_df.reset_index(inplace=True)
    processed_df = processed_df.rename(columns={'index': 'sample_id_adj'})

    processed_df = processed_df.merge(metadata[["sample_id_adj", "source_directory", "Group", "tumor_type"]], on = "sample_id_adj", how = "left")
    
    return processed_df

def find_annot_overlap(df, annot):

    dataframes = {
        'df': ['segment_id'],
        'annot': ['segment_id', 'gene_symbol']
    }

    for df_name, columns in dataframes.items():
        for column in columns:
            assert_column(eval(df_name), column, df_name)

    warnings.filterwarnings('ignore', category=FutureWarning)

    list1 = df.segment_id
    list2 = annot.segment_id

    # Convert the lists to PyRanges objects
    def convert_to_pyranges(list_of_segments):
        chromosomes = [seg.split(':')[0] for seg in list_of_segments]
        starts = [int(seg.split(':')[1].split('-')[0]) for seg in list_of_segments]
        ends = [int(seg.split(':')[1].split('-')[1]) for seg in list_of_segments]
        return pr.PyRanges(chromosomes=chromosomes, starts=starts, ends=ends)

    pr1 = convert_to_pyranges(list1)
    pr2 = convert_to_pyranges(list2)

    # Find overlapping segments
    overlapping_segments = pr1.join(pr2)

    # Convert to DataFrame
    overlapping_segments_df = overlapping_segments.df

    # Add segment_id column
    overlapping_segments_df['segment_id'] = overlapping_segments_df['Chromosome'].astype(str) + ':' + overlapping_segments_df['Start_b'].astype(str) + '-' + overlapping_segments_df['End_b'].astype(str)

    # Merge with gene_bodies_105 to add gene symbols
    overlapping_segments_df = overlapping_segments_df.merge(annot[["segment_id", "gene_symbol"]], on='segment_id', how='left').drop_duplicates()

    # Add a column for the length of the overlap
    overlapping_segments_df['overlap_length'] = overlapping_segments_df['End_b'] - overlapping_segments_df['Start_b']

    # Group by segment_id and keep only the row with the largest overlap for each segment_id
    overlapping_segments_df = overlapping_segments_df.loc[overlapping_segments_df.groupby('segment_id')['overlap_length'].idxmax()]

    overlapping_segments_df['segment_id'] = overlapping_segments_df['Chromosome'].astype(str) + ':' + overlapping_segments_df['Start'].astype(str) + '-' + overlapping_segments_df['End'].astype(str)

    df_annot = df.merge(overlapping_segments_df[["segment_id", "gene_symbol"]], on = "segment_id", how = "left").drop_duplicates()

    # Remove duplicate segment_ids
    df_annot = df_annot.drop_duplicates(subset='segment_id')

    return df_annot

def pca_plot(df, n_components=2, group_column=None, label_column=None):

    # Perform PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df.select_dtypes(exclude=['object']))

    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # Add the 'Group' column back to the DataFrame
    principalDf["group"] = df[group_column]

    # If label_column is provided, use it as label
    if label_column:
        principalDf["label"] = df[label_column]
    else:
        principalDf["label"] = df[group_column]

    # Create a scatter plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1, Explained variance: {:.2%}'.format(pca.explained_variance_ratio_[0]), fontsize = 15)
    ax.set_ylabel('Principal Component 2, Explained variance: {:.2%}'.format(pca.explained_variance_ratio_[1]), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    groups = principalDf["group"].unique()
    colors = ['r', 'g', 'b', 'y']  # Add more colors if you have more groups

    for group, color in zip(groups, colors):
        indicesToKeep = principalDf["group"] == group
        scatter = ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                   , principalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
        
        # Add labels to the data points
        for i in range(len(principalDf.loc[indicesToKeep, 'principal component 1'])):
            ax.text(principalDf.loc[indicesToKeep, 'principal component 1'].values[i], principalDf.loc[indicesToKeep, 'principal component 2'].values[i], principalDf.loc[indicesToKeep, 'label'].values[i])
    
    # Add legend for colors
    ax.legend(groups)
    ax.grid()

    plt.show()

def tm_pcc(meth_seg, metadata, DEG_full, pcc_file):

    dataframes = {
        'meth_seg': ['segment_id', 'source_directory', 'sample_id_adj', 'positions', 'mod_qual_positions', 'avg_methylation'],
        'metadata': ['source_directory', 'sample_id_adj', 'sample_id_adj_rnaseq'],
        'DEG_full': ['gene_symbol']
    }

    for df_name, columns in dataframes.items():
        for column in columns:
            assert_column(eval(df_name), column, df_name)

    sample_id_adj_rnaseq = metadata[metadata["source_directory"].isin(meth_seg["source_directory"].unique())]["sample_id_adj_rnaseq"]

    # Melt the DataFrame
    DEG_full_longer = DEG_full.melt(id_vars=['gene_symbol'], value_vars=sample_id_adj_rnaseq, var_name='sample_id_adj', value_name='counts')

    tm_df = meth_seg.merge(DEG_full_longer, on = ["gene_symbol", "sample_id_adj"], how = "left").dropna()
    tm_df["log2_counts"] = np.log2(tm_df["counts"] + 1)

    correlation, p_value = pearsonr(tm_df['avg_methylation'], tm_df['log2_counts'])

    print(f"Average methylation - log2counts pearson correlation: {correlation}")
    print(f"p-value: {p_value}")

    correlation, p_value = pearsonr(tm_df['total_methylation'], tm_df['log2_counts'])

    print(f"Total methylation - log2counts pearson correlation: {correlation}")
    print(f"p-value: {p_value}")

    # Calculate Pearson correlation for each gene
    correlations = []
    for gene in tm_df['gene_symbol'].unique():
        df_gene = tm_df[tm_df['gene_symbol'] == gene]
        if df_gene['avg_methylation'].nunique() > 1 and df_gene['log2_counts'].nunique() > 1:
            correlation, p_value = pearsonr(df_gene['avg_methylation'], df_gene['log2_counts'])
        else:
            correlation, p_value = np.nan, np.nan
        correlations.append((gene, correlation, p_value))

    # Convert to DataFrame and sort by correlation
    correlations_df = pd.DataFrame(correlations, columns=['gene_symbol', 'correlation', 'p_value'])

    # Correct p-values for multiple comparisons
    # reject, pvals_corrected, _, _ = multipletests(correlations_df['p_value'], method='fdr_bh')
    # correlations_df['p_value_corrected'] = pvals_corrected
    
    correlations_df = correlations_df.sort_values('p_value', ascending=True).dropna()

    # Save to file
    correlations_df.to_csv(pcc_file, index=False)

    return correlations_df

def plot_correlation_distribution(correlations_df):
    plt.figure(figsize=(10, 5))
    plt.hist(correlations_df['correlation'], bins=30, edgecolor='black')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correlations')
    plt.show()

def process_wgbs_seg_files(folder):
    """
    Function to process .tsv files in a folder and save the result to a .csv file.

    Parameters:
    folder (str): The folder containing the .tsv files.
    output_file (str): The path to the output .csv file.
    """
    # Find all .tsv files in the folder
    file_paths = glob.glob(folder + '/**/*.tsv', recursive=True)

    # Read each file into a DataFrame and append it to the list, skipping empty files
    dfs = []
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path, sep='\t')
            # Drop columns that are empty or all NA
            df = df.dropna(how='all', axis=1)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file_path}")

    # Concatenate all DataFrames in the list
    df_concat = pd.concat(dfs, ignore_index=True)

    # Rename columns with multiple periods to the substring before the first period
    df_concat = df_concat.rename(columns=lambda x: x.split('.')[0] if '.' in x else x)

    df_concat["segment_id"] = df_concat["chr"].astype(str) + ":" + df_concat["start"].astype(str) + "-" + df_concat["end"].astype(str)

    df_concat = df_concat.rename(columns={'chr': 'chrom'})

    return df_concat

def process_wgbs_dmr_files(folder):
    """
    Function to process .bed files in a folder and return a DataFrame.

    Parameters:
    folder (str): The folder containing the .bed files.

    Returns:
    pandas.DataFrame: The resulting DataFrame.
    """
    # Find all .bed files in the folder
    file_paths = glob.glob(folder + '/**/*.bed', recursive=True)

    # Read each file into a DataFrame and append it to the list, skipping empty files
    dfs = []
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path, sep='\t')
            # Drop columns that are empty or all NA
            df = df.dropna(how='all', axis=1)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file_path}")

    # Concatenate all DataFrames in the list
    df_concat = pd.concat(dfs, ignore_index=True)

    df_concat = df_concat.rename(columns={'#chr': 'chrom', 'region': 'segment_id'})

    df_concat["length"] = df_concat["end"] - df_concat["start"]

    return df_concat

def filter_dmr(X_train_df, X_test_df, groups_train, test = 'ttest', p_value_threshold=0.05):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Keep a copy of the column names
        column_names = X_train_df.columns

        # Convert the data to numpy arrays for faster computation
        X_train = X_train_df.values
        X_test = X_test_df.values
        groups_train = groups_train.values

        # Get the indices of the 'R' and 'S' groups
        R_indices = np.where(groups_train == 'R')[0]
        S_indices = np.where(groups_train == 'S')[0]

        # Split the data into the 'R' and 'S' groups
        R_data = X_train[R_indices]
        S_data = X_train[S_indices]

        # Calculate the test statistic and p-value for all columns at once
        if test == 'ttest':
            test_statistic, p_values = ttest_ind(R_data, S_data, axis=0, nan_policy='omit')
        elif test == 'kruskal':
            p_values = np.empty(X_train.shape[1])
            for i in range(X_train.shape[1]):
                _, p_values[i] = kruskal(R_data[:, i], S_data[:, i], nan_policy='omit')
        else:
            raise ValueError("Invalid test. Only 'ttest' and 'kruskal' are supported.")

        # Correct for multiple testing
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

        # Create a DataFrame with the p-values
        dmr_results = pd.DataFrame({
            'segment_id': column_names,
            'q_value': corrected_p_values,
            'p_value': p_values
        })

        # Filter the DataFrame based on the p-value threshold
        filtered_dmr = dmr_results[dmr_results['p_value'] <= p_value_threshold]

        # Keep only the columns in X_test and X_train that are in filtered_dmr
        X_test_filtered = X_test_df.filter(filtered_dmr['segment_id'])
        X_train_filtered = X_train_df.filter(filtered_dmr['segment_id'])

    return X_train_filtered, X_test_filtered, filtered_dmr

def plot_roc_curve(y_test, y_pred_proba):
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def split(X, y, test_size):

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, np.arange(len(X)), test_size=test_size)

    return X_train, X_test, y_train, y_test, train_indices, test_indices

def train_and_predict_single(meth_seg_fm, test_size = 0.25, train_indices=None, test_indices=None, reg = False, dmr = None, oversamp = None):
    # Initialize the encoder and scaler
    encoder = LabelEncoder()
    scaler = StandardScaler()

    X = meth_seg_fm.select_dtypes(exclude=['object'])
    y = meth_seg_fm["Group"]

    if train_indices is None and train_indices is None:
        X_train, X_test, y_train, y_test, train_indices, test_indices = split(X, y, test_size)
    else:
        if test_indices is None and train_indices is not None:
            test_indices = [i for i in range(len(X)) if i not in train_indices]
        elif train_indices is None and test_indices is not None:
            train_indices = [i for i in range(len(X)) if i not in test_indices]
        X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

    groups_train = meth_seg_fm["Group"].iloc[train_indices]
    groups_test = meth_seg_fm["Group"].iloc[test_indices]
    cell_lines_train = meth_seg_fm["sample_id_adj"].iloc[train_indices]
    cell_lines_test = meth_seg_fm["sample_id_adj"].iloc[test_indices]

    X_features_prev = X_train.columns
    X_features_current = X_features_prev

    if dmr:
        X_train, X_test, filtered_dmr = filter_dmr(X_train, X_test, groups_train, dmr)
        X_features_prev = X_features_current
        X_features_current = X_train.columns
        print(f"DMR has removed {len(X_features_prev)-len(X_features_current)} features of the original {len(X_features_prev)}.")

    # Convert the target variable to numeric values
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Standardize the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert back to dataframes
    X_train = pd.DataFrame(X_train, columns=X_features_current)
    X_test = pd.DataFrame(X_test, columns=X_features_current)

    # Fit the logistic regression model
    if reg:
        model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter=10000)
    else:
        model = LogisticRegression()
    
    model.fit(X_train, y_train)

    # Assuming X is your feature matrix and model is your trained model
    X_features_prev = X_features_current
    X_features_current = X_features_current[model.coef_[0] != 0]

    if reg:
        print(f"Regularization has removed {len(X_features_prev)-len(X_features_current)} features of the original {len(X_features_prev)}.")

    # Get the probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Get the predicted classes
    y_pred = model.predict(X_test)

    # Convert the predicted classes back to the original string labels
    y_pred_labels = encoder.inverse_transform(y_pred)

    class_names = model.classes_

    # Print the predicted labels and the true labels for each sample
    for i in range(len(X_test)):
        print("Sample:", i)
        print("Cell line:", cell_lines_test.iloc[i])
        print("True label:", f"{encoder.inverse_transform([y_test[i]])[0]} ({y_test[i]})")
        print("Predicted label:", f"{encoder.inverse_transform([y_pred[i]])[0]} ({y_pred[i]})")
        print("Probabilities:")
        for j in range(len(class_names)):
            print(f"{encoder.inverse_transform([class_names[j]])}: {y_pred_proba[i, j]}")
        print()

    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names = X_test.columns, show=False)

    if dmr or reg:
        # Use loc instead of iloc
        retained_features_fm = X.loc[:, X_features_current]
        retained_features_fm["sample_id_adj"] = meth_seg_fm["sample_id_adj"]
        retained_features_fm["Group"] = meth_seg_fm["Group"]
        print("PCA after filtering:")
        retained_features_fm.reset_index(drop=True, inplace=True)
        pca_plot(retained_features_fm, n_components=2, group_column="Group", label_column = "sample_id_adj")

    # Get the feature importances
    coeff = model.coef_[0]

    # Create a DataFrame with the feature names and their corresponding importances
    coeff_df = pd.DataFrame({'Feature': X_features_prev, 'Coefficients': coeff})

    return model, X_train, X_test, coeff_df, explainer, shap_values

def train_and_predict_loo(meth_seg_fm, reg = False, dmr = None):
    # Initialize the encoder, scaler and classifier
    encoder = LabelEncoder()
    scaler = StandardScaler()

    X = meth_seg_fm.select_dtypes(exclude=['object'])
    y = meth_seg_fm["Group"]
    
    # Fit the logistic regression model
    if reg:
        clf = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter=10000)
    else:
        clf = LogisticRegression()

    loo = LeaveOneOut()
    accuracies = []

    # Initialize the dictionary
    shap_dict = {col: [0, 0] for col in X.columns}

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if dmr:
            groups_train = meth_seg_fm["Group"].iloc[train_index]
            X_train, X_test, filtered_dmr = filter_dmr(X_train, X_test, groups_train, dmr)
        X_features_current = X_train.columns
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Convert back to dataframes
        X_train = pd.DataFrame(X_train, columns=X_features_current)
        X_test = pd.DataFrame(X_test, columns=X_features_current)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        # Create a SHAP explainer and calculate SHAP values
        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_test)

        # Update the dictionary with the new SHAP values
        for i, col in enumerate(X_features_current):
            shap_dict[col][0] += shap_values.values[0][i]  # Update average_shap_value
            if y_test == 1:
                shap_dict[col][1] += shap_values.values[0][i]  # Update reliability_shap_value
            else:
                shap_dict[col][1] -= shap_values.values[0][i]  # Update reliability_shap_value

    # Divide the accumulated SHAP values by the number of iterations to get the average
    for col in shap_dict:
        shap_dict[col][0] /= len(accuracies)
        shap_dict[col][1] /= len(accuracies)

    # Convert the dictionary to a DataFrame
    shap_df = pd.DataFrame.from_dict(shap_dict, orient='index', columns=['average_shap_value', 'reliability_shap_value'])

    # Reset the index to add the feature names as a column
    shap_df.reset_index(inplace=True)

    # Rename the new column
    shap_df.rename(columns={'index': 'segment_id'}, inplace=True)

    # Replace zeros with np.nan
    shap_df = shap_df.replace(0, np.nan)

    # Remove rows with np.nan
    shap_df = shap_df.dropna()

    # Sort the DataFrame by 'reliability_shap_value' in descending order
    shap_df = shap_df.sort_values(by='reliability_shap_value', ascending=False)

    # Calculate the z-scores of the average SHAP values
    shap_df['z_score'] = zscore(shap_df['reliability_shap_value'])

    # Calculate the average methylation across R and S samples for each feature
    avg_meth_R = X[meth_seg_fm["Group"] == "R"].mean()
    avg_meth_S = X[meth_seg_fm["Group"] == "S"].mean()

    # Convert the series to dataframes
    avg_meth_R_df = avg_meth_R.to_frame().reset_index().rename(columns={"index": "segment_id", 0: "avg_meth_R"})
    avg_meth_S_df = avg_meth_S.to_frame().reset_index().rename(columns={"index": "segment_id", 0: "avg_meth_S"})

    # Merge the average methylation dataframes into shap_df
    shap_df = pd.merge(shap_df, avg_meth_R_df, on="segment_id", how="left")
    shap_df = pd.merge(shap_df, avg_meth_S_df, on="segment_id", how="left")

    shap_df["diff"] = shap_df["avg_meth_R"] - shap_df["avg_meth_S"]

    shap_df["direction"] = np.where(shap_df["diff"] > 0, "M", "U")

    print(f"Average accuracy: {np.mean(accuracies)}")

    return shap_df

def plot_pvalue_distribution(pvalues):
    """
    Function to plot the distribution of p-values.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the p-values.
    pvalue_column (str): Name of the column in df that contains the p-values.
    """
    # Create a new figure
    sns.displot(pvalues, kde=False, bins=30)

    # Set the title and labels
    plt.title('P-value Distribution')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()