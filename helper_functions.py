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
from joblib import dump
from adjustText import adjust_text
import matplotlib.patches as mpatches
from itertools import product
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import io

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN

import matplotlib.ticker as mtick

def plot_importance_kde(df, whitelist_df, column = 'z_score'):
    # Plot the KDE of the original DataFrame
    sns.kdeplot(df[column], label='Before Filtering')

    # Create a new DataFrame that only includes the rows where the segment_id is in the whitelist
    df_whitelist = df[df['segment_id'].isin(whitelist_df['segment_id'])]

    # Plot the KDE of the whitelisted DataFrame
    sns.kdeplot(df_whitelist[column], label='After Filtering')

    # Set the labels
    plt.xlabel('Importance Score')
    plt.ylabel('Density')
    plt.title('KDE of Importance Scores')

    # Show the plot
    plt.legend()
    plt.show()

def plot_features_importances_removed_histogram(df, whitelist_df, column = 'z_score'):
    # Calculate the quantiles for the original DataFrame
    df['quantile'], bin_edges = pd.qcut(df[column], q=10, retbins=True, labels=False)

    # Create a new DataFrame that only includes the rows where the segment_id is in the whitelist
    df_whitelist = df[df['segment_id'].isin(whitelist_df['segment_id'])]

    # Calculate the counts for each quantile in the original and whitelisted DataFrames
    total_counts = df['quantile'].value_counts().sort_index()
    whitelist_counts = df_whitelist['quantile'].value_counts().sort_index()

    # Calculate the percentages of removed features
    percentages_removed = 1 - (whitelist_counts / total_counts)

    # Create a bar plot
    plt.bar(range(10), percentages_removed, width = 1, edgecolor="black")

    # Set the labels
    plt.xlabel('Percent Importance Intervals')
    plt.ylabel('Percentage of Features Removed')
    plt.xticks(range(10), [f'{i*10}-{(i+1)*10}%' for i in range(10)], rotation = 45)

    # Format the y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Show the plot
    plt.show()

    # Drop the 'quantile' column
    df.drop(columns=['quantile'], inplace=True)

def plot_venn_top_features(df1, df2, feature_col='segment_id', importance_col='z_score', top_n=100):
    """
    Function to plot a Venn diagram of the top features in two dataframes.

    Parameters:
    df1 (pandas.DataFrame): The first DataFrame.
    df2 (pandas.DataFrame): The second DataFrame.
    feature_col (str): The name of the column containing the features.
    importance_col (str): The name of the column containing the importance scores.
    top_n (int): The number of top features to consider.
    """
    # Get the top features in both dataframes
    top_features_df1 = set(df1.nlargest(top_n, importance_col)[feature_col])
    top_features_df2 = set(df2.nlargest(top_n, importance_col)[feature_col])

    # Plot the Venn diagram
    plt.figure(figsize=(8, 4))
    venn2([top_features_df1, top_features_df2], set_labels = ('All', 'FRGs'))
    plt.title('Venn Diagram of Top Features in both models')
    plt.show()

from sklearn.preprocessing import MaxAbsScaler

def plot_overlap_importance(df1, df2, segment_id_col='segment_id', importance_col='z_score'):
    """
    Function to find the overlap between segments in two dataframes and plot their importance.

    Parameters:
    df1 (pandas.DataFrame): The first DataFrame.
    df2 (pandas.DataFrame): The second DataFrame.
    segment_id_col (str): The name of the column containing the segment IDs.
    importance_col (str): The name of the column containing the importance scores.
    """
    # Find the overlap between segments in the two dataframes
    overlap_df = pd.merge(df1, df2, on=segment_id_col, suffixes=('_df1', '_df2'))

    # Initialize a MaxAbsScaler
    scaler = MaxAbsScaler()

    # Normalize the importance scores in both dataframes
    overlap_df[[importance_col + '_df1', importance_col + '_df2']] = scaler.fit_transform(overlap_df[[importance_col + '_df1', importance_col + '_df2']])

    # Plot the importance of overlapping segments in both datasets
    plt.figure(figsize=(8, 4))
    plt.scatter(overlap_df[importance_col + '_df1'], overlap_df[importance_col + '_df2'], alpha=0.6)
    plt.xlabel('Scaled Importance in All')
    plt.ylabel('Scaled Importance in FRGs')
    plt.title('Scaled Importance of Overlapping Segments in both models')
    plt.show()

def plot_importance_and_regions(df, importance_col = "z_score"):
    df["chrom"] = df["segment_id"].str.split(":").str[0]

    # Calculate the average importance and count of regions for each chromosome
    grouped = df.groupby('chrom')[importance_col].agg(['mean', 'count'])

    # Create a list of all possible chromosomes
    all_chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrMT']

    # Reindex the grouped DataFrame to include all chromosomes
    grouped = grouped.reindex(all_chromosomes)

    # Sort the index of the grouped dataframe
    grouped.sort_index(key=lambda x: pd.to_numeric(x.str.extract('(\d+)', expand=False), errors='coerce'), inplace=True)

    # Create a bar plot
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Define the width of the bars and the positions of the x-coordinates
    width = 0.35
    x = np.arange(len(grouped.index))

    # Plot the average importance
    ax1.bar(x - width/2, grouped['mean'], width, label='Average Importance', alpha=0.6, color='b')

    # Create a second y-axis for the count of regions
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, grouped['count'], width, label='Number of Regions', alpha=0.6, color='r')

    # Draw lines through the x ticks
    for i in x:
        ax1.axvline(i, color='gray', linestyle='--', linewidth=0.5)

    # Set the labels and title
    ax1.set_xlabel('Chromosome')
    ax1.set_ylabel('Average Importance')
    ax2.set_ylabel('Number of Regions')
    plt.title('Average Importance and Number of Regions for Each Chromosome')

    # Rotate the x-labels and set them manually
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped.index, rotation=45, ha='right')

    # Get the limits of the first y-axis
    y1_min, y1_max = ax1.get_ylim()

    # Calculate the corresponding limits for the second y-axis
    y2_min = y1_min * grouped['count'].max() / grouped['mean'].max()
    y2_max = y1_max * grouped['count'].max() / grouped['mean'].max()

    # Set the limits of the second y-axis
    ax2.set_ylim(y2_min, y2_max)

    # Add a legend
    fig.legend(loc="upper right")

    plt.show()

def filter_seg_annot(wgbs_seg_annot, wgbs_segcov, whitelist):
    wgbs_seg_annot["segment_id"] = wgbs_seg_annot["seqnames"] + ":" + wgbs_seg_annot["start"].astype(str) + "-" + wgbs_seg_annot["end"].astype(str)
    wgbs_seg_annot = wgbs_seg_annot.merge(wgbs_segcov[["segment_id", "avg_depth"]], on = "segment_id", how = "left")

    frg_genes = whitelist['gene_symbol'].unique()
    unique_segment_ids = wgbs_seg_annot['segment_id'].unique()
    frg_associated_segment_ids = wgbs_seg_annot[wgbs_seg_annot['annotated_genes'].isin(frg_genes)]['segment_id'].unique()
    percentage = (len(frg_associated_segment_ids) / len(unique_segment_ids)) * 100
    print(f"The percentage of unique segment_ids associated with at least one frg is {percentage:.2f}%")

    wgbs_seg_annot_filt = wgbs_seg_annot[
        (wgbs_seg_annot["annotated_genes"].isin(whitelist["gene_symbol"])) &
        (wgbs_seg_annot["avg_depth"] > 10) &
        (wgbs_seg_annot["avg_depth"] < 40)
    ]
    plot_chromosome_distribution(wgbs_seg_annot_filt)

    plot_depth_distribution(wgbs_seg_annot_filt)

    return wgbs_seg_annot_filt

def plot_depth_distribution(df):
    # Plot avg_depth distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df['avg_depth'], bins=30, alpha=0.5, color='g')
    plt.title('avg_depth Distribution')
    plt.xlabel('avg_depth')
    plt.ylabel('Frequency')
    
    # Calculate average
    avg_depth = df['avg_depth'].mean()
    
    # Add average as text
    plt.text(0.7, 0.9, 'Average: {:.2f}'.format(avg_depth), transform=plt.gca().transAxes)
    
    # Set y-axis to log scale if necessary
    if df['avg_depth'].max() / df['avg_depth'].min() > 1000:
        plt.yscale('log')
    
    plt.show()

def filter_fm_segments(df, whitelist):
    """
    This function filters numeric columns based on a whitelist and then appends non-numeric columns back.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to filter.
    whitelist (pandas.DataFrame): The whitelist DataFrame. It should have a 'segment_id' column.
    
    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    # Get the intersection of the numeric columns and the whitelist
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filtered_numeric_cols = numeric_cols.intersection(whitelist['segment_id'])

    # Get the non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Combine the filtered numeric columns and the non-numeric columns
    filtered_cols = filtered_numeric_cols.union(non_numeric_cols)

    return df[filtered_cols]

def read_shap_annot(file, whitelist):
    shap_annot_df = pd.read_csv(file, index_col = 0)
    shap_annot_df.drop("gene_symbol", axis=1, inplace=True)
    shap_annot_df.rename(columns={"annotated_genes": "gene_symbol"}, inplace=True)
    shap_annot_df['segment_id'] = shap_annot_df['seqnames'].astype(str) + ':' + shap_annot_df['start'].astype(str) + '-' + shap_annot_df['end'].astype(str)
    shap_annot_df_whitelist = shap_annot_df[shap_annot_df["gene_symbol"].isin(whitelist["gene_symbol"])]
    return shap_annot_df, shap_annot_df_whitelist

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
    
    # Sort by 'sample_id_adj' and 'source_directory'
    processed_df = processed_df.sort_values(by=['tumor_type', 'sample_id_adj'])

    # Reset the index and drop the old index
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df

def find_annot_overlap(df, annot, gene_column = "gene_symbol"):

    dataframes = {
        'df': ['segment_id'],
        'annot': ['segment_id', gene_column]
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
    overlapping_segments_df = overlapping_segments_df.merge(annot[["segment_id", gene_column]], on='segment_id', how='left').drop_duplicates()

    # Add a column for the length of the overlap
    overlapping_segments_df['overlap_length'] = overlapping_segments_df['End_b'] - overlapping_segments_df['Start_b']

    # Group by segment_id and keep only the row with the largest overlap for each segment_id
    overlapping_segments_df = overlapping_segments_df.loc[overlapping_segments_df.groupby('segment_id')['overlap_length'].idxmax()]

    overlapping_segments_df['segment_id'] = overlapping_segments_df['Chromosome'].astype(str) + ':' + overlapping_segments_df['Start'].astype(str) + '-' + overlapping_segments_df['End'].astype(str)

    df_annot = df.merge(overlapping_segments_df[["segment_id", gene_column]], on = "segment_id", how = "left").drop_duplicates()

    # Remove duplicate segment_ids
    df_annot = df_annot.drop_duplicates(subset='segment_id')

    return df_annot

def pca_plot(df, n_components=2, color_column=None, marker_column=None, set_column = None, label_column=None, colors = None, markers = None):

    # Standardize the features to have mean=0 and variance=1
    features = df.select_dtypes(exclude=['object'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(scaled_features)

    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # If color_column is provided, use it as label
    if color_column:
        principalDf["color_group"] = df[color_column]
        color_groups = np.sort(principalDf["color_group"].unique())
        color_dict = {group: color for group, color in zip(color_groups, colors)}

    if marker_column:
        principalDf["marker_group"] = df[marker_column]
        marker_groups = np.sort(principalDf["marker_group"].unique())
        marker_dict = {group: markers[i % len(markers)] for i, group in enumerate(marker_groups)}

    # If label_column is provided, use it as label
    if label_column:
        principalDf["label"] = df[label_column]
    else:
        principalDf["label"] = principalDf["color"]

    # If set_column is provided, append it to the label
    if set_column:
        principalDf["label"] = principalDf["label"] + " (" + df[set_column] + ")"

    # Create a scatter plot
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1, Explained variance: {:.2%}'.format(pca.explained_variance_ratio_[0]), fontsize = 15)
    ax.set_ylabel('PC2, Explained variance: {:.2%}'.format(pca.explained_variance_ratio_[1]), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    texts = []  # list to store the labels
    for color_group, marker_group in product(color_groups, marker_groups):
        indicesToKeep = (principalDf["color_group"] == color_group) & (principalDf["marker_group"] == marker_group)
        scatter = ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                , principalDf.loc[indicesToKeep, 'principal component 2']
                , c = color_dict[color_group]
                , s = 50
                , marker = marker_dict[marker_group])
        
        # Add labels to the data points
        for i in range(len(principalDf.loc[indicesToKeep, 'principal component 1'])):
            x = principalDf.loc[indicesToKeep, 'principal component 1'].values[i]
            y = principalDf.loc[indicesToKeep, 'principal component 2'].values[i]
            label = principalDf.loc[indicesToKeep, 'label'].values[i]
            texts.append(ax.text(x, y, label, fontsize=12))  # Increase the font size here

    # Adjust the labels to avoid overlap
    adjust_text(texts, 
            only_move={'points':'', 'texts':'xy', 'objects':'xy'}, 
            arrowprops=dict(arrowstyle="->", color='gray'))

    # Create lists to store the legend elements for colors and markers
    color_legend_elements = []
    marker_legend_elements = []

    # Add a legend entry for each color group
    for color_group in color_groups:
        color_legend_elements.append(Line2D([0], [0], marker='o', color='w', label=color_group,
                                            markerfacecolor=color_dict[color_group], markersize=10))

    # Add a legend entry for each marker group
    for marker_group in marker_groups:
        marker_legend_elements.append(Line2D([0], [0], marker=marker_dict[marker_group], color='w', label=marker_group,
                                            markerfacecolor='black', markersize=10))

    # Create the legends
    color_legend = plt.legend(handles=color_legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title='Color Groups')
    plt.gca().add_artist(color_legend)
    plt.legend(handles=marker_legend_elements, loc='upper left', bbox_to_anchor=(1, 0.85), title='Marker Groups')
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

def plot_correlation_distribution(correlations_df, title='Distribution of Correlations'):
    plt.figure(figsize=(10, 5))
    plt.hist(correlations_df['correlation'], bins=30, edgecolor='black')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title(title)

    # Calculate and display the number of correlations above and below 0
    above_zero = sum(correlations_df['correlation'] > 0)
    below_zero = sum(correlations_df['correlation'] < 0)
    plt.text(0, 1.15, f'Negative correlations: {above_zero}', transform=plt.gca().transAxes)
    plt.text(0, 1.1, f'Positive correlations: {below_zero}', transform=plt.gca().transAxes)

    plt.show()

import natsort

def plot_chromosome_distribution(df):
    """
    Function to plot the distribution of regions across chromosomes.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the chromosome data.
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()

    df_copy["chrom"] = df_copy["segment_id"].str.split(":").str[0]

    # Calculate value counts
    chrom_counts = df_copy['chrom'].value_counts()

    # Use natsort to naturally sort the chromosome names
    chrom_counts = chrom_counts.reindex(index=natsort.natsorted(chrom_counts.index))

    # Plot the distribution of regions across chromosomes
    chrom_counts.plot(kind='bar', color='b', alpha=0.6)
    plt.xlabel('Chromosome')
    plt.ylabel('Number of Regions')
    plt.title('Distribution of Regions Across Chromosomes')
    plt.show()

def process_wgbs_seg_files(seg_folder, cov_folder = None, convert_to_m_values=False):
    """
    Function to process .tsv files in a folder and save the result to a .csv file.

    Parameters:
    folder (str): The folder containing the .tsv files.
    output_file (str): The path to the output .csv file.
    """
    # Find all .tsv files in the folder
    file_paths = glob.glob(seg_folder + '/**/*.tsv', recursive=True)

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
    df_concat = df_concat.rename(columns=lambda x: x.replace('.sorted', ''))

    df_concat["segment_id"] = df_concat["chr"].astype(str) + ":" + df_concat["start"].astype(str) + "-" + df_concat["end"].astype(str)

    df_concat = df_concat.rename(columns={'chr': 'chrom'})

    plot_chromosome_distribution(df_concat)

    df_concat = df_concat.drop(columns=['chrom', 'start', 'end', 'startCpG', 'endCpG'], errors='ignore')

    df_concat.dropna(inplace=True)

    cols = ['segment_id'] + [col for col in df_concat if col != 'segment_id']
    df_concat = df_concat[cols]

    cov_df = pd.DataFrame(df_concat["segment_id"].copy())

    if convert_to_m_values:
        # Identify numeric columns
        beta_cols = df_concat.select_dtypes(include=[np.number]).columns.tolist()

        # Convert beta values to M-values
        for col in beta_cols:
            epsilon = 0.01
            df_concat[col] = np.log2((df_concat[col]+epsilon) / (1 - df_concat[col]+epsilon))

    if cov_folder:
        # Get the list of sample folders
        sample_folders = [f.path for f in os.scandir(cov_folder) if f.is_dir()]

        for sample_folder in sample_folders:
            # Get the list of bed files in the sample folder
            bed_files = glob.glob(sample_folder + '/**/*.bed.gz', recursive=True)

            bed_dfs = []
            for bed_file in bed_files:
                df = pd.read_csv(bed_file, sep='\t', compression='gzip', header=None, usecols=[0, 1, 2, 4])
                df['segment_id'] = df[0].astype(str) + ":" + df[1].astype(str) + "-" + df[2].astype(str)
                sample_name = os.path.basename(sample_folder)
                df = df.rename(columns={4: f'{sample_name}_depth'})
                df = df[['segment_id', f'{sample_name}_depth']]
                bed_dfs.append(df)

            # Concatenate all bed DataFrames vertically
            bed_df_concat = pd.concat(bed_dfs, ignore_index=True)

            cov_df = cov_df.merge(bed_df_concat, on='segment_id', how='left')

        cov_df["avg_depth"] = cov_df.iloc[:, 1:].mean(axis=1)

        plot_depth_distribution(cov_df)

        return df_concat, cov_df

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

def filter_dmr(X_train_df, X_test_df, groups_train, test = 'ttest', p_value_threshold=0.05, diff_threshold=0.1):
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
    p_values = np.empty(X_train.shape[1])
    if test == 'ttest':
        test_statistic, p_values = ttest_ind(R_data, S_data, axis=0, nan_policy='omit')
        # for i in range(X_train.shape[1]):
        #     if len(np.unique(R_data[:, i])) > 1 or len(np.unique(S_data[:, i])) > 1:
        #         _, p_values[i] = ttest_ind(R_data[:, i], S_data[:, i], nan_policy='omit')
        #     else:
        #         p_values[i] = np.nan
    elif test == 'kruskal':
        test_statistic, p_values = kruskal(R_data, S_data, axis=0, nan_policy='omit')
        # for i in range(X_train.shape[1]):
        #     if len(np.unique(R_data[:, i])) > 1 or len(np.unique(S_data[:, i])) > 1:
        #         _, p_values[i] = kruskal(R_data[:, i], S_data[:, i], nan_policy='omit')
        #     else:
        #         p_values[i] = np.nan
    else:
        raise ValueError("Invalid test. Only 'ttest' and 'kruskal' are supported.")

    # Correct for multiple testing
    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

    # Calculate the mean difference
    mean_diff = np.abs(R_data.mean(axis=0) - S_data.mean(axis=0))

    # Create a DataFrame with the p-values and mean differences
    dmr_results = pd.DataFrame({
        'segment_id': column_names,
        'q_value': corrected_p_values,
        'p_value': p_values,
        'mean_diff': mean_diff
    }).sort_values(by = "p_value")

    # Filter the DataFrame based on the p-value threshold and mean difference
    filtered_dmr = dmr_results[(dmr_results['p_value'] <= p_value_threshold) & (dmr_results['mean_diff'] >= diff_threshold)]

    # Keep only the columns in X_test and X_train that are in filtered_dmr
    X_test_filtered = X_test_df.filter(filtered_dmr['segment_id'])
    X_train_filtered = X_train_df.filter(filtered_dmr['segment_id'])

    return X_train_filtered, X_test_filtered, filtered_dmr, dmr_results

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

def train_and_predict_single(meth_seg_fm, 
                             test_size = 0.25, 
                             train_indices=None, 
                             test_indices=None, 
                             reg = False, 
                             dmr = None, 
                             perform_pca = False, 
                             n_components = 2, 
                             model_str = "lr", 
                             p_value_threshold = 0.05,
                             diff_threshold = 0.1):
    # Initialize the encoder and scaler
    encoder = LabelEncoder()
    scaler = StandardScaler()

    if model_str in ["lr"]:
        model_type = "linear"
    elif model_str in ["rf", "dt"]:
        model_type = "tree"

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
        X_train, X_test, filtered_dmr, dmr_results = filter_dmr(X_train, X_test, groups_train, dmr, p_value_threshold=p_value_threshold, diff_threshold=diff_threshold)
        X_features_prev = X_features_current
        X_features_current = X_train.columns
        print(f"DMR has removed {len(X_features_prev)-len(X_features_current)} features of the original {len(X_features_prev)}.")

        # Create a 1x2 grid for the plots
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Plot a histogram of the p-values
        n, bins, patches = axs[0].hist(dmr_results['p_value'], bins=30)
        for i in range(len(patches)):
            if (bins[i] <= p_value_threshold):
                patches[i].set_facecolor('red')
        axs[0].set_title('Distribution of P-values')
        axs[0].set_xlabel('P-value')
        axs[0].set_ylabel('Frequency')

        # Plot a histogram of the mean differences
        n, bins, patches = axs[1].hist(dmr_results['mean_diff'], bins=30)
        for i in range(len(patches)):
            if (bins[i] <= diff_threshold):
                patches[i].set_facecolor('red')
        axs[1].set_title('Distribution of Mean Differences')
        axs[1].set_xlabel('Mean Difference')
        axs[1].set_ylabel('Frequency')

        # Display the plots
        plt.tight_layout()
        plt.show()

    # Convert the target variable to numeric values
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Standardize the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform PCA if specified
    if perform_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_features_current = [f'PC{i+1}' for i in range(n_components)]

    # Convert back to dataframes
    X_train = pd.DataFrame(X_train, columns=X_features_current)
    X_test = pd.DataFrame(X_test, columns=X_features_current)

    # Fit the logistic regression model
    if model_str == "rf":
        model = RandomForestClassifier()
    elif model_str == "dt":
        model = DecisionTreeClassifier()
    elif model_str == "lr":
        if reg:
            model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter=10000)
        else:
            model = LogisticRegression()
    
    model.fit(X_train, y_train)

    if reg and model_type == "linear":
        # Assuming X is your feature matrix and model is your trained model
        X_features_prev = X_features_current
        X_features_current = X_features_current[model.coef_[0] != 0]
        print(f"Regularization has removed {len(X_features_prev)-len(X_features_current)} features of the original {len(X_features_prev)}.")

    # Get the probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Get the predicted classes
    y_pred = model.predict(X_test)

    # Convert the predicted classes back to the original string labels
    y_pred_labels = encoder.inverse_transform(y_pred)

    class_names = model.classes_

    print(f"Number of features remaining: {len(X_features_current)}")

    # Create a DataFrame with the prediction probabilities for the positive class, predicted labels, and true labels
    pred_df = pd.DataFrame({
        'Prediction Probability': y_pred_proba[:, 1],
        'Predicted Label': y_pred,
        'True Label': y_test
    })

    accuracies = (y_pred == y_test).astype(int)

    pred_df['True Label'] = pred_df['True Label'].apply(lambda x: str(x))

    # Convert accuracies to a pandas Series
    accuracies_series = pd.Series(accuracies)

    # Map 1 to 'Correct' and 0 to 'Incorrect'
    pred_df['Correct Prediction'] = accuracies_series.map({1: 'Correct', 0: 'Incorrect'})
    pred_df['Sample'] = meth_seg_fm.loc[test_indices, "sample_id_adj"].values

    labels_dict = {0: encoder.inverse_transform([0])[0], 1: encoder.inverse_transform([1])[0]}

    # Plot the prediction probability distribution
    plot_prediction_probability(pred_df, labels_dict)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    if dmr or reg:
        # Use loc instead of iloc
        retained_features_fm = X.loc[:, X_features_current]
        retained_features_fm["sample_id_adj"] = meth_seg_fm["sample_id_adj"]
        retained_features_fm["Group"] = meth_seg_fm["Group"]
        retained_features_fm["tumor_type"] = meth_seg_fm["tumor_type"]
        retained_features_fm["set"] = "test"  # Initially set all rows to "test"
        retained_features_fm.loc[retained_features_fm.index.isin(train_indices), "set"] = "train"  # Set rows in train_indices to "train"
        retained_features_fm.reset_index(drop=True, inplace=True)
        unique_tumor_types = retained_features_fm["tumor_type"].unique()

        if np.array_equal(unique_tumor_types, ["NB"]):
            markers = ["o"]
        elif np.array_equal(unique_tumor_types, ["MM"]):
            markers = ["^"]
        elif all(elem in unique_tumor_types for elem in ["NB", "MM"]):
            markers = ["^", "o"]
        pca_plot(retained_features_fm, color_column = "Group", marker_column = "tumor_type", set_column = "set", label_column = "sample_id_adj", colors = ["r", "g"], markers = markers)

    # Get the feature importances
    if model_type == "tree":
        importances = model.feature_importances_
    elif model_type == "linear":
        importances = model.coef_[0]
    
    if perform_pca:
        importances_df = pd.DataFrame({'Feature': X_features_current, 'Importances': importances})
    else:
        # Create a DataFrame with the feature names and their corresponding importances
        importances_df = pd.DataFrame({'Feature': X_features_prev, 'Importances': importances})

    # Filter the coefficients to remove zero values
    non_zero_coeff = importances[importances != 0]

    # Plot a histogram of the non-zero coefficients
    # Adjust the figure size
    plt.figure(figsize=(5, 5))
    plt.hist(non_zero_coeff, bins=30)
    plt.title('Distribution of Non-Zero Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.show()

    # Create a SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names = X_test.columns, show=False, plot_size = [7, 4])

    return model, X_train, X_test, importances_df, explainer, shap_values

def plot_prediction_probability(pred_df, labels_dict = None):
    """
    Function to create a scatter plot with samples on the x-axis and probabilities on the y-axis.
    Points are colored based on whether the prediction was correct or not.

    Parameters:
    pred_df (pandas.DataFrame): DataFrame containing the prediction probabilities and whether the prediction was correct.

    Returns:
    None
    """

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Define a color dictionary for the hue parameter
    color_dict = {'Correct': 'blue', 'Incorrect': 'orange'}

    # Create a scatter plot with samples on the x-axis and probabilities on the y-axis
    # Color the points based on whether the prediction was correct or not
    scatter = sns.scatterplot(data=pred_df, x='Sample', y='Prediction Probability', hue='Correct Prediction', palette=color_dict, ax=ax, legend = False)

    # Add a horizontal dashed line at y=0.5
    ax.axhline(0.5, color='red', linestyle='--')

    # Set the y-axis limits
    ax.set_ylim(0, 1)

    # Set the title and labels
    ax.set_title('Prediction Probability Distribution')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Prediction Probability')

    # Color the background of the plot based on whether they are above or below 0
    ax.axhspan(0.5, 1, facecolor='green', alpha=0.1)
    ax.axhspan(0, 0.5, facecolor='red', alpha=0.1)

    # Get the handles and labels from seaborn
    handles, labels = scatter.get_legend_handles_labels()

    # Create custom patches for labels if labels_dict is provided
    if labels_dict:
        red_patch = mpatches.Patch(color='red', alpha=0.1, label=labels_dict[0])
        green_patch = mpatches.Patch(color='green', alpha=0.1, label=labels_dict[1])
        handles.extend([red_patch, green_patch])
        labels.extend([labels_dict[0], labels_dict[1]])

    # Create custom Line2D objects for the legend
    blue_dot = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    orange_dot = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)

    # Add the Line2D objects to the handles list
    handles.extend([blue_dot, orange_dot])

    # Add the labels to the labels list
    labels.extend(['Correct', 'Incorrect'])

    # Create the legend
    plt.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1,1))

    # Rotate the feature names and give them more space
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()

def train_and_predict_loo(meth_seg_fm, reg = False, dmr = None, diff_threshold = 0.1):
    # Initialize the encoder, scaler and classifier
    encoder = LabelEncoder()
    scaler = StandardScaler()

    shap_values_list = []

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
    shap_dict = {col: [0, 0, []] for col in X.columns}

    # Initialize the dictionary to store coefficients
    coef_dict = {col: [] for col in X.columns}

    y_true_list = []
    y_pred_list = []
    # Initialize lists to store prediction probabilities
    y_pred_proba_list = []

    fold = 1

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if dmr:
            groups_train = meth_seg_fm["Group"].iloc[train_index]
            X_train, X_test, filtered_dmr, dmr_results = filter_dmr(X_train, X_test, groups_train, dmr, diff_threshold = 0.1)
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
        # Store the prediction probabilities
        y_pred_proba = clf.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        # Create a SHAP explainer and calculate SHAP values
        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_test)

        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        y_pred_proba_list.append(y_pred_proba[0][1])

        # Store the coefficients
        for i, col in enumerate(X_features_current):
            coef_dict[col].append(clf.coef_[0][i])

        # Count the number of non-zero coefficients
        non_zero_coef_count = np.count_nonzero(clf.coef_)

        print(f'Fold {fold}: Number of non-zero coefficients: {non_zero_coef_count}')
        fold += 1

        # Update the dictionary with the new SHAP values
        for i, col in enumerate(X_features_current):
            shap_dict[col][0] += shap_values.values[0][i]  # Update average_shap_value
            if y_test == 1:
                shap_dict[col][1] += shap_values.values[0][i]  # Update reliability_shap_value
            else:
                shap_dict[col][1] -= shap_values.values[0][i]  # Update reliability_shap_value
            shap_dict[col][2].append(shap_values.values[0][i])  # Append the SHAP value to the list

    # Divide the accumulated SHAP values by the number of iterations to get the average
    for col in shap_dict:
        shap_dict[col][0] /= len(accuracies)
        shap_dict[col][1] /= len(accuracies)
        shap_dict[col][2] = np.var(shap_dict[col][2])

    labels_dict = {0: encoder.inverse_transform([0])[0], 1: encoder.inverse_transform([1])[0]}

    # Convert the dictionary to a DataFrame
    shap_df = pd.DataFrame.from_dict(shap_dict, orient='index', columns=['average_shap_value', 'reliability_shap_value', 'shap_variance'])
    
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

    # Create a DataFrame from the coefficient dictionary
    coef_df_ori = pd.DataFrame.from_dict(coef_dict, orient='index').dropna(how = 'all')
    coef_df = coef_df_ori.copy()
    coef_df - coef_df

    # Calculate the absolute mean of the coefficients for each feature
    coef_df['abs_mean'] = coef_df_ori.abs().mean(axis=1)

    # Calculate the mean of the coefficients for each feature
    coef_df['mean_coef'] = coef_df_ori.mean(axis=1)

    # Reset the index to add the feature names as a column
    coef_df.reset_index(inplace=True)

    # Rename the new column
    coef_df.rename(columns={'index': 'segment_id'}, inplace=True)

    # Merge the average coefficient dataframe into shap_df
    shap_df = pd.merge(shap_df, coef_df[['segment_id', 'mean_coef']], on="segment_id", how="left")

    # Get the top 10 features
    top_10_features = coef_df_ori.abs().mean(axis = 1).nlargest(10).index

    # Create a new DataFrame with only the top 10 features
    top_10_coef_df = coef_df_ori.loc[top_10_features].T

    # Create a figure and axis for the boxplot
    plt.figure(figsize=(6, 4))

    # Create a list of colors based on the mean of each feature
    colors = ['green' if mean > 0 else 'red' for mean in top_10_coef_df.mean()]

    # Create boxplots for the top 10 features
    sns.boxplot(data=top_10_coef_df, palette=colors)

    plt.axhline(0, color='black', linestyle='--')

    # Set the title and labels
    plt.title('Coefficient Boxplots for Top 10 Features')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Values')

    # Create custom patches for the legend
    red_patch = mpatches.Patch(color='red', label=labels_dict[0])
    green_patch = mpatches.Patch(color='green', label=labels_dict[1])

    # Add the legend
    plt.legend(handles=[red_patch, green_patch], loc="upper left", bbox_to_anchor=(1,1))

    # Rotate the feature names and give them more space
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()

    # Create a DataFrame with the prediction probabilities, predicted labels, and true labels
    pred_df = pd.DataFrame({
        'Prediction Probability': y_pred_proba_list,
        'Predicted Label': y_pred_list,
        'True Label': y_true_list
    })

    pred_df['True Label'] = pred_df['True Label'].apply(lambda x: str(x))

    # Convert accuracies to a pandas Series
    accuracies_series = pd.Series(accuracies)

    # Map 1 to 'Correct' and 0 to 'Incorrect'
    pred_df['Correct Prediction'] = accuracies_series.map({1: 'Correct', 0: 'Incorrect'})
    pred_df['Sample'] = meth_seg_fm["sample_id_adj"].values

    plot_prediction_probability(pred_df, labels_dict)

    # Add a plot for the distribution of 'z_score'
    plt.figure(figsize=(6, 4))
    plt.hist(shap_df['z_score'], bins=30, alpha=0.5, color='b')
    plt.title('z_score Distribution')
    plt.xlabel('z_score')
    plt.ylabel('Frequency')
    plt.show()

    return shap_df

def plot_pvalue_distribution(pvalues, title='P-value Distribution'):
    """
    Function to plot the distribution of p-values.

    Parameters:
    pvalues (pandas.Series): Series containing the p-values.
    title (str): Title for the plot.
    """
    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Plot the histogram, capture the output
    counts, bins, patches = plt.hist(pvalues, bins=30, edgecolor='black')

    # Color bars under or equal to 0.05
    for count, bin, patch in zip(counts, bins, patches):
        if bin <= 0.05:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('gray')

    # Set the title and labels
    plt.title(title)
    plt.xlabel('P-value')
    plt.ylabel('Frequency')

    # Calculate and display the percentage of p-values under or equal to 0.05
    under_05 = sum(pvalues <= 0.05)
    total = len(pvalues)
    percentage = (under_05 / total) * 100
    plt.text(0, 1.1, f'Percentage p-values < 0.05: {percentage:.2f}%', transform=plt.gca().transAxes)

    # Show the plot
    plt.show()

def save_model(model, train_indices, test_indices, name, base_path='/data/lvisser/models/model_'):
    # Convert indices to strings
    train_indices_str = '_'.join(map(str, train_indices))
    test_indices_str = '_'.join(map(str, test_indices))

    # Create the filename
    filename = f'{base_path}{name}_train_{train_indices_str}_test_{test_indices_str}.joblib'

    # Save the model to a file
    dump(model, filename)