## contains the functions for intercoder reliability

import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations

def compute_coincidence_matrix(Q_choices, Q_df):
    """
    Compute the coincidence matrix for a set of annotations.

    This function calculates a coincidence matrix that represents the frequency with 
    which pairs of choices co-occur across annotations made by different coders for 
    the same transcript. The matrix is normalized by the number of annotators.

    Parameters:
    -----------
    Q_choices : list
        A list of all possible annotation choices.
    Q_df : pandas.DataFrame
        A DataFrame where rows correspond to coders, columns correspond to transcripts, 
        and each cell contains the choice made by a coder for a given transcript. 
        Cells with `NaN` represent missing annotations.

    Returns:
    --------
    numpy.ndarray
        A 2D array (coincidence matrix) where the entry at position (i, j) indicates 
        the normalized co-occurrence frequency of choice `i` and choice `j` across annotations.
    """
    # Initialize the coincidence matrix with zeros
    coincidence_matrix = np.zeros([len(Q_choices), len(Q_choices)])

    # Iterate over all transcripts (columns in the DataFrame)
    for transcript in Q_df.columns:
        # Filter coders who have made annotations for this transcript
        filtered_annotators = Q_df[Q_df[transcript].notna()].index.tolist()
        num_annotators = len(filtered_annotators)

        # Compare pairs of coders to compute co-occurrences
        for coder1 in filtered_annotators:
            for coder2 in filtered_annotators:
                if coder1 != coder2:  # Avoid self-comparisons
                    # Get the indices of the choices for the two coders
                    Qindex1 = Q_choices.index(Q_df.loc[coder1, transcript])
                    Qindex2 = Q_choices.index(Q_df.loc[coder2, transcript])

                    # Increment the coincidence matrix entry for the pair of choices
                    # Normalize by the number of annotators - 1
                    coincidence_matrix[Qindex1, Qindex2] += 1 / (num_annotators - 1)

    # Return the completed coincidence matrix
    return coincidence_matrix

def bootstrap_transcripts_and_compute_coincidence_matrix(Q_choices, Q_df):
    """
    Compute a coincidence matrix using bootstrapped transcripts and annotator pairs.

    This function performs bootstrapping over transcripts to sample transcripts with replacement.
    It calculates the coincidence matrix by comparing annotation choices from bootstrapped annotator
    pairs for each transcript.

    Parameters:
    -----------
    Q_choices : list
        A list of all possible annotation choices.
    Q_df : pandas.DataFrame
        A DataFrame where rows correspond to coders, columns correspond to transcripts,
        and each cell contains the choice made by a coder for a given transcript. 
        Cells with `NaN` represent missing annotations.

    Returns:
    --------
    numpy.ndarray
        A 2D array (coincidence matrix) where the entry at position (i, j) represents
        the average normalized co-occurrence frequency of choice `i` and choice `j`
        across bootstrapped samples.
    """
    # Initialize an empty coincidence matrix
    coincidence_matrix = np.zeros([len(Q_choices), len(Q_choices)])

    # List all transcript columns in the DataFrame
    transcripts = Q_df.columns.tolist()

    # Bootstrap transcripts: sample with replacement
    bootstrapped_transcripts = np.random.choice(transcripts, size=len(transcripts), replace=True)
    
    # Count occurrences of each transcript in the bootstrap sample
    col_count = {col: 0 for col in transcripts}
    for col in bootstrapped_transcripts:
        col_count[col] += 1

    # Create a bootstrapped DataFrame with sampled transcripts
    bootstrap_df = Q_df[bootstrapped_transcripts].copy()
    bootstrap_df.columns = range(len(bootstrap_df.columns))  # Rename columns for consistency

    # Iterate over each bootstrapped transcript
    for transcript in bootstrap_df.columns:
        # Only consider rows (annotators) with non-NaN values for the transcript
        filtered_annotators = bootstrap_df[bootstrap_df[transcript].notna()].index.tolist()
        num_annotators = len(filtered_annotators)

        # If there are fewer than 2 annotators, skip this transcript
        if num_annotators < 2:
            continue

        # Bootstrap: Sample annotator pairs with replacement
        bootstrapped_pairs = np.random.choice(filtered_annotators, size=(num_annotators, 2), replace=True)

        for coder1, coder2 in bootstrapped_pairs:
            if coder1 != coder2:  # Avoid self-comparisons
                # Get the indices of the choices for the two coders
                Qindex1 = Q_choices.index(bootstrap_df.loc[coder1, transcript])
                Qindex2 = Q_choices.index(bootstrap_df.loc[coder2, transcript])
                
                # Update the coincidence matrix with normalized co-occurrence frequency
                coincidence_matrix[Qindex1, Qindex2] += 1 / (num_annotators - 1)

    # Normalize the coincidence matrix by the total number of bootstrap samples
    coincidence_matrix /= len(transcripts)

    # Return the averaged coincidence matrix
    return coincidence_matrix

def alpha(coincidence_matrix):
    """
    Calculate Krippendorff's alpha for a given coincidence matrix.

    Krippendorff's alpha is a statistical measure of inter-coder reliability,
    which quantifies the agreement between coders based on their annotations.
    This implementation uses the coincidence matrix as input to compute the metric.

    Parameters:
    -----------
    coincidence_matrix : numpy.ndarray
        A square 2D array where the entry at (i, j) represents the number of times
        choice `i` and choice `j` co-occur across all annotations.

    Returns:
    --------
    float
        The Krippendorff's alpha value, ranging from -1 (perfect disagreement)
        to 1 (perfect agreement), with 0 indicating no agreement beyond chance.
    """
    # Compute the sum of co-occurrences for each row (marginal totals)
    nc = np.sum(coincidence_matrix, axis=1)
    
    # Compute the total number of annotations
    n = int(np.sum(coincidence_matrix))

    # Calculate the observed agreement (term1)
    term1 = np.sum(np.diagonal(coincidence_matrix)) / n

    # Calculate the expected agreement by chance (term2)
    term2 = np.sum(np.multiply(nc, nc - 1)) / (n * (n - 1))

    # Calculate Krippendorff's alpha
    alpha = (term1 - term2) / (1 - term2)

    # Return the computed alpha value
    return alpha

import numpy as np

def compute_entropy(column, choices):
    """
    Compute the Shannon entropy for a given column of choices.

    Shannon entropy quantifies the uncertainty or randomness in a distribution of 
    values. This function calculates the normalized entropy, where a value of 1 
    indicates maximum uncertainty (uniform distribution) and 0 indicates no uncertainty.

    Parameters:
    -----------
    column : pandas.Series
        A column of data containing choices made by coders, with potential `NaN` values 
        representing missing data.
    choices : list
        A list of all possible choices that can appear in the column.

    Returns:
    --------
    float
        The normalized Shannon entropy for the given column, ranging from 0 to 1.
    """
    # Drop NaN values from the column
    valid_choices = column.dropna()

    # Compute the frequency of each choice as probabilities
    counts = valid_choices.value_counts(normalize=True)

    # Create a list of probabilities for all choices, defaulting to 0 for missing ones
    probs = [counts.get(choice, 0) for choice in choices]

    # Compute Shannon entropy (only for non-zero probabilities)
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

    # Normalize entropy by the maximum possible entropy (log2 of the number of choices)
    normalized_entropy = entropy / np.log2(len(choices))

    # Return the normalized Shannon entropy
    return normalized_entropy


def compute_normalized_entropies(df, choices):
    """
    Compute and sort normalized Shannon entropies for all columns in a DataFrame.

    This function calculates the normalized Shannon entropy for each column in the 
    given DataFrame using a predefined set of choices. The entropies quantify the 
    uncertainty in the distribution of values within each column. The results are 
    returned in descending order of entropy.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame where columns represent different data categories or transcripts, 
        and rows represent coders or observations. Columns may contain `NaN` values 
        for missing data.
    choices : list
        A list of all possible choices that can appear in the DataFrame.

    Returns:
    --------
    pandas.Series
        A Series containing normalized entropies for each column in the DataFrame, 
        sorted in descending order.
    """
    # Apply the compute_entropy function to each column in the DataFrame
    entropies = df.apply(lambda col: compute_entropy(col, choices))

    # Sort the computed entropies in descending order
    sorted_entropies = entropies.sort_values(ascending=False)

    # Return the sorted entropies
    return sorted_entropies
def sample_df(df, frac_replace):
    """
    Randomly replace a fraction of non-NaN values in a DataFrame with NaN.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to sample from. It may contain `NaN` values.
    frac_replace : float
        The fraction of non-NaN values to replace with NaN (e.g., 0.1 to replace 10%).

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the specified fraction of non-NaN values replaced with NaN.
    """
    Q_sampled = df.copy()

    # Flatten the df to a 1D array 
    flat_values = Q_sampled.to_numpy().flatten()
    non_nan_mask = ~pd.isna(flat_values)

    # Calculate the number of non-NaN values to replace
    num_values_to_replace = int(non_nan_mask.sum() * frac_replace)

    # Randomly choose indices to replace
    indices_to_replace = np.random.choice(len(flat_values), num_values_to_replace, replace=False)

    # Replace the selected values with NaN
    for index in indices_to_replace:
        row_index, col_index = divmod(index, Q_sampled.shape[1]) 
        Q_sampled.iat[row_index, col_index] = np.nan

    return Q_sampled


def find_disagreements(Q_df):
    """
    Identify disagreements between annotators for each project.

    Parameters:
    -----------
    Q_df : pandas.DataFrame
        A DataFrame where rows correspond to annotators and columns to projects.
        Each cell contains an annotation or NaN for missing data.

    Returns:
    --------
    dict
        A dictionary mapping each annotator to a list of project IDs where disagreements occurred.
    """
    revisit_dict = {annotator: [] for annotator in Q_df.index}

    # Iterate over each project (column) in the DataFrame
    for project_id, col in Q_df.iteritems():
        for annotator in Q_df.index:
            annotator_value = col.loc[annotator]

            # Identify the other annotator and their value
            other_annotator = [a for a in Q_df.index if a != annotator][0]
            other_annotator_value = Q_df.loc[other_annotator, project_id]

            # Check for non-NaN disagreement
            if pd.notna(annotator_value) and pd.notna(other_annotator_value) and annotator_value != other_annotator_value:
                revisit_dict[annotator].append(project_id)
                revisit_dict[other_annotator].append(project_id)

    return revisit_dict

def merge_revisit_dicts(dict1, dict2, dict3):
    """
    Merge multiple revisit dictionaries into one.

    Parameters:
    -----------
    dict1, dict2, dict3 : dict
        Dictionaries mapping annotators to project IDs for disagreement for each question.

    Returns:
    --------
    dict
        A merged dictionary containing all unique project IDs for each annotator, sorted.
    """
    merged_dict = {}
    for annotator in set(dict1.keys()).union(dict2.keys()).union(dict3.keys()):
        merged_dict[annotator] = sorted(set(dict1.get(annotator, []) + dict2.get(annotator, []) + dict3.get(annotator, [])))
    return merged_dict

def merge_revisit_dicts_to_csv(merged_dict, file_name='disagreements_dict.csv'):
    """
    Save the merged revisit dictionary to a CSV file.

    Parameters:
    -----------
    merged_dict : dict
        A dictionary mapping annotators to lists of disagreement project IDs.
    file_name : str, optional
        The name of the CSV file to save. Defaults to 'disagreements_dict.csv'.

    Returns:
    --------
    None
    """
    merged_df = pd.DataFrame.from_dict(merged_dict, orient='index').transpose()
    merged_df.to_csv(file_name, index=False)


def compute_lengths(disagreements_dict):
    """
    Compute the number of disagreements for each annotator.

    Parameters:
    -----------
    disagreements_dict : dict
        A dictionary mapping annotators to lists of disagreement project IDs.

    Returns:
    --------
    dict
        A dictionary mapping each annotator to the count of disagreement projects.
    """
    return {annotator: len(projects) for annotator, projects in disagreements_dict.items()}


def get_pair_counter(Q3_df):
    """
    Count pairwise annotator interactions in a DataFrame.

    Parameters:
    -----------
    Q3_df : pandas.DataFrame
        A DataFrame where rows represent annotators and columns represent projects.

    Returns:
    --------
    Counter
        A Counter object where keys are annotator pairs and values are their interaction counts.
    """
    pair_counter = Counter()

    for column in Q3_df.columns:
        non_nan_annotators = Q3_df[column].dropna().index.tolist()
        pairs = list(combinations(non_nan_annotators, 2))
        pair_counter.update(pairs)

    return pair_counter

def get_matrix_df(pair_counter):
    """
    Create a DataFrame representing pairwise annotator interactions.

    Parameters:
    -----------
    pair_counter : Counter
        A Counter object mapping annotator pairs to interaction counts.

    Returns:
    --------
    pandas.DataFrame
        A square DataFrame where rows and columns are annotators, and values represent pair counts.
    """
    actual_annotators = list(set([annotator for pair in pair_counter for annotator in pair]))
    matrix_df = pd.DataFrame(index=actual_annotators, columns=actual_annotators).fillna(0)


    for (annotator1, annotator2), count in pair_counter.items():
        matrix_df.loc[annotator1, annotator2] = count
        matrix_df.loc[annotator2, annotator1] = count  # Bidirectional relationship

    return matrix_df

def extract_text(entry):
    """
    Extract all text from a nested list of dictionaries.

    Parameters:
    -----------
    entry : list
        A list containing dictionaries with a 'text' key, or other structures.

    Returns:
    --------
    list or None
        A list of extracted text if the input structure is valid; otherwise, None.
    """
    if entry and isinstance(entry, list):
        texts = []
        for item in entry:
            if isinstance(item, dict) and 'text' in item:
                texts.extend(item['text'])  # Extract all text entries
        return texts
    return None
