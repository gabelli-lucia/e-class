import csv
import re
import argparse
from enum import Enum
from math import sqrt
from statistics import mean, stdev
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas
from scipy.stats import mannwhitneyu

import pandas as pd
import logging as log
import numpy as np
import textwrap

from deprecated import utils
import conf

# This is needed to export XLSX files. I do not know why pandas does not import it by itself.
import openpyxl
import matplotlib

log.basicConfig(level=log.INFO)


class PrePost(Enum):
    """This enum specifies if we're reading a PRE, MID, or POST file."""
    PRE = 1
    MID = 2
    POST = 3


def read_files(pp, *files):
    """Reads and processes CSV files containing survey data.

    This function reads multiple CSV files, concatenates them, removes specified columns,
    filters rows based on CHECK columns, standardizes IDs, handles duplicate IDs,
    inverts specified columns, and processes matricola numbers.

    Args:
        pp (PrePost): Enum indicating if this is PRE or POST data
        *files: Variable number of CSV file paths to read

    Returns:
        pandas.DataFrame: Processed dataframe containing the survey data
    """
    df = pd.DataFrame()

    for file in files:
        df = pd.concat([df, pd.read_csv(file, skiprows=0 if file == files[0] else 1)])
        log.info(f"{pp.name}: Reading {file}: {len(df)} rows")
    log.debug(f"{pp.name}: Read shape {df.shape}")

    log.debug(f"Found columns: {df.columns}")
    log.debug("Removing _ prefixes from column names")
    df = utils.remove_prefix_from_columns(df)
    log.debug(f"Current columns: {df.columns}")

    # Removing unuseful columns
    df.drop(conf.COL_TO_DROP, axis=1, inplace=True)

    # Identifying the CHECK columns
    check_columns = [col for col in df.columns if conf.COL_CHECK in col]
    log.debug(f"{pp.name}: CHECK columns {check_columns}")

    # Removing rows where people did not answer 4 in all the CHECK columns
    for check in check_columns:
        df = df[df[check] == 4]
    # Removing the CHECK columns
    df.drop(check_columns, axis=1, inplace=True)
    log.info(f"{pp.name}: Passed CHECK {len(df)} rows")

    # all ID should be upper case
    for index, row in df.iterrows():
        df.at[index, conf.COL_ID] = row[conf.COL_ID].upper()

    # We reject the file if there are duplicate IDs.
    dup = df.duplicated(subset=[conf.COL_ID])
    ids = df[conf.COL_ID]
    if True in dup.unique():
        log.warning(
            f"{pp.name}: Removing duplicate IDs: {df[ids.isin(ids[ids.duplicated()])].sort_values(conf.COL_ID)[conf.COL_ID].unique()}")
        df.drop_duplicates(conf.COL_ID, keep=False, inplace=True)
        # exit(1)

    # Inverting the columns that need inversion
    log.debug(f"Inverting columns {conf.COL_TO_INVERT}")
    invert_columns = [col for col in df.columns if column_is_to_be_inverted(col)]
    log.debug(f"Inverting columns {invert_columns}")
    for index, row in df.iterrows():
        for col in invert_columns:
            df.at[index, col] = conf.MAX_POINTS - int(row[col])

    # log.info(f.head())
    # Removing invalid Matricola
    df[conf.COL_MATRICOLA] = df[conf.COL_MATRICOLA].map(lambda x: str(x))
    df[conf.COL_MATRICOLA] = df[conf.COL_MATRICOLA].map(lambda x: x.replace('.0', ''))
    df[conf.COL_MATRICOLA] = df[conf.COL_MATRICOLA].map(lambda x: x if x.isnumeric() else '')
    df[conf.COL_MATRICOLA] = df[conf.COL_MATRICOLA].map(lambda x: '' if x != '' and int(x) < 1000000 else x)
    df[conf.COL_MATRICOLA] = df[conf.COL_MATRICOLA].map(lambda x: '' if x != '' and int(x) > 3000000 else x)

    log.debug(f"{df.dtypes}")
    return df


def column_is_to_be_inverted(col):
    """This methods returns True if the name of column col is in the list of the columns
     that need to be inverted."""
    for column_name in conf.COL_TO_INVERT:
        if col.startswith(column_name + "->"):
            return True
    return False


def restore_matricola_from_id(sx: pd.DataFrame, dx: pd.DataFrame):
    """This method populates empty Matricola in sx looking up the Matricola in sx
    with the same ID."""
    log.debug(f"Restoring Matricola")
    # for each Matricola in sx
    for index, sx_matricola in sx[conf.COL_MATRICOLA].items():
        # if the Matricola is missing
        if sx_matricola == '':
            log.debug(f"Matricola not found at index {index}, trying to restore it")
            # The ID for this Matricola
            sx_id = sx.at[index, conf.COL_ID]
            log.debug(f" Matricola at index {index} has ID {sx_id}")
            if sx_id is not None and sx_id != '':
                dx_row = dx.loc[dx[conf.COL_ID] == sx_id]
                if not dx_row.empty:
                    log.debug(f" DX has ID {sx_id} at row {dx_row.index}")
                    dx_matricola = dx_row.iloc[0].at[conf.COL_MATRICOLA]
                    if dx_matricola is not None and dx_matricola != '':
                        log.debug(f" ID {sx_id} has Matricola {dx_matricola}")
                        sx.at[index, conf.COL_MATRICOLA] = dx_matricola


def populate_matricola_from_id(df: pd.DataFrame):
    """This method populates the empty Matricola with the ID of that row."""
    for index, row in df.iterrows():
        if row[conf.COL_MATRICOLA] == '':
            df.at[index, conf.COL_MATRICOLA] = row[conf.COL_ID]


def remove_unmatched_rows(sx: pd.DataFrame, dx: pd.DataFrame):
    log.debug("Removing unmatched rows")
    # log.info(f"BEFORE Sizes: sx={len(sx)}, dx={len(dx)}")
    sx = sx.loc[np.where(sx[conf.COL_MATRICOLA].isin(dx[conf.COL_MATRICOLA]), True, False)]
    # log.info(f"AFTER Sizes: sx={len(sx)}, dx={len(dx)}")
    # log.info(sx)
    return sx


def column_is_to_be_mapped(column_name, veto):
    """Determines if a column should be mapped based on its name and veto list.

    This function checks if a column name contains the mapping indicator '->' and
    is not in the veto list of column prefixes that should not be mapped.

    Args:
        column_name (str): The name of the column to check
        veto (list): List of column name prefixes that should not be mapped

    Returns:
        bool: True if the column should be mapped, False otherwise
    """
    if not '->' in column_name:
        return False
    for c in veto:
        if column_name.startswith(c + "->"):
            return False
    return True


def clone_and_map(df: pd.DataFrame, mapping: list, veto: list):
    """Creates a copy of the dataframe with values mapped according to the provided mapping.

    This function creates a copy of the input dataframe and maps the values in mappable columns
    (determined by column_is_to_be_mapped) using the provided mapping array.

    Args:
        df (pandas.DataFrame): DataFrame to clone and map
        mapping (list): List of values to map to (index+1 in original maps to value at index)
        veto (list): List of column name prefixes to exclude from mapping

    Returns:
        pandas.DataFrame: A new dataframe with mapped values
    """
    copy = df.copy()
    for c in [c for c in copy.columns if column_is_to_be_mapped(c, veto)]:
        # all ID should be upper case
        for index, row in copy.iterrows():
            copy.at[index, c] = mapping[int(row[c]) - 1]
    return copy


def mean_and_sigma_of_columns_by_name(df: pd.DataFrame, sub_column_name: str):
    """Calculates statistics for columns containing a specific substring.

    This method calculates the mean, standard deviation, and 95% confidence interval
    error for all columns in the dataframe that contain the specified substring in their name.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to analyze
        sub_column_name (str): Substring to filter columns (e.g., 'TU' for student columns)

    Returns:
        tuple: A tuple containing (mean, standard_deviation, error_95_percent)
    """
    s = pd.concat([df[col] for col in df.columns if sub_column_name in col])

    err = 1.96 * (s.std() / np.sqrt(s.count()))

    return round(s.mean(), 2), round(s.std(), 2), round(err, 2)


def chart_means(pre, mid, post, filename):
    """Creates a bar chart comparing pre, mid, and post means for 'YOU' and 'Expert' categories.

    This function calculates means, standard deviations, and error values for
    pre, mid, and post data, then creates a bar chart visualization and saves it to a file.

    Args:
        pre (pandas.DataFrame): DataFrame containing pre-test data
        mid (pandas.DataFrame): DataFrame containing mid-test data
        post (pandas.DataFrame): DataFrame containing post-test data
        filename (str): Path where the chart image will be saved

    Returns:
        None
    """

    mean_pre_tu, sigma_pre_tu, err_pre_tu = mean_and_sigma_of_columns_by_name(pre, conf.COL_TU)
    mean_pre_exp, sigma_pre_exp, err_pre_exp = mean_and_sigma_of_columns_by_name(pre, conf.COL_EXP)

    mean_mid_tu, sigma_mid_tu, err_mid_tu = mean_and_sigma_of_columns_by_name(mid, conf.COL_TU)
    mean_mid_exp, sigma_mid_exp, err_mid_exp = mean_and_sigma_of_columns_by_name(mid, conf.COL_EXP)

    mean_post_tu, sigma_post_tu, err_post_tu = mean_and_sigma_of_columns_by_name(post, conf.COL_TU)
    mean_post_exp, sigma_post_exp, err_post_exp = mean_and_sigma_of_columns_by_name(post, conf.COL_EXP)

    log.debug(f"Averages pre:  TU: {mean_pre_tu}, EXP: {mean_pre_exp}")
    log.debug(f"Averages mid:  TU: {mean_mid_tu}, EXP: {mean_mid_exp}")
    log.debug(f"Averages post: TU: {mean_post_tu}, EXP: {mean_post_exp}")

    log.debug(f"Sigmas pre:    TU: {sigma_pre_tu}, EXP: {sigma_pre_exp}")
    log.debug(f"Sigmas mid:    TU: {sigma_mid_tu}, EXP: {sigma_mid_exp}")
    log.debug(f"Sigmas post:   TU: {sigma_post_tu}, EXP: {sigma_post_exp}")

    log.debug(f"Err 95% pre:   TU: {err_pre_tu}, EXP: {err_pre_exp}")
    log.debug(f"Err 95% mid:   TU: {err_mid_tu}, EXP: {err_mid_exp}")
    log.debug(f"Err 95% post:  TU: {err_post_tu}, EXP: {err_post_exp}")

    means = pd.DataFrame(
        {'stage': ['Pre', 'Mid', 'Post'], 
         'YOU': [mean_pre_tu, mean_mid_tu, mean_post_tu], 
         'Expert': [mean_pre_exp, mean_mid_exp, mean_post_exp]})

    err = [[err_pre_tu, err_mid_tu, err_post_tu], [err_pre_exp, err_mid_exp, err_post_exp]]

    ax = means.plot.bar(x='stage', y=['YOU', 'Expert'], rot=0, capsize=4, yerr=err, color=['green', 'red'])
    # Titolo del grafico
    ax.set_title('Overall E-CLASS score on "What do YOU think..."\nand "What do EXPERTS think..." statements')
    # Cancella il titolo dell'asse X (che sarebbe 'stage')
    ax.set_xlabel('')
    # Titolo dell'asse Y
    ax.set_ylabel('Fraction of statements\nwith expert-like response')
    # Estremi dell'asse Y
    ax.set_ylim([0, 1])
    # Scrive i valori sulle barre. Bisogna considerare solo i container di indice pari poiché i dispari sono i container
    # degli errori.
    for container in ax.containers[1::2]:
        ax.bar_label(container)

    fig = ax.get_figure()
    fig.savefig(filename, dpi=200)


def cohensd(c0, c1):
    """Calculates Cohen's d effect size between two collections of values.

    Cohen's d is a standardized measure of effect size, representing the difference
    between two means divided by the pooled standard deviation.

    Args:
        c0 (list or array-like): First collection of numeric values
        c1 (list or array-like): Second collection of numeric values

    Returns:
        float: Cohen's d effect size value
    """
    cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
    return cohens_d


def chart_what_do_you_think(pre, mid, post, pre3, mid3, post3, substring, filename):
    """Creates a chart showing pre/mid/post comparison with statistical significance indicators.

    This function creates a visualization that shows pre, mid, and post means for columns containing
    the specified substring, along with Cohen's d effect size and significance markers (stars)
    for statistically significant changes (p < conf.EFFECT_THRESHOLD).

    Args:
        pre (pandas.DataFrame): Original pre-test data
        mid (pandas.DataFrame): Original mid-test data
        post (pandas.DataFrame): Original post-test data
        pre3 (pandas.DataFrame): Mapped pre-test data
        mid3 (pandas.DataFrame): Mapped mid-test data
        post3 (pandas.DataFrame): Mapped post-test data
        substring (str): Substring to filter columns (e.g., 'TU' for student columns)
        filename (str): Path where the chart image will be saved

    Returns:
        None
    """
    columns = [c for c in pre3.columns if column_is_to_be_mapped(c, conf.COL_DONT_MAP_POST) and substring in c]
    pre3_means = {col_name: pre3[col_name].mean() for col_name in columns}
    mid3_means = {col_name: mid3[col_name].mean() for col_name in columns}
    post3_means = {col_name: post3[col_name].mean() for col_name in columns}

    # Calculate effect sizes and p-values for pre-mid, mid-post, and pre-post
    effect_pre_mid = {col_name: mannwhitneyu(x=pre[col_name], y=mid[col_name]).pvalue for col_name in columns}
    effect_mid_post = {col_name: mannwhitneyu(x=mid[col_name], y=post[col_name]).pvalue for col_name in columns}
    effect_pre_post = {col_name: mannwhitneyu(x=pre[col_name], y=post[col_name]).pvalue for col_name in columns}

    log.debug(f"Effect pre-mid: {effect_pre_mid}")
    log.debug(f"Effect mid-post: {effect_mid_post}")
    log.debug(f"Effect pre-post: {effect_pre_post}")

    cohen_pre_mid = {col_name: abs(cohensd(pre[col_name], mid[col_name])) for col_name in columns}
    cohen_mid_post = {col_name: abs(cohensd(mid[col_name], post[col_name])) for col_name in columns}
    cohen_pre_post = {col_name: abs(cohensd(pre[col_name], post[col_name])) for col_name in columns}

    log.debug(f"Cohen pre-mid: {cohen_pre_mid}")
    log.debug(f"Cohen mid-post: {cohen_mid_post}")
    log.debug(f"Cohen pre-post: {cohen_pre_post}")

    # We'll use pre-post effect for stars, but could be modified to show all significant changes
    columns_with_effect = []

    # Zero out cohen values for non-significant changes
    for col_name in columns:
        if effect_pre_post[col_name] > conf.EFFECT_THRESHOLD:
            cohen_pre_post[col_name] = 0.0
        else:
            columns_with_effect.append(col_name)

    # Place stars at the maximum value plus a small offset
    stars = {col_name: max(pre3_means[col_name], mid3_means[col_name], post3_means[col_name]) + 0.1 
             for col_name in columns_with_effect}
    log.debug(f"Stars: {stars}")

    df2 = (pd.DataFrame({
        'Pre': pre3_means,
        'Mid': mid3_means,
        'Post': post3_means,
        'Cohen': cohen_pre_post,  # Using pre-post cohen for the bars
        'Stars': stars})
           .sort_values(by=['Pre']))

    # Plot the data
    ax1 = df2[['Pre']].plot(zorder=0, marker='s', markersize=8, linestyle='dashed')
    df2[['Mid']].plot(ax=ax1, zorder=1, marker='^', markersize=8, linestyle='dashed')
    df2[['Post']].plot(ax=ax1, zorder=1, marker='o', markersize=8, linestyle='dashed')
    df2[['Stars']].plot(ax=ax1, marker='*', markersize=10, color='black', zorder=2, legend=False, linestyle='none')
    df2[['Cohen']].plot(ax=ax1, kind='bar', width=0.8, color='grey', ylim=(0, 1), zorder=1, alpha=0.35,
                        secondary_y=True, legend=False)

    # Titolo del grafico
    ax1.set_title('Average score on "What do YOU think" questions')
    # Etichette sull'asse X
    labels = [re.search('(.*)->', item.get_text()).group(1) for item in ax1.get_xticklabels()]
    ax1.set_xticklabels(labels, fontsize=8)
    # Titolo dell'asse X
    ax1.set_xlabel('Question')
    # Etichette sull'asse Y
    ax1.tick_params(axis='y', labelsize=8)
    # Titolo dell'asse Y
    ax1.set_ylabel('Average Agreement with Experts')
    # Titolo dell'asse Y secondario
    ax1.twinx().set_ylabel('Effect Size (Pre-Post)')
    # Griglia con righe solo verticali
    ax1.xaxis.grid(True, linestyle='dashed', linewidth=0.5)

    fig = ax1.get_figure()
    fig.savefig(filename, dpi=200)


def find_column(i, substring, df: pd.DataFrame):
    """Finds a column in the DataFrame that starts with a specific index and contains a substring.

    This function searches through the columns of a DataFrame to find a column that 
    starts with the specified 1-based index followed by '->' and contains the given substring.

    Args:
        i (int): 1-based index to look for at the start of column names
        substring (str): Substring that must be present in the column name
        df (pd.DataFrame): DataFrame to search in

    Returns:
        str or None: The name of the matching column if found, None otherwise
    """
    for col in df.columns:
        if substring in col and col.startswith(f"{i}->"):
            return col
    return None


def find_question(i, df: pd.DataFrame):
    """Retrieves a formatted question string based on a 1-based index.

    This function formats a question string by combining a 'Q' prefix with the index
    and the corresponding question text from the conf.Q list.

    Args:
        i (int): 1-based index of the question to retrieve
        df (pd.DataFrame): DataFrame (not used in the function but kept for interface consistency)

    Returns:
        str: Formatted question string in the format "Q{i}: {question_text}"
    """
    return f"Q{i}: {conf.Q[i - 1]}"


def draw_arrow(ax, start_x, end_x, y_position, arrowhead_size, color):
    """
    Draw an arrow between two points on the x-axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the arrow will be drawn

    start_x : float
        The starting x-coordinate of the arrow

    end_x : float
        The ending x-coordinate of the arrow

    y_position : float
        The y-coordinate for both start and end points

    arrowhead_size : float
        The size of the arrowhead

    color : str
        The color of the arrow
    """
    # Calculate arrow length, accounting for the arrowhead
    if abs(end_x - start_x) >= arrowhead_size:
        if end_x - start_x >= 0:  # Moving right
            arrow_length = end_x - start_x - arrowhead_size
        else:  # Moving left
            arrow_length = end_x - start_x + arrowhead_size
    else:  # Very small change, use minimal arrow length
        arrow_length = 0.000001
        arrowhead_size = end_x - start_x

    # Draw the arrow
    ax.arrow(start_x, y_position, arrow_length, 0, 
             head_width=arrowhead_size, head_length=arrowhead_size, 
             fc=color, ec=color, zorder=10)


# Function to draw the plot for each row triplet
def draw_plot(ax, top_row, bottom_row, question):
    """
    Draw a comparison plot showing user and expert responses for a single question.

    This function creates a visualization that compares user ("YOU") and expert responses
    for a given question. It draws squares representing the responses, arrows showing
    the direction and magnitude of changes, and confidence intervals as colored regions.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the plot will be drawn

    top_row : numpy.ndarray
        Array containing data for the user ("YOU") response:
        - top_row[0]: Initial (PRE) position
        - top_row[1]: Middle (MID) position
        - top_row[2]: Final (POST) position
        - top_row[-2]: Lower confidence interval bound
        - top_row[-1]: Upper confidence interval bound

    bottom_row : numpy.ndarray
        Array containing data for the expert response:
        - bottom_row[0]: Initial (PRE) position
        - bottom_row[1]: Middle (MID) position
        - bottom_row[2]: Final (POST) position
        - bottom_row[-2]: Lower confidence interval bound
        - bottom_row[-1]: Upper confidence interval bound

    question : str
        The text of the question to display as the y-axis label
    """
    square_height = 0.15  # Adjust the size of the squares
    fill_height = 4 * square_height / 5  # - 0.01  # Adjust as needed
    top_position = 0.75  # Adjust the vertical position of the top square
    bottom_position = 0.25  # Adjust the vertical position of the bottom square

    # Draw confidence interval regions as semi-transparent rectangles
    # For the user (top) response: green rectangle from lower CI to upper CI
    ax.fill_betweenx([top_position - fill_height / 2, top_position + fill_height / 2], top_row[-2], top_row[-1],
                     color='green', alpha=0.5,
                     edgecolor='none', linewidth=0)
    # For the expert (bottom) response: red rectangle from lower CI to upper CI
    ax.fill_betweenx([bottom_position - fill_height / 2, bottom_position + fill_height / 2], bottom_row[-2],
                     bottom_row[-1], color='red', alpha=0.5,
                     edgecolor='none', linewidth=0)

    # Draw square representing the user's initial (PRE) position (green outline)
    square_top_pre = Rectangle((top_row[0] - square_height / 4, top_position - square_height / 2), square_height / 2,
                           square_height,
                           edgecolor='green', facecolor='white', label='YOU')
    ax.add_patch(square_top_pre)

    # Draw triangle representing the user's middle (MID) position (green outline)
    # We'll use a different shape for MID to distinguish it
    mid_marker_size = square_height / 2
    ax.scatter(top_row[1], top_position, marker='^', s=mid_marker_size*100, 
               edgecolor='green', facecolor='white', zorder=5)

    # Draw square representing the expert's initial (PRE) position (red outline)
    square_bottom_pre = Rectangle((bottom_row[0] - square_height / 4, bottom_position - square_height / 2),
                              square_height / 2, square_height,
                              edgecolor='red', facecolor='white', label='EXPERT')
    ax.add_patch(square_bottom_pre)

    # Draw triangle representing the expert's middle (MID) position (red outline)
    ax.scatter(bottom_row[1], bottom_position, marker='^', s=mid_marker_size*100, 
               edgecolor='red', facecolor='white', zorder=5)

    # Define the size of arrowheads for the arrows
    arrowhead_size = 0.02

    # Draw arrows for user (top) response - from PRE to MID
    draw_arrow(ax, top_row[0], top_row[1], top_position, arrowhead_size, 'black')

    # Draw arrows for user (top) response - from MID to POST
    draw_arrow(ax, top_row[1], top_row[2], top_position, arrowhead_size, 'black')

    # Draw arrows for expert (bottom) response - from PRE to MID
    draw_arrow(ax, bottom_row[0], bottom_row[1], bottom_position, arrowhead_size, 'black')

    # Draw arrows for expert (bottom) response - from MID to POST
    draw_arrow(ax, bottom_row[1], bottom_row[2], bottom_position, arrowhead_size, 'black')

    # Set the question text as the y-axis label
    # Position the label in the middle of the y-axis (at 0.5)
    # Use textwrap to wrap long question text to multiple lines for better readability
    ax.set_yticks([0.5])
    ax.set_yticklabels([textwrap.fill(question, width=22)])  # Width of 22 characters per line

    ax.set_xlim(0, 1)  # Set x-axis range from 0 to 1 (fraction of responses)
    ax.set_ylim(0, 1)  # Set y-axis range

    # Add vertical light grey dashed grid lines every 0.2 on the x-axis
    # This helps with reading the values from the plot
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_xticklabels([f'{i:.1f}' for i in np.arange(0, 1.2, 0.2)])
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)


def chart_before_after(pre: pd.DataFrame, mid: pd.DataFrame, post: pd.DataFrame, filename):
    """Creates a comprehensive chart comparing pre, mid, and post data for all questions.

    This function generates a grid of subplots (15 rows, 2 columns) showing the pre, mid, and post
    test results for all questions, sorted by pre-test mean values. Each subplot displays
    the question text and visualizes the pre/mid/post comparison with error bars.

    Args:
        pre (pd.DataFrame): DataFrame containing pre-test data
        mid (pd.DataFrame): DataFrame containing mid-test data
        post (pd.DataFrame): DataFrame containing post-test data
        filename (str): Path where the chart image will be saved

    Returns:
        None
    """
    log.debug('chart_before_after')
    # Create a single figure with 30 subplots (15 rows, 2 columns)
    # This will display all 30 questions in a grid layout
    fig, axes = plt.subplots(15, 2, figsize=(10, 20), sharex=True, gridspec_kw={'width_ratios': [1, 1]})

    sorted_by_pre_mean_tu = np.argsort(list(map(lambda col: col.mean(),
                                                map(lambda c: pre[c],
                                                    map(lambda i: find_column(i, conf.COL_TU, pre),
                                                        range(1, len(conf.Q) + 1))))))

    i = 0
    for ii in sorted_by_pre_mean_tu:
        log.debug('Looking for Question %d', ii + 1)
        column_tu = find_column(ii + 1, conf.COL_TU, pre)
        pre_mean_tu = pre[column_tu].mean()
        pre_std_tu = pre[column_tu].std()
        pre_err_tu = 1.96 * (pre_std_tu / np.sqrt(pre[column_tu].count()))

        mid_mean_tu = mid[column_tu].mean()
        mid_std_tu = mid[column_tu].std()

        post_mean_tu = post[column_tu].mean()
        post_std_tu = post[column_tu].std()

        column_exp = find_column(ii + 1, conf.COL_EXP, pre)
        pre_mean_exp = pre[column_exp].mean()
        pre_std_exp = pre[column_exp].std()
        pre_err_exp = 1.96 * (pre_std_exp / np.sqrt(pre[column_exp].count()))

        mid_mean_exp = mid[column_exp].mean()
        mid_std_exp = mid[column_exp].std()

        post_mean_exp = post[column_exp].mean()
        post_std_exp = post[column_exp].std()

        question = find_question(ii + 1, post)

        if question is not None:
            ax = axes[i // 2, i % 2]
            draw_plot(ax,
                      [pre_mean_tu, mid_mean_tu, post_mean_tu, pre_mean_tu - pre_err_tu, pre_mean_tu + pre_err_tu],
                      [pre_mean_exp, mid_mean_exp, post_mean_exp, pre_mean_exp - pre_err_exp, pre_mean_exp + pre_err_exp],
                      question)

        if i % 2 == 1:
            ax = axes[i // 2, i % 2]
            # Position the y-axis ticks on the right side for the right column
            # This improves readability when questions are displayed on both sides
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        i = i + 1

    # Add a legend to the last plot in the grid
    # This explains the color coding (blue for user, green for expert)
    axes[14, 1].legend()

    # Add x-axis labels only to the bottom plots
    # This avoids cluttering the figure with redundant labels
    axes[14, 0].set_xlabel('Fraction of class with expert-like response')
    axes[14, 1].set_xlabel('Fraction of class with expert-like response')

    # Set the main title for the entire figure
    axes[0, 0].set_title('What do YOU think? vs. what do EXPERTS think?', y=1.0, x=1.05, fontsize=21)

    # Adjust the margins to ensure all elements fit properly
    plt.subplots_adjust(bottom=0.15, top=0.8)

    # Set consistent font size for all y-tick labels (question text)
    for ax in axes.flatten():
        ax.tick_params(axis='y', labelsize=10)

    # Optimize the layout and adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.1)  # No vertical space, minimal horizontal space

    fig.savefig(filename, format='png', dpi=400)
    plt.close()  # Close the figure to free up memory resources


def dump_success(df: pd.DataFrame, filename: str):
    """Calculates and exports success rates for questions to a CSV file.

    This function identifies columns containing the conf.COL_SUCCESS substring,
    calculates the fraction of successful responses (value=1) for each question,
    and writes the results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the survey data with success columns
        filename (str): Path where the CSV file will be saved

    Returns:
        None
    """
    log.debug('Looking for Important Questions')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['question', 'success'], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i in range(1, len(conf.Q) + 1):
            col_name = find_column(i, conf.COL_SUCCESS, df)
            if col_name is not None:
                count = df[col_name].count()
                v_count = df[col_name].value_counts()[1]
                fraction = v_count / count
                log.debug(f"Q{i},{fraction:.3f}")
                writer.writerow({
                    'question': col_name,
                    'success': f"{fraction:.3f}"
                })


def dump_averages(pre, mid, post, filename):
    """Calculates and exports statistical comparisons between pre, mid, and post data to a CSV file.

    This function computes the average values for pre, mid, and post data for each mappable column,
    calculates Mann-Whitney U test p-values and Cohen's d effect sizes for the differences,
    and writes all these statistics to a CSV file.

    Args:
        pre (pd.DataFrame): DataFrame containing pre-test data
        mid (pd.DataFrame): DataFrame containing mid-test data
        post (pd.DataFrame): DataFrame containing post-test data
        filename (str): Path where the CSV file will be saved

    Returns:
        None
    """
    columns = [c for c in pre.columns if column_is_to_be_mapped(c, conf.COL_DONT_MAP_POST)]
    effect_pre_mid = {col_name: mannwhitneyu(x=pre[col_name], y=mid[col_name]).pvalue for col_name in columns}
    effect_mid_post = {col_name: mannwhitneyu(x=mid[col_name], y=post[col_name]).pvalue for col_name in columns}
    effect_pre_post = {col_name: mannwhitneyu(x=pre[col_name], y=post[col_name]).pvalue for col_name in columns}

    cohen_pre_mid = {col_name: abs(cohensd(pre[col_name], mid[col_name])) for col_name in columns}
    cohen_mid_post = {col_name: abs(cohensd(mid[col_name], post[col_name])) for col_name in columns}
    cohen_pre_post = {col_name: abs(cohensd(pre[col_name], post[col_name])) for col_name in columns}

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, 
                               fieldnames=['question', 'pre avg', 'mid avg', 'post avg', 
                                          'pre-mid mann-whitney', 'mid-post mann-whitney', 'pre-post mann-whitney',
                                          'pre-mid cohen', 'mid-post cohen', 'pre-post cohen'],
                               quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for col_name in columns:
            writer.writerow({
                'question': col_name,
                'pre avg': pre[col_name].mean(),
                'mid avg': mid[col_name].mean(),
                'post avg': post[col_name].mean(),
                'pre-mid mann-whitney': effect_pre_mid[col_name],
                'mid-post mann-whitney': effect_mid_post[col_name],
                'pre-post mann-whitney': effect_pre_post[col_name],
                'pre-mid cohen': cohen_pre_mid[col_name],
                'mid-post cohen': cohen_mid_post[col_name],
                'pre-post cohen': cohen_pre_post[col_name]
            })


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Create argument parser
    parser = argparse.ArgumentParser(description='Process E-CLASS survey data.')
    parser.add_argument('input_files', nargs='+', help='Input PRE file(s). MID and POST files will be derived automatically.')
    parser.add_argument('--threshold', type=float, default=conf.EFFECT_THRESHOLD,
                        help=f'Effect size threshold (default: {conf.EFFECT_THRESHOLD})')
    parser.add_argument('--matricola', type=str, default=conf.COL_MATRICOLA,
                        help=f'Matricola column name (default: {conf.COL_MATRICOLA})')
    parser.add_argument('--lang', type=str, default="it",
                        help=f'File language (it, en; default: it)')

    # Parse arguments
    args = parser.parse_args()

    # Set the effect threshold from arguments
    conf.EFFECT_THRESHOLD = args.threshold
    conf.COL_MATRICOLA = args.matricola
    if args.lang == 'en':
        COL_TU = " YOU "
        COL_EXP = " experts "
        COL_SUCCESS = " important, "

    filenames = args.input_files
    isvalid, message = utils.verify_pre_post_files(*filenames)
    if not isvalid:
        log.error(message)
        exit(1)

    # log.info(f"Reading {filenames}")
    pre = read_files(PrePost.PRE, *filenames)

    mid_filenames = list(map(lambda x: x.replace(PrePost.PRE.name, PrePost.MID.name), filenames))
    # log.info(f"Reading {mid_filenames}")
    mid = read_files(PrePost.MID, *mid_filenames)

    post_filenames = list(map(lambda x: x.replace(PrePost.PRE.name, PrePost.POST.name), filenames))
    # log.info(f"Reading {post_filenames}")
    post = read_files(PrePost.POST, *post_filenames)

    log.info("Restoring Matricola")
    restore_matricola_from_id(pre, mid)
    restore_matricola_from_id(pre, post)
    restore_matricola_from_id(mid, pre)
    restore_matricola_from_id(mid, post)
    restore_matricola_from_id(post, pre)
    restore_matricola_from_id(post, mid)
    populate_matricola_from_id(pre)
    populate_matricola_from_id(mid)
    populate_matricola_from_id(post)

    # Remove unmatched rows across all three datasets
    pre = remove_unmatched_rows(pre, mid)
    mid = remove_unmatched_rows(mid, pre)
    pre = remove_unmatched_rows(pre, post)
    post = remove_unmatched_rows(post, pre)
    mid = remove_unmatched_rows(mid, post)
    post = remove_unmatched_rows(post, mid)

    # Final check to ensure all datasets are fully matched
    pre = remove_unmatched_rows(pre, mid)
    pre = remove_unmatched_rows(pre, post)
    mid = remove_unmatched_rows(mid, pre)
    mid = remove_unmatched_rows(mid, post)
    post = remove_unmatched_rows(post, pre)
    post = remove_unmatched_rows(post, mid)

    log.info(f"Removing unmatched rows. Survived pre: {len(pre)}, mid: {len(mid)}, post: {len(post)}")

    # TODO se ci sono duplicati, bisogna tranquillamente ignorare quelli che dall'altra parte non ci sono

    # Now, IDs are no more needed
    pre.drop([conf.COL_ID], axis=1, inplace=True)
    mid.drop([conf.COL_ID], axis=1, inplace=True)
    post.drop([conf.COL_ID], axis=1, inplace=True)
    # pre.to_excel(f"pre.xlsx")
    # post.to_excel(f"post.xlsx")

    pre2 = clone_and_map(pre, conf.MAPPING2, conf.COL_DONT_MAP_PRE)
    pre3 = clone_and_map(pre, conf.MAPPING3, conf.COL_DONT_MAP_PRE)
    mid2 = clone_and_map(mid, conf.MAPPING2, conf.COL_DONT_MAP_POST)  # Using POST settings for MID
    mid3 = clone_and_map(mid, conf.MAPPING3, conf.COL_DONT_MAP_POST)  # Using POST settings for MID
    post2 = clone_and_map(post, conf.MAPPING2, conf.COL_DONT_MAP_POST)
    post3 = clone_and_map(post, conf.MAPPING3, conf.COL_DONT_MAP_POST)
    # pre2.to_excel(f"pre2.xlsx")
    # pre3.to_excel(f"pre3.xlsx")
    # post2.to_excel(f"post2.xlsx")
    # post3.to_excel(f"post3.xlsx")

    log.info(f"Saving CSV files")
    dump_success(post3, 'out-success.csv')
    dump_success(mid3, 'out-success-mid.csv')
    dump_averages(pre2, mid2, post2, "out-medie.csv")

    log.info(f"Saving charts")
    chart_means(pre2, mid2, post2, 'out-chart-means.png')
    chart_what_do_you_think(pre, mid, post, pre3, mid3, post3, conf.COL_TU, 'out-chart-what-do-you-think.png')
    chart_before_after(pre2, mid2, post2, 'out-chart-after-before.png')

    log.info("Done.")
