import re
import sys
from enum import Enum
from math import sqrt
from statistics import mean, stdev

import pandas
from scipy.stats import mannwhitneyu

import pandas as pd
import logging as log
import numpy as np

import conf
# This is needed to export XLSX files. I do not know why pandas does not import it by itself.
import openpyxl
import matplotlib

import utils

log.basicConfig(level=log.DEBUG)


class PrePost(Enum):
    """This enum specifies if we're reading a PRE or POST file."""
    PRE = 1
    POST = 2


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
        log.info(f"{pp.name}: Reading {file}")
        df = pd.concat([df, pd.read_csv(file, skiprows=0 if file == files[0] else 1)])
    log.info(f"{pp.name}: Read shape {df.shape}")

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
    log.info(f"{pp.name}: Passed CHECK shape {df.shape}")

    # all ID should be upper case
    for index, row in df.iterrows():
        df.at[index, conf.COL_ID] = row[conf.COL_ID].upper()

    # We reject the file if there are duplicate IDs.
    dup = df.duplicated(subset=[conf.COL_ID])
    ids = df[conf.COL_ID]
    if True in dup.unique():
        log.error(
            f"{pp.name}: Duplicate IDs: {df[ids.isin(ids[ids.duplicated()])].sort_values(conf.COL_ID)[conf.COL_ID].unique()}")
        # exit(1)

    # Inverting the columns that need inversion
    log.info(f"Inverting columns {conf.COL_TO_INVERT}")
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


def column_is_to_be_mapped(column_name, veto):
    """Says whether this column values should be remapped.
    >>> column_is_to_be_mapped('Q06_3', ['Q01'])
    True
    >>> column_is_to_be_mapped('Q01_ID-1', ['Q01'])
    False
    >>> column_is_to_be_mapped('Titolo', ['Q01'])
    False
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


def join_by_matricola(sx: pd.DataFrame, dx: pd.DataFrame):
    """Joins two DataFrames by the matricola column.

    This method joins two DataFrames by the conf.COL_MATRICOLA column, adding suffixes
    "_pre" and "_post" to distinguish columns from each dataframe.

    Args:
        sx (pandas.DataFrame): First DataFrame (typically pre-test data)
        dx (pandas.DataFrame): Second DataFrame (typically post-test data)

    Returns:
        pandas.DataFrame: Joined DataFrame containing only rows with matching matricola values
    """
    # Joining the two tables
    log.info(f"Joining...")
    log.debug(f"{sx[conf.COL_MATRICOLA]}")
    log.debug(f"{dx[conf.COL_MATRICOLA]}")
    result = pd.merge(sx, dx, on=conf.COL_MATRICOLA, suffixes=("_pre", "_post"))
    log.info(f"Joined: got {result.shape}")
    log.debug(result.head())
    return result


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


def chart_means(pre, post, filename):
    """Creates a bar chart comparing pre and post means for 'YOU' and 'Expert' categories.

    This function calculates means, standard deviations, and error values for both
    pre and post data, then creates a bar chart visualization and saves it to a file.

    Args:
        pre (pandas.DataFrame): DataFrame containing pre-test data
        post (pandas.DataFrame): DataFrame containing post-test data
        filename (str): Path where the chart image will be saved

    Returns:
        None
    """

    mean_pre_tu, sigma_pre_tu, err_pre_tu = mean_and_sigma_of_columns_by_name(pre, conf.COL_TU)
    mean_pre_exp, sigma_pre_exp, err_pre_exp = mean_and_sigma_of_columns_by_name(pre, conf.COL_EXP)
    mean_post_tu, sigma_post_tu, err_post_tu = mean_and_sigma_of_columns_by_name(post, conf.COL_TU)
    mean_post_exp, sigma_post_exp, err_post_exp = mean_and_sigma_of_columns_by_name(post, conf.COL_EXP)

    log.info(f"Averages pre:  TU: {mean_pre_tu}, EXP: {mean_pre_exp}")
    log.info(f"Averages post: TU: {mean_post_tu}, EXP: {mean_post_exp}")

    log.info(f"Sigmas pre:    TU: {sigma_pre_tu}, EXP: {sigma_pre_exp}")
    log.info(f"Sigmas post:   TU: {sigma_post_tu}, EXP: {sigma_post_exp}")

    log.info(f"Err 95% pre:   TU: {err_pre_tu}, EXP: {err_pre_exp}")
    log.info(f"Err 95% post:  TU: {err_post_tu}, EXP: {err_post_exp}")

    means = pd.DataFrame(
        {'pre/post': ['Pre', 'Post'], 'YOU': [mean_pre_tu, mean_post_tu], 'Expert': [mean_pre_exp, mean_post_exp]})
    # err = pd.DataFrame(
    #     {'pre/post': ['Pre', 'Post'], 'YOU': [sigma_pre_tu, sigma_post_tu], 'Expert': [sigma_pre_exp, sigma_post_exp]})
    err = [[err_pre_tu, err_post_tu], [err_pre_exp, err_post_exp]]

    ax = means.plot.bar(x='pre/post', y=['YOU', 'Expert'], rot=0, capsize=4, yerr=err, color=['green', 'red'])
    # Titolo del grafico
    ax.set_title('Overall E-CLASS score on "What do YOU think..."\nand "What do EXPERTS think..." statements')
    # Cancella il titolo dell'asse X (che sarebbe 'pre/post')
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


def chart_what_do_you_think(pre, post, pre2, post2, substring, filename):
    """Creates a chart showing pre/post comparison with statistical significance indicators.

    This function creates a visualization that shows pre and post means for columns containing
    the specified substring, along with Cohen's d effect size and significance markers (stars)
    for statistically significant changes (p < 0.05).

    Args:
        pre (pandas.DataFrame): Original pre-test data
        post (pandas.DataFrame): Original post-test data
        pre2 (pandas.DataFrame): Mapped pre-test data
        post2 (pandas.DataFrame): Mapped post-test data
        substring (str): Substring to filter columns (e.g., 'TU' for student columns)
        filename (str): Path where the chart image will be saved

    Returns:
        None
    """
    columns = [c for c in pre2.columns if column_is_to_be_mapped(c, conf.COL_DONT_MAP_POST) and substring in c]
    pre2_means = {col_name: pre2[col_name].mean() for col_name in columns}
    post2_means = {col_name: post2[col_name].mean() for col_name in columns}
    effect = {col_name: mannwhitneyu(x=pre[col_name], y=post[col_name]).pvalue for col_name in columns}
    print("Effect")
    print(effect)
    cohen = {col_name: abs(cohensd(pre[col_name], post[col_name])) for col_name in columns}
    columns_with_effect = []
    print("Cohens")
    print(cohen)

    # Bisogna azzerare cohen per le colonne che hanno effect > 0.05 (o 0.001)
    for col_name in columns:
        if effect[col_name] > 0.05:
            cohen[col_name] = 0.0
        else:
            columns_with_effect.append(col_name)
    stars = {col_name: max(pre2_means[col_name], post2_means[col_name]) + 0.1 for col_name in columns_with_effect}
    print("Stars")
    print(stars)

    df2 = (pd.DataFrame({
        'Pre': pre2_means,
        'Post': post2_means,
        # 'Effect': effect,
        'Cohen': cohen,
        'Stars': stars})
           .sort_values(by=['Pre']))

    ax1 = df2[['Pre']].plot(zorder=0, marker='s', markersize=8, linestyle='dashed')
    df2[['Post']].plot(ax=ax1, zorder=1, marker='o', markersize=8, linestyle='dashed')
    df2[['Stars']].plot(ax=ax1, marker='*', markersize=10, color='black', zorder=2, legend=False)
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
    ax1.twinx().set_ylabel('Effect Size')
    # Griglia con righe solo verticali
    ax1.xaxis.grid(True, linestyle='dashed', linewidth=0.5)

    fig = ax1.get_figure()
    fig.savefig(filename, dpi=200)


def dump_averages(df):
    """Prints the average values for all mappable columns in the dataframe.

    This function identifies all columns that should be mapped (based on configuration),
    calculates their mean values, and prints them to the console.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to analyze

    Returns:
        None
    """
    columns = [c for c in df.columns if column_is_to_be_mapped(c, conf.COL_DONT_MAP_PRE)]

    pre2_means = {col_name: df[col_name].mean() for col_name in columns}
    for col_name in columns:
        print(f"{col_name}: {pre2_means[col_name]}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    filenames = sys.argv[1:]
    isvalid, message = utils.verify_pre_post_files(*filenames)
    if not isvalid:
        log.error(message)
        exit(1)

    log.info(f"Reading {filenames}")
    pre = read_files(PrePost.PRE, *filenames)

    filenames = list(map(lambda x: x.replace(PrePost.PRE.name, PrePost.POST.name), filenames))
    log.info(f"Reading {filenames}")
    post = read_files(PrePost.POST, *filenames)

    restore_matricola_from_id(pre, post)
    restore_matricola_from_id(post, pre)
    populate_matricola_from_id(pre)
    populate_matricola_from_id(post)

    # TODO se ci sono duplicati, bisogna tranquillamente ignorare quelli che dall'altra parte non ci sono

    # Now, IDs are no more needed
    pre.drop([conf.COL_ID], axis=1, inplace=True)
    post.drop([conf.COL_ID], axis=1, inplace=True)
    pre.to_excel(f"pre.xlsx")
    post.to_excel(f"post.xlsx")

    pre2 = clone_and_map(pre, conf.MAPPING2, conf.COL_DONT_MAP_PRE)
    pre3 = clone_and_map(pre, conf.MAPPING3, conf.COL_DONT_MAP_PRE)
    post2 = clone_and_map(post, conf.MAPPING2, conf.COL_DONT_MAP_POST)
    post3 = clone_and_map(post, conf.MAPPING3, conf.COL_DONT_MAP_POST)
    pre2.to_excel(f"pre2.xlsx")
    pre3.to_excel(f"pre3.xlsx")
    post2.to_excel(f"post2.xlsx")
    post3.to_excel(f"post3.xlsx")

    print("MEDIE di pre2")
    dump_averages(pre2)

    join = join_by_matricola(pre, post)
    join2 = join_by_matricola(pre2, post2)
    join3 = join_by_matricola(pre3, post3)
    join.to_excel(f"join.xlsx")
    join2.to_excel(f"join2.xlsx")
    join3.to_excel(f"join3.xlsx")

    chart_means(pre2, post2, 'chart_means.png')
    chart_what_do_you_think(pre, post, pre3, post3, conf.COL_TU, 'chart_what_do_you_think.png')
