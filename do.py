import sys
from enum import Enum
from math import sqrt

import numpy as np
import pandas as pd
import logging as log

import conf

log.basicConfig(level=log.INFO)


class PrePost(Enum):
    """This enum specifies if we're reading a PRE or POST file."""
    PRE = 1
    POST = 2


def read_files(pp, *files):
    df = pd.DataFrame()

    for file in files:
        log.info(f"{pp.name}: Reading {file}")
        df = pd.concat([df, pd.read_csv(file, skiprows=0 if file == files[0] else 1)])
    log.info(f"{pp.name}: Read shape {df.shape}")

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
        if column_name in col:
            return True


def populate_matricola_from_id(df):
    """This method populates the empty Matricola with the ID of that row."""
    for index, row in df.iterrows():
        if row[conf.COL_MATRICOLA] == '':
            df.at[index, conf.COL_MATRICOLA] = row[conf.COL_ID]


def restore_matricola(sx, dx):
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


def column_is_to_be_mapped(column_name, veto):
    """Says whether this column values should be remapped.
    >>> column_is_to_be_mapped('Q06_3', ['Q01'])
    True
    >>> column_is_to_be_mapped('Q01_ID-1', ['Q01'])
    False
    >>> column_is_to_be_mapped('Titolo', ['Q01'])
    False
    """
    if not column_name.startswith('Q'):
        return False
    for c in veto:
        if column_name.startswith(c):
            return False
    return True


def clone_and_map(df, mapping, veto):
    copy = df.copy()
    for c in [c for c in copy.columns if column_is_to_be_mapped(c, veto)]:
        # all ID should be upper case
        for index, row in copy.iterrows():
            copy.at[index, c] = mapping[int(row[c]) - 1]
    return copy


def join_by_matricola(sx, dx):
    """This methods joins two DataFrames by the conf.COL_MATRICOLA column."""
    # Joining the two tables
    log.info(f"Joining...")
    log.debug(f"{sx[conf.COL_MATRICOLA]}")
    log.debug(f"{dx[conf.COL_MATRICOLA]}")
    result = pd.merge(sx, dx, on=conf.COL_MATRICOLA, suffixes=("_pre", "_post"))
    log.info(f"Joined: got {result.shape}")
    log.debug(result.head())
    return result


def mean_and_sigma_of_columns_by_name(df, sub_column_name):
    """This method returns the sigmas of all the values of the column that have
     sub_column_name in the name."""
    s = pd.concat([df[col] for col in df.columns if sub_column_name in col])
    return s.mean(), s.std()


def save_means_chart(pre, post, filename):
    mean_pre_tu, sigma_pre_tu = mean_and_sigma_of_columns_by_name(pre, conf.COL_TU)
    mean_pre_exp, sigma_pre_exp = mean_and_sigma_of_columns_by_name(pre, conf.COL_EXP)
    mean_post_tu, sigma_post_tu = mean_and_sigma_of_columns_by_name(post, conf.COL_TU)
    mean_post_exp, sigma_post_exp = mean_and_sigma_of_columns_by_name(post, conf.COL_EXP)
    means_pre = [mean_pre_tu, mean_pre_exp]
    means_post = [mean_post_tu, mean_post_exp]
    log.info(f"Averages pre: {means_pre}")
    log.info(f"Averages post: {means_post}")

    sigma_pre = [sigma_pre_tu, sigma_pre_exp]
    sigma_post = [sigma_post_tu, sigma_post_exp]
    log.info(f"Sigmas pre: {sigma_pre}")
    log.info(f"Sigmas post: {sigma_post}")

    means = pd.DataFrame({'': ['Pre', 'Post'], 'YOU': means_pre, 'Expert': means_post})
    err = pd.DataFrame({'': ['Pre', 'Post'], 'YOU': sigma_pre, 'Expert': sigma_post})
    ax = means.plot.bar(y=['YOU', 'Expert'], rot=0, yerr=err)
    fig = ax.get_figure()
    fig.savefig(filename, dpi=200)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    filenames = sys.argv[1:]
    log.info(f"Reading {filenames}")
    pre = read_files(PrePost.PRE, *filenames)

    filenames = list(map(lambda x: x.replace(PrePost.PRE.name, PrePost.POST.name), filenames))
    log.info(f"Reading {filenames}")
    post = read_files(PrePost.POST, *filenames)

    restore_matricola(pre, post)
    restore_matricola(post, pre)
    populate_matricola_from_id(pre)
    populate_matricola_from_id(post)

    # TODO se ci son duplicati, bisogna tranquillamente ignorare quelli che dall'altra parte non ci sono

    # Now, IDs are no more needed
    pre.drop([conf.COL_ID], axis=1, inplace=True)
    post.drop([conf.COL_ID], axis=1, inplace=True)
    pre.to_excel(f"pre.xlsx")
    post.to_excel(f"post.xlsx")

    pre2 = clone_and_map(pre, conf.MAPPING2, conf.COL_DONT_MAP_PRE)
    pre2.to_excel(f"pre2.xlsx")
    pre3 = clone_and_map(pre, conf.MAPPING3, conf.COL_DONT_MAP_PRE)
    pre3.to_excel(f"pre3.xlsx")
    post2 = clone_and_map(post, conf.MAPPING2, conf.COL_DONT_MAP_POST)
    post2.to_excel(f"post2.xlsx")
    post3 = clone_and_map(post, conf.MAPPING3, conf.COL_DONT_MAP_POST)
    post3.to_excel(f"post3.xlsx")

    join = join_by_matricola(pre, post)
    join.to_excel(f"join.xlsx")
    join2 = join_by_matricola(pre2, post2)
    join2.to_excel(f"join2.xlsx")
    join3 = join_by_matricola(pre3, post3)
    join3.to_excel(f"join3.xlsx")

    save_means_chart(pre, post, 'chart_means.png')
