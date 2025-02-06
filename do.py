import sys
from enum import Enum

import pandas as pd
import logging as log

COL_CHECK = 'CHECK'
COL_ID = 'Q01_ID-1'
COL_MATRICOLA = 'Q02_Matricola'
COL_TO_DROP = ['Response', 'Submitted on:', 'Study Plan', 'Degree Code', 'Course', 'Group', 'ID', 'Full name',
               'Username']

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
    df.drop(COL_TO_DROP, axis=1, inplace=True)

    # Identifying the CHECK columns
    check_columns = [col for col in df.columns if COL_CHECK in col]
    log.debug(f"{pp.name}: CHECK columns {check_columns}")

    # Removing rows where people did not answer 4 in all the CHECK columns
    for check in check_columns:
        df = df[df[check] == 4]
    # Removing the CHECK columns
    df.drop(check_columns, axis=1, inplace=True)
    log.info(f"{pp.name}: Passed CHECK shape {df.shape}")

    # log.info(f.head())
    # Removing invalid Matricola
    df[COL_MATRICOLA] = df[COL_MATRICOLA].map(lambda x: str(x))
    df[COL_MATRICOLA] = df[COL_MATRICOLA].map(lambda x: x.replace('.0', ''))
    df[COL_MATRICOLA] = df[COL_MATRICOLA].map(lambda x: x if x.isnumeric() else '')
    df[COL_MATRICOLA] = df[COL_MATRICOLA].map(lambda x: '' if x != '' and int(x) < 1000000 else x)
    df[COL_MATRICOLA] = df[COL_MATRICOLA].map(lambda x: '' if x != '' and int(x) > 3000000 else x)

    # all ID should be upper case
    for index, row in df.iterrows():
        df.at[index, COL_ID] = row[COL_ID].upper()

    log.debug(f"{df.dtypes}")
    return df


def populate_matricola_from_id(df):
    """This method populates the empty Matricola with the ID of that row."""
    for index, row in df.iterrows():
        if row[COL_MATRICOLA] == '':
            df.at[index, COL_MATRICOLA] = row[COL_ID]


def restore_matricola(sx, dx):
    """This method populates empty Matricola in sx looking up the Matricola in sx
    with the same ID."""
    log.debug(f"Restoring Matricola")
    # for each Matricola in sx
    for index, sx_matricola in sx[COL_MATRICOLA].items():
        # if the Matricola is missing
        if sx_matricola == '' or sx_matricola is None:
            log.debug(f"Matricola not found at index {index}, trying to restore it")
            # The ID for this Matricola
            sx_id = sx.at[index, COL_ID]
            log.debug(f" Matricola at index {index} has ID {sx_id}")
            if sx_id is not None and sx_id != '':
                dx_row = dx.loc[dx[COL_ID] == sx_id]
                if not dx_row.empty:
                    log.debug(f" DX has ID {sx_id} at row {dx_row.index}")
                    dx_matricola = dx_row.iloc[0].at[COL_MATRICOLA]
                    if dx_matricola is not None and dx_matricola != '':
                        log.debug(f" ID {sx_id} has Matricola {dx_matricola}")
                        sx.at[index, COL_MATRICOLA] = dx_matricola


if __name__ == "__main__":
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

    log.info(f"Joining...")
    log.debug(f"{pre[COL_MATRICOLA]}")
    log.debug(f"{post[COL_MATRICOLA]}")
    join = pd.merge(pre, post, on=COL_MATRICOLA, suffixes=("_pre", "_post"))
    log.info(f"Joined: got {join.shape}")
    log.debug(join.head())
