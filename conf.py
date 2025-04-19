COL_CHECK = 'CHECK'
COL_ID = 'ID-1'
COL_MATRICOLA = 'Matricola'
# Complete names of the columns that we need to drop from all files.
COL_TO_DROP = ['Response', 'Submitted on:', 'Study Plan', 'Degree Code', 'Course', 'Group', 'ID',
               'Full name', 'Username', 'Course-3']
MAX_POINTS = 6
# Substrings of columns name that we need to invert (their value well be mapped to MAX_POINTS - value).
COL_TO_INVERT = ['3', '4', '7', '12', '16', '17', '21', '25', '27', '28', '29']

# These are the columns we should not map to shortest values (pre and post may be different)
COL_DONT_MAP_PRE =  ['Gender', 'Equity', 'Background']
COL_DONT_MAP_POST = ['Gender', 'Equity', 'Background']
# This is the mapping to two values for grades 1,2,3,4,5
MAPPING2 = [0, 0, 0, 1, 1]
# This is the mapping to three values for grades 1,2,3,4,5
MAPPING3 = [-1, -1, 0, 1, 1]

# Substrings of the tile of columns that are about yourself
COL_TU = " TU "
# Substrings of the tile of columns that are about experts
COL_EXP = " fisici "
