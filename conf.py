COL_CHECK = 'CHECK'
COL_ID = 'Q01_ID-1'
COL_MATRICOLA = 'Q02_Matricola'
COL_TO_DROP = ['Q03_Course-3', 'Response', 'Submitted on:', 'Study Plan', 'Degree Code', 'Course', 'Group', 'ID',
               'Full name', 'Username']
MAX_POINTS = 6
COL_TO_INVERT = ['Q06_3', 'Q07_4', 'Q10_7', 'Q15_12', 'Q19_16', 'Q20_17', 'Q24_21', 'Q29_25', 'Q31_27',
                 'Q32_28', 'Q33_29']

# These are the columns we should not map to shortest values (pre and post are different)
COL_DONT_MAP_PRE = ['Q01', 'Q02', 'Q03', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40']
COL_DONT_MAP_POST = ['Q01', 'Q02', 'Q03', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42']
# This is the mapping to two values for grades 1,2,3,4,5
MAPPING2 = [0, 0, 0, 1, 1]
# This is the mapping to three values for grades 1,2,3,4,5
MAPPING3 = [-1, -1, 0, 1, 1]

# Substrings of the tile of columns that are about yourself
COL_TU = " TU "
# Substrings of the tile of columns that are about experts
COL_EXP = " fisici "
