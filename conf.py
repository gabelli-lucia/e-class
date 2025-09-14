# The default delimiter for CSV files
DELIMITER=","

COL_CHECK = 'CHECK'
COL_ID = 'ID-1'
COL_MATRICOLA = 'Matricola'
# Complete names of the columns that we need to drop from all files.
COL_TO_DROP = ['Response', 'Submitted on:', 'Study Plan', 'Degree Code', 'Course', 'Group', 'ID',
               'Full name', 'Username', 'Course-3']
MAX_POINTS = 6
# Substrings of columns name that we need to invert (their value well be mapped to MAX_POINTS - value).
COL_TO_INVERT = ['3', '4', '7', '12', '16', '17', '21', '25', '27', '28', '29']

# These are the columns we should not map to shortest values
COL_DONT_MAP = ['Gender', 'Equity', 'Background']
# This is the mapping to two values for grades 1,2,3,4,5
MAPPING2 = [0, 0, 0, 1, 1]
# This is the mapping to three values for grades 1,2,3,4,5
MAPPING3 = [-1, -1, 0, 1, 1]
# Threshold of the effect size.
PVALUE_THRESHOLD = 0.05

# Substrings of the tile of columns that are about yourself
COL_TU = " TU "
# Substrings of the tile of columns that are about experts
COL_EXP = " fisici "
# Substring of the tile of columns that are about success.
COL_SUCCESS = " importante "

# This is the full list of the questions.
Q = [
    "When doing an experiment, I try to understand how the experimental setup works.",
    "If I wanted to, I think I could be good at doing research.",
    "When doing a physics experiment, I don't think much about sources of systematic error.",
    "If I am communicating results from an experiment, my main goal is to have the correct sections and formatting.",
    "Calculating uncertainties usually helps me understand my results better.",
    "Scientific journal articles are helpful for answering my own questions and designing experiments",
    "I don't enjoy doing physics experiments.",
    "When doing an experiment, I try to understand the relevant equations.",
    "When I approach a new piece of lab equipment, I feel confident I can learn how to use it well enough for my purposes.",
    "Whenever I use a new measurement tool, I try to understand its performance limitations.",
    "Computers are helpful for plotting and analyzing data.",
    "I don't need to understand how the measurement tools and sensors work in order to carry out an experiment.",
    "If I try hard enough, I can succeed at doing physics experiments.",
    "When doing an experiment, I usually think up my own questions to investigate.",
    "Designing and building things is an important part of doing physics experiments.",
    "The primary purpose of doing a physics experiment is to confirm previously known results.",
    "When I encounter difficulties in the lab, my first step is to ask an expert, like the instructor.",
    "Communicating scientific results to peers is a valuable part of doing physics experiments.",
    "Working in a group is an important part of doing physics experiments.",
    "I enjoy building things and working with my hands.",
    "I am usually able to complete an experiment without understanding the equations and physics ideas that describe the system I am investigating.",
    "If I am communicating results from an experiment, my main goal is to make conclusions based on my data using scientific reasoning.",
    "When I am doing an experiment, I try to make predictions to see if my results are reasonable.",
    "Nearly all students are capable of doing a physics experiment if they work at it.",
    "A common approach for fixing a problem with an experiment is to randomly change things until the problem goes away.",
    "It is helpful to understand the assumptions that go into making predictions.",
    "When doing an experiment, I just follow the instructions without thinking about their purpose.",
    "I do not expect doing an experiment to help my understanding of physics.",
    "If I don't have clear directions for analyzing data, I am not sure how to choose an appropriate analysis method.",
    "Physics experiments contribute to the growth of scientific knowledge."
]
