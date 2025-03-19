
# ========= DATA-RELEATED =========
# data location
# data_dir = './data'
data_dir = '/home/hygo/Development/recommendations/data'

# data description
timeid = 'timestamp'
userid = 'userid'

# data preprocessing
sequence_length_movies = 200

# data splits
time_offset_q = 0.95
max_test_interactions = 50_000


# ========= MODEL-RELEATED =========
validation_interval = 4 # frequency of validation for iterative models; 1 means validate on each iteration
