import time

# function to tell us how long an epoch took.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
# function to tell us how long an epoch took.
def total_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - (elapsed_hours * 3600))/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_hours, elapsed_mins, elapsed_secs