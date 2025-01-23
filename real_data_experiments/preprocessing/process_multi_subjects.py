import numpy as np
import pickle
from utils import get_participants, process_data_one_subject
import os


# I'm not sure if the following lines limit the number of jobs when using the functions
# apply_inverse_epochs and make_inverse_operator from mne.minimum_norm
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Parameters
n_subjects = 40
n_batches = 10
sfreq_envelope = 10
metadata_tmin, metadata_tmax = -5., 0. 
baseline = (-1.25, -1.0)
tmin, tmax = -5, 5
n_crop_edges = 5
moving_avg = True
parcellation = "aparc"

# Get subjects
participants = get_participants()
subjects = participants[
    'participant_id'].head(n_subjects).str.replace('sub-', '', regex=False).tolist()

# Get data
X = []
for i, subject in enumerate(subjects):
    print(f"Starting processing of subject {i}")
    envelope, labels = process_data_one_subject(
        subject,
        n_batches=n_batches,
        sfreq_envelope=sfreq_envelope,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        metadata_tmin=metadata_tmin,
        metadata_tmax=metadata_tmax,
        n_crop_edges=n_crop_edges,
        moving_avg=moving_avg,
        parcellation=parcellation,
    )
    X.append(envelope)
X = np.array(X)

# Save data
save_dir = "/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments/data_envelopes/"
np.save(save_dir + f"X_{parcellation}_{n_subjects}_subjects.npy", X)
with open(save_dir + f"labels_{parcellation}_{n_subjects}_subjects.pkl", "wb") as f:
    pickle.dump(labels, f)
