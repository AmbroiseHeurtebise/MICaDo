import pandas as pd
from utils import get_participants, process_data_one_subject, plot_envelope


# Parameters
subject_idx = 0
n_batches = 10  # used to batch-average epochs
sfreq_envelope = 10  # used to downsample envelope across timepoints dimension
metadata_tmin, metadata_tmax = -5., 0. 
baseline = (-1.25, -1.0)
tmin, tmax = -5, 5
n_crop_edges = 5
moving_avg = True
parcellation = "aparc"

# Get DataFrame of participants
participants = get_participants()
subject = participants.iloc[subject_idx]['participant_id'].replace('sub-', '')

# Get envelope of shape (n_labels, n_batches*(tmax-tmin)*sfreq_envelope)
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

# Plot
plot_envelope(
    envelope,
    n_batches=n_batches,
    plot_avg=False,
    labels=labels,
    save=True,
    subject=subject,
)
