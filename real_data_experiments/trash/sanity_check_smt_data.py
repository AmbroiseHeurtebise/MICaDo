# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import pickle
import mne
import mne_connectivity
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator

# %%
# Parameters
subject = "CC110033"
n_batches = 4  # used to batch-average epochs
sfreq_envelope = 10  # used to downsample envelope across timepoints dimension

# %%
# Paths
DATA_DIR = Path("/storage/store/data")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"
FREESURFER_DIR = DATA_DIR / "camcan-mne/freesurfer"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
# The file "fsaverage-ico-5-src.fif" has a better definition than "fsaverage-oct-5-src.fif"
SRC = FREESURFER_DIR / "fsaverage/bem/fsaverage-ico-5-src.fif"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"
trans = TRANS_DIR / f"sub-{subject}-trans.fif"
bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"

# %%
# Check that the subject has a trans file.
# There are 356 subjects with a trans file at this path, out of the 643 subjects in total.
if not trans.exists():
    print("The ``trans`` file for this subject does not exist at this path. \
        Choose another subject.")

# %%
# DataFrame of all participants
participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
participants

# %%
# Read raw data of one participant
bp = BIDSPath(
    root=BIDS_ROOT,
    subject=subject,
    task="smt",
    datatype="meg",
    extension=".fif",
    session="smt",
)   
raw = read_raw_bids(bp, verbose=False)

# %%
# Some info
print(f"First sample: {raw.first_samp}, Last sample: {raw.last_samp}")
print(f"Recording duration (seconds): {raw.times[-1]}")
sfreq = raw.info["sfreq"]
print(f"Sampling frequency: {sfreq} Hz.")

# %%
# In the MNE file that you sent, Alex, the authors used the following preprocessing.
# (See https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_envelope_correlation.html).
# I'm not sure that we necessarily have to use the same preprocessing, but I left this code snippet anyway.
go = False
if go:
    raw.crop(0, 60).pick_types(meg=True, eeg=False).load_data().resample(80)
    # I don't know how to make the following method work, so I commented it
    # raw.apply_gradient_compensation(3)
    projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
    eog_channels = mne.pick_types(raw.info, eog=True)
    eog_ch_names = [raw.ch_names[idx] for idx in eog_channels]
    projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name=eog_ch_names)
    raw.add_proj(projs_ecg + projs_eog)
    raw.apply_proj()
    # next line helps with symmetric orthogonalization, if we decide to use it later
    raw.filter(0.1, None)

# %%
# Here is the preprocessing used in Cédric's paper about CDL on sensorimotor task.
# I think we can use the same. 
go = True
if go:
    # Question: should we pick only gradiometers or magnetometers, or keep both?
    raw.load_data(verbose=False)
    # Question: Why using a low-pass filter at 125 Hz? In DriPP paper, they say it's to remove slow drifts, 
    # but I thought that it was the role of high-pass filters...
    # raw.filter(l_freq=None, h_freq=125)
    # I replaced l_freq=None by l_freq=0.1, as it removed the slow drifts.
    raw.filter(l_freq=0.1, h_freq=125, verbose=False)
    # Question: Why using a notch-filter? And why attenuating frequencies between 50 and 100 Hz, specifically?
    # Answer: to remove power line noise and its harmonics.
    raw.notch_filter([50, 100])
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=SSS_CAL_FILE, cross_talk=CT_SPARSE_FILE, verbose=False)
    # Question: we didn't remove ECG and EOG (probably because Cédric wanted their atoms), 
    # but maybe we should?

# %%
# Remove ECG and EOG artifacts.
# In practice, it does not work well but I don't know why.
go = False
if go:
    projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
    eog_channels = mne.pick_types(raw.info, eog=True)
    eog_ch_names = [raw.ch_names[idx] for idx in eog_channels]
    projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name=eog_ch_names)
    raw.add_proj(projs_ecg + projs_eog)
    raw.apply_proj()

# %%
# Get events.
# We set metadata_tmin=-5 to remove button press events that occur within 5s of the previous event.
# Remark: Cédric used 3s, not 5s, but we want longer epochs.
# We set keep_last="audiovis" to remove button press events that occur more than 1s after the audiovis stimulus.
all_events, all_event_id = mne.events_from_annotations(raw)
metadata_tmin, metadata_tmax = -5., 0
metadata, events, event_id = mne.epochs.make_metadata(
    events=all_events, event_id=all_event_id,
    tmin=metadata_tmin, tmax=metadata_tmax, sfreq=sfreq,
    row_events="button", 
    keep_last="audiovis")

print(f"Time of the first event: {events[0, 0] / 1000}s.")
print(f"Time of the last event: {events[-1, 0] / 1000}s.")

# %%
# Create epochs.
# Cédric used tmin=-1.7 and tmax=1.7 but we need longer data.
# Remark: Aapo suggested using epochs of length 10s or even 20s. 
# Question: does the values tmin=-5 and tmax=5 seem good to you?
tmin = -5
tmax = 5
# Question: what values should we choose for the baseline? 
baseline = (-1.25, -1.0)
epochs = mne.Epochs(
    raw, events, event_id, metadata=metadata,
    tmin=tmin, tmax=tmax,
    baseline=baseline,
    preload=True, verbose=False)

# %%
# Keep only the epochs that match the conditions.
epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]
print(f"Number of epochs: {len(epochs)}.")

# %%
# Evoked (only for vizualization).
evoked = epochs.average()
# On my laptop, it plots 2 times the same figure (I don't know why).
# evoked.plot(picks="mag", spatial_colors=True, gfp=True)
evoked.plot(spatial_colors=True, gfp=True)

# %%
# Compute the forward and inverse.
# Question: we acquired the source space ``src`` from the average brain (fsaverage), 
# whereas the transformer file ``trans`` and BEM file ``bem`` are subject-specific.
# Is it a problem?
# Remark: we could also use ``trans`` and ``bem`` files from fsaverage., WDYT?
src = mne.read_source_spaces(SRC, verbose=False)
fwd = mne.make_forward_solution(raw.info, trans, src, bem, verbose=False)
# del src
cov = mne.compute_raw_covariance(raw)
inv = make_inverse_operator(raw.info, fwd, cov, verbose=False)
del fwd

# %%
# Get the source time courses (inside the cortex; each source corresponds to a dipole).
# Question: I used the same parameters ``lambda2`` and ``pick_ori`` as in the example you sent,
# but are they the best choices (especially lambda2)?
stcs = apply_inverse_epochs(
    epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal", return_generator=False,
    verbose=False)

# %%
# Vizualize mean source time course.
# I guess that the figure does not look like neural activity 
# because only a few out of the 1618 sources represent neural activity, 
# so we mainly average noise sources.
stcs_np = np.array([stc.data for stc in stcs])  # shape (32, 1618, 10001)
mean_time_course = np.mean(stcs_np, axis=(0, 1))
plt.plot(stcs[0].times, mean_time_course)
plt.xlabel("Time (s)")
plt.ylabel("Mean amplitude")
plt.title("Mean (over epochs and sources) Time Course")
plt.show()

# %%
# Get labels.
# Alex said that we should use the aparc_sub parcellation.
parcellation = "aparc_sub"
if parcellation == "aparc":
    # Get aparc labels from subject.
    labels = mne.read_labels_from_annot(
        subject, "aparc", subjects_dir=FREESURFER_DIR)
    # Unfortunately, because ``src`` comes from fsaverage, 
    # we have to morph subject-specific labels to fsaverage.
    morphed_labels = mne.morph_labels(
        labels, subject_to="fsaverage", subject_from=subject,
        subjects_dir=FREESURFER_DIR)
elif parcellation == "aparc_sub":
    # Get aparc_sub labels from fsaverage.
    labels = mne.read_labels_from_annot(
        "fsaverage", "aparc_sub", subjects_dir=FREESURFER_DIR, verbose=False)
    # Among the 450 labels, 36 do not have vertices in the source space inv["src"].
    # Indeed, src has 10242 vertices in each hemisphere, whereas inv["src"] only has 
    # 8428 vertices in the left hemisphere and 7819 in the right hemisphere.
    # In the following, we remove these 36 labels, so that 414 labels remain.
    filtered_labels = [
        label for label in labels
        if (
            (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
            or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
        )
    ]

# %%
# Compute the average time course across all sources (dipoles) 
# that belong to each label (region of interest, ROI),
# so there is one time series per epoch and per label.
label_ts = mne.extract_label_time_course(
    stcs, filtered_labels, inv["src"], return_generator=False, verbose=False)
print(f"Shape of label_ts: {np.array(label_ts).shape}.")

# %%
# Vizualize label time series.
# We averaged over epochs, so there is one curve per label in the parcellation.
label_ts_avg = np.mean(label_ts, axis=0)
plt.plot(stcs[0].times, label_ts_avg.T)
plt.title("Mean (over epochs) label time series")
plt.xlabel("Time (s)")
plt.ylabel("Mean amplitude")
plt.show()

# %%
# Compute label-specific variance.
# The general idea is that labels related to neural activity 
# should have a higher variance.
var = np.mean(np.var(label_ts, axis=2), axis=0)  # shape (n_labels,)
label_idx_good = np.argsort(var)[::-1][:20]
labels_high_variance = [labels[id] for id in label_idx_good]

# %%
# Save the list of most important labels.
with open("labels_high_variance.pkl", "wb") as f:
    pickle.dump(labels_high_variance, f)

# %%
# We can eventually orthogonalize the time series.
# Question: should we?
go = False
if go:
    label_ts = mne_connectivity.envelope.symmetric_orth(label_ts)

# %%
# Band-pass ``label_ts`` at the beta waves range, i.e. between 14 Hz and 30 Hz. 
# However, it removes the peak.
label_ts_filtered = mne.filter.filter_data(label_ts, sfreq, 14, 30, verbose=False)
print(f"Shape of the label_ts_filtered: {np.array(label_ts_filtered).shape}.")

# %%
label_ts_avg = np.mean(label_ts, axis=0)
plt.plot(stcs[0].times, label_ts_avg.T)
plt.title("Mean (over epochs) label time series")
plt.xlabel("Time (s)")
plt.ylabel("Mean amplitude")
plt.show()

# %%
# Only select a subset of labels.
# We should select labels now to avoid unuseful computations.
label_ts_subset = [ts[label_idx_good] for ts in label_ts]
hilbert_ts_subset = hilbert(label_ts_subset, axis=2)
envelope_subset = np.abs(hilbert_ts_subset)

# %%
# Compute the envelope.
# Question: in the example you sent, they band-pass ``label_ts`` on the fly at the beta waves range, 
# i.e. between 14 Hz and 30 Hz. Do we need a similar filter?
hilbert_ts = hilbert(label_ts, axis=2)
envelope = np.abs(hilbert_ts)
print(f"Shape of the envelope: {envelope.shape}.")

# %%
# Crop.
n_removed = 5
envelope_cropped = envelope[:, :, n_removed:-n_removed]
print(f"Shape of the envelope_cropped: {envelope_cropped.shape}.")

# %%
# Vizualize the average (over epochs) envelope (one curve per label in the parcellation).
# Question: is it normal for the figure to look like this,
# especially with peaks at the edges of the figure?
envelope_avg = np.mean(envelope_cropped, axis=0)
times = stcs[0].times[n_removed:-n_removed]
plt.plot(times, envelope_avg.T)
plt.title("Mean envelope time series (over epochs)")
plt.xlabel("Time (s)")
plt.ylabel("Mean amplitude")
plt.show()

# %%
# Downsample data.
# The envelope has shape (n_epochs, n_labels, n_timepoints).
# We should reduce the numbers of epochs and time points, 
# and maybe also reduce the number of labels, 
# as 68 components will probably be too much for ShICA-ML.
# Question: if we want to reduce the number of labels, which ones do we choose?
# Is it okay to select some of them visually like I did?
# Question: to downsample the timepoints, should we only keep one timepoint 
# every ``factor`` timepoints, or should we use a moving average?
def batch_average(envelope, n_batches):
    n_epochs, n_labels, n_timepoints = envelope.shape
    batch_size = n_epochs // n_batches
    remainder = n_epochs % n_batches
    results = []
    start = 0
    for i in range(n_batches):
        extra = 1 if i < remainder else 0
        end = start + batch_size + extra
        results.append(np.mean(envelope[start:end], axis=0))
        start = end
    return np.array(results)

print(f"Shape of envelope_cropped: {envelope_cropped.shape}.")
# Average by batch.
envelope_batch_avg = batch_average(envelope_cropped, n_batches=n_batches)
print(f"Shape of envelope_batch_avg: {envelope_batch_avg.shape}.")

# Concatenate batches.
n_labels = envelope_batch_avg.shape[1]
envelope_concat = envelope_batch_avg.swapaxes(0, 1).reshape(n_labels, -1)
print(f"Shape of envelope_concat: {envelope_concat.shape}.")

# Downsample timepoints.
# maybe more a moving average
factor = int(sfreq // sfreq_envelope)
envelope_reduced = envelope_concat[:, ::factor]
print(f"Shape of envelope_reduced: {envelope_reduced.shape}.")

# Only keep the best labels.
envelope_subset = envelope_reduced[label_idx_good]
print(f"Shape of envelope_subset: {envelope_subset.shape}.")

# %%
# Vizualize obtained envelopes.
# The figure is decomposed into n_batches=4 batches.
# Each batch contains time series from tmin=-5 to tmax=5.
# I don't like the results because I expected a greater peak 
# in the middle of each batch.
plt.plot(envelope_subset.T)
plt.title("Final data")
batch_length = envelope_subset.shape[1] // n_batches
tick_positions = batch_length * np.arange(n_batches) + batch_length // 2
tick_labels = np.arange(n_batches)
plt.xticks(tick_positions, tick_labels)
ymin, ymax = np.min(envelope_subset), np.max(envelope_subset)
for i in range(n_batches+1):
    plt.vlines(x=batch_length*i, ymin=ymin, ymax=ymax, linestyles="--", colors="black")
plt.xlabel("Batch")
plt.ylabel("Mean amplitude")
plt.show()
# Question: do you think that we can use these data?

# %%
