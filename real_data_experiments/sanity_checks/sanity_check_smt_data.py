# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import warnings
import mne
import mne_connectivity
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator


# %%
# Paths
DATA_DIR = Path("/storage/store/data")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"
FREESURFER_DIR = DATA_DIR / "camcan-mne/freesurfer"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
SRC = FREESURFER_DIR / "fsaverage/bem/fsaverage-ico-5-src.fif"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"

# %%
# Only keep a subset of participants
# Get the list of the 354 participants who have a trans file
participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
participants_names = participants['participant_id'].str.replace('sub-', '', regex=False).tolist()
names_trans_available = [
    name for name in participants_names 
    if (TRANS_DIR / f"sub-{name}-trans.fif").exists()]
participants_with_trans = participants[
    participants['participant_id'].str.replace('sub-', '', regex=False).isin(
        names_trans_available)]

# Only keep participants with clean evoked data (40 out of the first 107)
goods = [
    'CC110033', 'CC120120', 'CC120182', 'CC120313', 'CC120376', 'CC120550', 'CC120727',
    'CC120795', 'CC121106', 'CC121111', 'CC121144', 'CC121428', 'CC210051', 'CC210124',
    'CC210519', 'CC210617', 'CC220151', 'CC220352', 'CC220506', 'CC220518', 'CC220843',
    'CC220901', 'CC221031', 'CC221107', 'CC221220', 'CC221324', 'CC221352', 'CC221565',
    'CC222264', 'CC310086', 'CC310129', 'CC310135', 'CC310361', 'CC310400', 'CC310450',
    'CC310473', 'CC320160', 'CC320206', 'CC320218', 'CC320379']
filtered_participants = participants_with_trans[
    participants_with_trans['participant_id'].str.replace('sub-', '', regex=False).isin(
        goods)]
filtered_participants.head()

# %%
# Parameters
subject = goods[0]
n_batches = 10  # used to batch-average epochs
sfreq_envelope = 10  # used to downsample envelope across timepoints dimension
parcellation = "aparc"
moving_avg = True
metadata_tmin, metadata_tmax = -5., 0
tmin, tmax = -5, 5
baseline = (-1.25, -1.0)

# %%
trans = TRANS_DIR / f"sub-{subject}-trans.fif"
bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"

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
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Unable to map the following column.*")
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
# Question: what values should we choose for the baseline? 
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
fwd = mne.make_forward_solution(raw.info, trans, src, bem, n_jobs=1, verbose=False)
del src
cov = mne.compute_raw_covariance(raw, n_jobs=1, verbose=False)
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
# Get labels from "fsaverage"
labels = mne.read_labels_from_annot(
    "fsaverage", parc=parcellation, subjects_dir=FREESURFER_DIR, verbose=False)

# For some participants, there are a few labels that have no vertex in the 
# source space inv["src"]. Indeed, inv["src"] has less vertices than src.
# So, we remove the empty labels. This is especially true for "aparc_sub", 
# but it also allows to remove 'unknown-lh' from "aparc".
filtered_labels = [
    label for label in labels
    if (
        (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
        or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
    )
]

# Print missing labels, if any
missing_labels_src = list(set(labels) - set(filtered_labels))
print(f"Missing labels: {missing_labels_src}")

# Select some interesting labels
if parcellation == "aparc":
    # 10 already chosen labels
    label_names = [
        'precentral-lh', 'postcentral-lh', 'parsopercularis-lh', 'transversetemporal-lh',
        'pericalcarine-lh', 'precentral-rh', 'postcentral-rh', 'parsopercularis-rh',
        'transversetemporal-rh', 'pericalcarine-rh']
elif parcellation == "aparc_sub":
    # 18 already chosen labels
    label_names = [
        'precentral_1-lh', 'precentral_2-lh', 'postcentral_1-lh', 'postcentral_2-lh',
        'paracentral_1-lh', 'superiorfrontal_1-lh', 'transversetemporal_1-lh',
        'lateraloccipital_1-lh', 'frontalpole_1-lh', 'precentral_1-rh', 'precentral_2-rh',
        'postcentral_1-rh', 'postcentral_2-rh', 'paracentral_1-rh', 'superiorfrontal_1-rh',
        'transversetemporal_1-rh', 'lateraloccipital_1-rh', 'frontalpole_1-rh']
    # alternative list of 20 labels
    # label_names = ['precentral_1-lh', 'precentral_2-lh', 'precentral_3-lh', 'precentral_4-lh',
    #       'postcentral_1-lh', 'postcentral_2-lh', 'postcentral_3-lh', 'postcentral_4-lh',
    #       'superiorfrontal_1-lh', 'superiorfrontal_2-lh', 'precentral_1-rh', 'precentral_2-rh',
    #       'precentral_3-rh', 'precentral_4-rh', 'postcentral_1-rh', 'postcentral_2-rh',
    #       'postcentral_3-rh', 'postcentral_4-rh', 'superiorfrontal_1-rh', 'superiorfrontal_2-rh']

selected_labels_total = [label for label in labels if label.name in label_names]
selected_labels = [label for label in filtered_labels if label.name in label_names]

# Print missing labels, if any
missing_labels_subset = list(set(selected_labels_total) - set(selected_labels))
print(f"Missing labels: {missing_labels_subset}")

# %%
# Compute the average time course across all sources (dipoles) 
# that belong to each label (region of interest, ROI),
# so there is one time series per epoch and per label.
label_ts = mne.extract_label_time_course(
    stcs, selected_labels, inv["src"], return_generator=False, verbose=False)
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
# We can eventually orthogonalize the time series.
# Question: should we?
go = False
if go:
    label_ts = mne_connectivity.envelope.symmetric_orth(label_ts)

# %%
# Band-pass ``label_ts`` at the beta waves range, i.e. between 14 Hz and 30 Hz. 
# However, it removes the peak.
go = False
if go:
    label_ts_filtered = mne.filter.filter_data(label_ts, sfreq, 14, 30, verbose=False)
    print(f"Shape of the label_ts_filtered: {np.array(label_ts_filtered).shape}.")

# %%
# Compute the envelope.
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

# Downsample timepoints.
factor = int(sfreq // sfreq_envelope)  # window size
if moving_avg:
    kernel = np.ones(factor) / factor
    smoothed_envelope = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='valid'), axis=2, arr=envelope_batch_avg)
    envelope_reduced = smoothed_envelope[:, :, ::factor]
else:
    envelope_reduced = envelope_batch_avg[:, :, ::factor]
print(f"Shape of envelope_reduced: {envelope_reduced.shape}.")

# Concatenate batches.
n_labels = len(selected_labels)
envelope_concat = envelope_reduced.swapaxes(0, 1).reshape(n_labels, -1)
print(f"Shape of envelope_concat: {envelope_concat.shape}.")

# %%
# Vizualize obtained envelopes.
# The figure is decomposed into n_batches intervals.
# Each batch contains time series from tmin=-5 to tmax=5.
plot_avg = True
n = envelope_concat.shape[1] // n_batches
if plot_avg:
    plt.subplots(figsize=(10, 8))
    envelope_avg = envelope_concat[:, :n].copy()
    for i in range(1, n_batches):
        envelope_avg += envelope_concat[:, n*i:n*(i+1)]
    envelope_avg /= n_batches
    times = np.linspace(tmin, tmax, n)
    for i in range(len(selected_labels)):
        plt.plot(times, envelope_avg[i], label=f"{selected_labels[i].name}")
    plt.vlines(
        x=0, ymin=np.min(envelope_avg), ymax=np.max(envelope_avg),
        linestyles="--", colors="grey")
    plt.legend()
    plt.xlabel("Time (s)")
else:
    plt.subplots(figsize=(12, 4))
    for i in range(len(selected_labels)):
        plt.plot(envelope_concat[i], label=f"{selected_labels[i].name}")
    plt.legend()
    tick_positions = n * np.arange(n_batches) + n // 2
    tick_labels = np.arange(n_batches)
    plt.xticks(tick_positions, tick_labels)
    ymin, ymax = np.min(envelope_concat), np.max(envelope_concat)
    for i in range(n_batches+1):
        plt.vlines(x=n*i, ymin=ymin, ymax=ymax, linestyles="--", colors="black")
    plt.xlabel("Batch")
plt.title("Final data")
plt.ylabel("Mean amplitude")
plt.show()
# Question: do you think that we can use these data?

# %%
