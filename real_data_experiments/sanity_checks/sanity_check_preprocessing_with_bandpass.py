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
# Get the list of the 354 participants who have a trans file
participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
participants_names = participants['participant_id'].str.replace('sub-', '', regex=False).tolist()
names_trans_available = [
    name for name in participants_names 
    if (TRANS_DIR / f"sub-{name}-trans.fif").exists()]
participants_with_trans = participants[
    participants['participant_id'].str.replace('sub-', '', regex=False).isin(
        names_trans_available)]

# Only keep participants with clean evoked data (156 out of 354)
goods = [
    'CC110033', 'CC120182', 'CC120313', 'CC120376', 'CC120550', 'CC120727',
    'CC120795', 'CC121106', 'CC121111', 'CC121428', 'CC210051',
    'CC210617', 'CC220151', 'CC220352', 'CC220506', 'CC220518', 'CC220843',
    'CC220901', 'CC221031', 'CC221107', 'CC221220', 'CC221324', 'CC221352', 'CC221565',
    'CC222264', 'CC310086', 'CC310129', 'CC310135', 'CC310361', 'CC310400', 'CC310450',
    'CC310473', 'CC320160', 'CC320206', 'CC320218', 'CC320379', 'CC320417', 'CC320478',
    'CC320621', 'CC320680', 'CC320686', 'CC320776', 'CC321025', 'CC321069', 'CC321137',
    'CC321203', 'CC321281', 'CC321464', 'CC321544', 'CC321557', 'CC321594', 'CC321595',
    'CC321880', 'CC321899', 'CC322186', 'CC410040', 'CC410084', 'CC410086', 'CC410091',
    'CC410113', 'CC410119', 'CC410243', 'CC410284', 'CC412004', 'CC420004', 'CC420075',
    'CC420148', 'CC420157', 'CC420162', 'CC420167', 'CC420173', 'CC420182', 'CC420202',
    'CC420231', 'CC420260', 'CC420322', 'CC420324', 'CC420356', 'CC420396', 'CC420402',
    'CC420454', 'CC420589', 'CC420776', 'CC510043', 'CC510050', 'CC510220', 'CC510284',
    'CC510329', 'CC510392', 'CC510438', 'CC510474', 'CC510486', 'CC520011', 'CC520013',
    'CC520055', 'CC520211', 'CC520239', 'CC520247', 'CC520279', 'CC520377', 'CC520390',
    'CC520391', 'CC520503', 'CC520517', 'CC520585', 'CC520597', 'CC520607', 'CC520673',
    'CC520868', 'CC520980', 'CC521040', 'CC610071', 'CC610099', 'CC610212', 'CC610227',
    'CC610285', 'CC610344', 'CC610462', 'CC610625', 'CC610653', 'CC620005', 'CC620026',
    'CC620152', 'CC620193', 'CC620354', 'CC620413', 'CC620429', 'CC620659', 'CC621118',
    'CC621184', 'CC621248', 'CC621284', 'CC710037', 'CC710088', 'CC710342', 'CC710350',
    'CC710382', 'CC710462', 'CC710486', 'CC710664', 'CC720023', 'CC720180', 'CC720188',
    'CC720238', 'CC720290', 'CC720329', 'CC720670', 'CC720941', 'CC721052', 'CC721224',
    'CC721418', 'CC721504', 'CC721519', 'CC721729', 'CC722421', 'CC723395']
filtered_participants = participants_with_trans[
    participants_with_trans['participant_id'].str.replace('sub-', '', regex=False).isin(
        goods)]
filtered_participants

# %%
# Parameters
subject = goods[11]
tmin, tmax = -1.5, 3.
baseline = (-1.5, -1.)
fmin, fmax = 8, 27  # alpha and beta
parcellation = "aparc_sub"
n_crop = 20
go_subtract_evoked = False
go_normalize = True
go_orthogonalize = False
all_labels = True

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
# Get events
sfreq = raw.info["sfreq"]
all_events, all_event_id = mne.events_from_annotations(raw, verbose=False)
metadata, events, event_id = mne.epochs.make_metadata(
    events=all_events, event_id=all_event_id,
    tmin=tmin, tmax=0., sfreq=sfreq,
    row_events="button", keep_last="audiovis")

# Add a column to metadata which counts the number of button events within [tmin, tmax]
button_times = [
    event[0] / sfreq for event in all_events if event[2] == all_event_id["button"]
]
nb_buttons = []
for row_event_time in button_times:
    count = sum(
        1
        for button_time in button_times
        if tmin <= button_time - row_event_time <= tmax
    )
    nb_buttons.append(count)
metadata["nb_buttons"] = nb_buttons

# %%
# Pick grad and mag
raw.pick(picks=["grad", "mag"])
raw.load_data(verbose=False)

# Maxwell filter
raw = mne.preprocessing.maxwell_filter(
    raw, calibration=SSS_CAL_FILE, cross_talk=CT_SPARSE_FILE, verbose=False)
    
# Band-pass filter at alpha or beta waves range
raw.filter(
    fmin,
    fmax,
    n_jobs=4,
    verbose=False,
)

# Get epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    metadata=metadata,
    baseline=baseline,
    reject=dict(grad=4000e-13, mag=4e-12),
    preload=True,
)

# %%
# Remove some epochs
epochs = epochs["audiovis > -1. and nb_buttons == 1"]

# %%
# Plot the envelope at this stage
epochs.copy().apply_hilbert(envelope=True).average().plot()

# %%
# Compute the forward and inverse
trans = TRANS_DIR / f"sub-{subject}-trans.fif"
bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"

src = mne.read_source_spaces(SRC, verbose=False)
fwd = mne.make_forward_solution(raw.info, trans, src, bem, n_jobs=1, verbose=False)
del src
cov = mne.compute_raw_covariance(raw, n_jobs=1, verbose=False)
inv = make_inverse_operator(raw.info, fwd, cov, verbose=False)
del fwd

# %%
# Get the source time courses
stcs = apply_inverse_epochs(
    epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal", return_generator=False,
    verbose=False)

# %%
# Get labels from "fsaverage"
labels = mne.read_labels_from_annot(
    "fsaverage", parc=parcellation, subjects_dir=FREESURFER_DIR, verbose=False)

# Only keep labels that have vertices
filtered_labels = [
    label for label in labels
    if (
        (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
        or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
    )
]

# Print missing labels, if any
missing_labels_src = list(set(labels) - set(filtered_labels))
print(f"Missing labels ({len(missing_labels_src)} out of {len(labels)}):\n{missing_labels_src}")

# %%
# Eventually select a subset of interesting labels
if parcellation == "aparc":
    # 10 already chosen labels
    label_names = [
        'paracentral-lh', 'precentral-lh', 'postcentral-lh', 'transversetemporal-lh', 
        'lateraloccipital-lh', 'paracentral-rh', 'precentral-rh', 'postcentral-rh',
        'transversetemporal-rh', 'lateraloccipital-rh']
elif parcellation == "aparc_sub":
    # 36 already chosen labels
    label_names = [
        'postcentral_3-lh', 'postcentral_5-rh', 'postcentral_6-lh', 'postcentral_6-rh',
        'postcentral_7-lh', 'postcentral_7-rh', 'postcentral_8-lh', 'postcentral_8-rh',
        'postcentral_9-lh', 'postcentral_9-rh', 'precentral_10-lh', 'precentral_10-rh',
        'precentral_11-lh', 'precentral_11-rh', 'precentral_7-lh', 'precentral_7-rh',
        'precentral_8-lh', 'precentral_8-rh', 'precentral_9-lh', 'precentral_9-rh',
        'lateraloccipital_1-lh', 'lateraloccipital_1-rh', 'lateraloccipital_2-lh',
        'lateraloccipital_2-rh', 'lateraloccipital_3-rh', 'pericalcarine_1-lh',
        'pericalcarine_2-lh', 'pericalcarine_2-rh', 'pericalcarine_3-rh', 'pericalcarine_4-rh',
        'transversetemporal_1-lh', 'transversetemporal_1-rh', 'transversetemporal_2-lh',
        'superiortemporal_1-rh', 'superiortemporal_3-lh', 'superiortemporal_4-lh',
        'superiortemporal_5-lh', 'superiortemporal_5-rh',
    ]
selected_labels_total = [label for label in labels if label.name in label_names]
selected_labels = [label for label in filtered_labels if label.name in label_names]

# Print missing labels, if any
missing_labels_subset = list(set(selected_labels_total) - set(selected_labels))
print(f"Missing labels: {missing_labels_subset}")

if not all_labels:
    filtered_labels = selected_labels

# %%
# Compute the average time course across all sources
label_ts = mne.extract_label_time_course(
    stcs, filtered_labels, inv["src"], return_generator=False, verbose=False)
label_ts = np.array(label_ts)
print(f"Shape of label_ts: {label_ts.shape}.")

# %%
# Plot the obtained label time series
plt.subplots(figsize=(10, 4))
label_ts_avg = np.mean(label_ts, axis=0)
plt.plot(stcs[0].times, label_ts_avg.T)
plt.vlines(
    x=0, ymin=np.min(label_ts_avg), ymax=np.max(label_ts_avg),
    linestyles="--", colors="grey")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Average label time series before normalization")

# %%
# Normalize
if go_normalize:
    baseline_bool = (stcs[0].times >= baseline[0]) & (stcs[0].times <= baseline[1])
    baseline_idx = np.arange(label_ts.shape[-1])[baseline_bool]
    baseline_means = np.mean(label_ts[:, :, baseline_idx], axis=-1, keepdims=True)
    baseline_stds = np.std(label_ts[:, :, baseline_idx], axis=-1, keepdims=True)
    label_ts_normalized = (label_ts - baseline_means) / baseline_stds

# %%
# Plot the label time series after normalization
plt.subplots(figsize=(10, 4))
label_ts_avg = np.mean(label_ts_normalized, axis=0)
plt.plot(stcs[0].times, label_ts_avg.T)
plt.vlines(
    x=0, ymin=np.min(label_ts_avg), ymax=np.max(label_ts_avg),
    linestyles="--", colors="grey")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Average label time series after normalization")

# %%
# Orthogonalize
if go_orthogonalize:
    label_ts = mne_connectivity.symmetric_orth(label_ts)

# %%
# Compute envelope
hilbert_ts = hilbert(label_ts_normalized, axis=2)
envelope = np.abs(hilbert_ts)
print(f"Shape of the envelope: {envelope.shape}.")

# %%
# Crop envelope
times = stcs[0].times[n_crop:-n_crop]
envelope_cropped = envelope[:, :, n_crop:-n_crop]

# Plot envelope
envelope_avg = np.mean(envelope_cropped, axis=0)
envelope_avg_centered = envelope_avg - np.mean(envelope_avg, axis=1, keepdims=True)

plt.subplots(figsize=(10, 4))
plt.plot(times, envelope_avg_centered.T)
plt.vlines(
    x=0, ymin=np.min(envelope_avg_centered), ymax=np.max(envelope_avg_centered),
    linestyles="--", colors="black", alpha=0.5)
plt.hlines(y=0, xmin=times[0], xmax=times[-1], linestyles="--", colors="black", alpha=0.5)
plt.xlim([times[0], times[-1]])
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Average envelope")

# %%
# Better vizualization of the label time series
height = 0.5
fig, ax = plt.subplots(figsize=(8, len(filtered_labels) // 3))
for i in range(len(filtered_labels)):
    plt.plot(times, envelope_avg_centered[i] - height * i)
    plt.hlines(y=-height*i, xmin=times[0], xmax=times[-1], linestyles="--", colors="grey")
plt.vlines(x=0, ymin=-len(filtered_labels)*height, ymax=height, linestyles="--", colors="grey")
yticks = np.arange(-len(filtered_labels)+1, 1) * height
yticklabels = [label.name for label in filtered_labels][::-1]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlim([times[0], times[-1]])
ax.set_ylim([-height*len(filtered_labels), height])
plt.title("Mean (over epochs) label time series")
plt.xlabel("Time (s)")
plt.ylabel("Mean amplitude")
plt.show()

# %%
# Print labels with the highest variance
nb_labels_high = 30
indices = np.argsort(np.var(envelope_avg, axis=1))[::-1][:nb_labels_high]
activated_labels = [filtered_labels[i].name for i in indices]
activated_labels

# %%
