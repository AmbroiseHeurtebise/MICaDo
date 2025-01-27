# %%
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

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
# Get the list of participants who have a trans file
participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
participants_names = participants['participant_id'].str.replace('sub-', '', regex=False).tolist()
names_trans_available = [
    name for name in participants_names 
    if (TRANS_DIR / f"sub-{name}-trans.fif").exists()]
filtered_participants = participants[
    participants['participant_id'].str.replace('sub-', '', regex=False).isin(
        names_trans_available)]
filtered_participants

# %%
i = 0
subject = names_trans_available[i]
print(f"Subject: {i}:'{subject}'")
trans = TRANS_DIR / f"sub-{subject}-trans.fif"
bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"

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

# Preprocessing
raw.load_data(verbose=False)
raw.filter(l_freq=0.1, h_freq=125, verbose=False)
raw.notch_filter([50, 100], verbose=False)
raw = mne.preprocessing.maxwell_filter(
    raw, calibration=SSS_CAL_FILE, cross_talk=CT_SPARSE_FILE, verbose=False)

# Get events.
sfreq = raw.info["sfreq"]
all_events, all_event_id = mne.events_from_annotations(raw)
metadata_tmin, metadata_tmax = -5., 0
metadata, events, event_id = mne.epochs.make_metadata(
    events=all_events, event_id=all_event_id,
    tmin=metadata_tmin, tmax=metadata_tmax, sfreq=sfreq,
    row_events="button", 
    keep_last="audiovis")

# Create epochs.
tmin = -5
tmax = 5
baseline = (-1.25, -1.0)
epochs = mne.Epochs(
    raw, events, event_id, metadata=metadata,
    tmin=tmin, tmax=tmax,
    baseline=baseline,
    preload=True, verbose=False)

# Keep only the epochs that match the conditions.
epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]

# Evoked (only for vizualization).
evoked = epochs.average()
evoked.plot(spatial_colors=True, gfp=True)

# %%
