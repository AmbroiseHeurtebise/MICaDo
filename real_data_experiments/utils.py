from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator


# Global variables
DATA_DIR = Path("/storage/store/data")
BIDS_ROOT = DATA_DIR / "camcan/BIDSsep/smt/"
FREESURFER_DIR = DATA_DIR / "camcan-mne/freesurfer"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
SSS_CAL_FILE = DATA_DIR / "camcan-mne/Cam-CAN_sss_cal.dat"
CT_SPARSE_FILE = DATA_DIR / "camcan-mne/Cam-CAN_ct_sparse.fif"
SRC = FREESURFER_DIR / "fsaverage/bem/fsaverage-ico-5-src.fif"
TRANS_DIR = DATA_DIR / "camcan-mne/trans"


def process_data_one_subject(
    subject,
    n_batches=1,
    sfreq_envelope=None,
    tmin=-5.,
    tmax=5.,
    baseline=(-1.25, -1.0),
    metadata_tmin=-5.,
    metadata_tmax=0.,
    n_crop_edges=None,
    band_pass_beta=False,
    moving_avg=True,
    parcellation="aparc",
    verbose=False,
):
    """Process data of one subject.
    """
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
        raw = read_raw_bids(bp, verbose=verbose)
    
    # Load data in memory to speed up computations
    raw.load_data(verbose=verbose)
    # High-pass at 0.1 Hz to remove slow drifts and 
    # low-pass at 125 Hz because it proves useful
    raw.filter(l_freq=0.1, h_freq=125, verbose=verbose)
    # Notch-filter at 50 Hz (and its harmonic 100 Hz)
    # to remove power line noise
    raw.notch_filter([50, 100], verbose=verbose)
    # Maxwell filtering using calibration files
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=SSS_CAL_FILE, cross_talk=CT_SPARSE_FILE,
        verbose=verbose)
    
    # Get events
    sfreq = raw.info["sfreq"]
    all_events, all_event_id = mne.events_from_annotations(
        raw, verbose=verbose)
    metadata, events, event_id = mne.epochs.make_metadata(
        events=all_events, event_id=all_event_id,
        tmin=metadata_tmin, tmax=metadata_tmax, sfreq=sfreq,
        row_events="button", keep_last="audiovis")
    
    # Create epochs
    epochs = mne.Epochs(
        raw, events, event_id, metadata=metadata, tmin=tmin, tmax=tmax,
        baseline=baseline, preload=True, verbose=verbose)
    
    # Only keep epochs matching the conditions: 
    # no other button press should occur within ``metadata_tmin`` seconds
    # and an audiovis stimulus should occur within one second
    epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]

    # Compute the forward and inverse operators
    trans = TRANS_DIR / f"sub-{subject}-trans.fif"
    bem = FREESURFER_DIR / f"{subject}/bem/{subject}-meg-bem.fif"
    src = mne.read_source_spaces(SRC, verbose=verbose)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem, verbose=verbose)
    del src
    cov = mne.compute_raw_covariance(raw, verbose=verbose)
    inv = make_inverse_operator(raw.info, fwd, cov, verbose=verbose)
    del fwd
    
    # Get the source time courses 
    # (inside the cortex; each source corresponds to a dipole)
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal",
        return_generator=False, verbose=verbose)

    # Get labels from "fsaverage"
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=parcellation, subjects_dir=FREESURFER_DIR, verbose=verbose)
    
    # Remove labels that do not have any vertex in the source space inv["src"]
    filtered_labels = [
        label for label in labels
        if (
            (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
            or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
        )
    ]
    
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
    selected_labels = [label for label in filtered_labels if label.name in label_names]

    
    # Compute average time course across all sources (dipoles) 
    # that belong to each label (region of interest, ROI),
    # so there is one time series per epoch and per label.
    label_ts = mne.extract_label_time_course(
        stcs, selected_labels, inv["src"], return_generator=False, verbose=verbose)
    
    # Band-pass ``label_ts`` at the beta waves range
    if band_pass_beta:
        label_ts = mne.filter.filter_data(
            label_ts, sfreq, 14, 30, verbose=verbose)

    # Compute the envelope.
    hilbert_ts = hilbert(label_ts, axis=2)
    envelope = np.abs(hilbert_ts)
    
    # Crop envelope to remove edge effects
    if n_crop_edges is not None:
        envelope = envelope[:, :, n_crop_edges:-n_crop_edges]

    # Perform batch-averaging
    envelope = batch_average(envelope, n_batches=n_batches)
    
    # Downsample time points
    if sfreq_envelope is not None:
        factor = int(sfreq // sfreq_envelope)  # window size
        if moving_avg:
            kernel = np.ones(factor) / factor
            envelope = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode='valid'), axis=2, arr=envelope)
        envelope = envelope[:, :, ::factor]

    # Concatenate batches
    n_labels = len(selected_labels)
    envelope = envelope.swapaxes(0, 1).reshape(n_labels, -1)
    
    return envelope, selected_labels


def batch_average(envelope, n_batches):
    """Perform batch-averaging.
    """
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


def plot_envelope(
    envelope,
    n_batches=1,
    tmin=-5.,
    tmax=5.,
    plot_avg=False,
    labels=None,
    save=False,
    subject=None,
):
    """Visualize envelope.
    """
    n_components, n_times = envelope.shape
    batch_size = n_times // n_batches
    if plot_avg:
        plt.subplots(figsize=(10, 8))
        envelope_avg = envelope[:, :batch_size].copy()
        for i in range(1, n_batches):
            envelope_avg += envelope[:, batch_size*i:batch_size*(i+1)]
        envelope_avg /= n_batches
        times = np.linspace(tmin, tmax, batch_size)
        if labels is not None:
            for i in range(n_components):
                plt.plot(times, envelope_avg[i], label=f"{labels[i].name}")
            plt.legend()
        else:
            plt.plot(times, envelope_avg.T)
        plt.vlines(
            x=0, ymin=np.min(envelope_avg), ymax=np.max(envelope_avg),
            linestyles="--", colors="grey")
        plt.xlabel("Time (s)")
    else:
        plt.subplots(figsize=(12, 4))
        if labels is not None:
            for i in range(n_components):
                plt.plot(envelope[i], label=f"{labels[i].name}")
            plt.legend()
        else:
            plt.plot(envelope.T)
        tick_positions = batch_size * np.arange(n_batches) + batch_size // 2
        tick_labels = np.arange(n_batches)
        plt.xticks(tick_positions, tick_labels)
        ymin, ymax = np.min(envelope), np.max(envelope)
        for i in range(n_batches+1):
            plt.vlines(x=batch_size*i, ymin=ymin, ymax=ymax, linestyles="--", colors="black")
        plt.xlabel("Batch")
    plt.grid()
    plt.ylabel("Mean amplitude")
    plt.title(f"Envelope of shape {envelope.shape}")
    if save:
        if subject is None:
            raise ValueError("subject should not be None when save=True")
        save_dir = "/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments/figures/"
        plt.savefig(save_dir + f"envelope_sub_{subject}.png")
    plt.show()


def get_participants():
    """Read participants file and only keep a subset of participants.
    """
    # Get the list of the 354 participants who have a trans file
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)
    participants_names = participants[
        'participant_id'].str.replace('sub-', '', regex=False).tolist()
    names_trans_available = [
        name for name in participants_names 
        if (TRANS_DIR / f"sub-{name}-trans.fif").exists()]
    participants_with_trans = participants[
        participants['participant_id'].str.replace('sub-', '', regex=False).isin(
            names_trans_available)]

    # Only keep participants with clean evoked data (40 out of the first 107 with a trans file)
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

    return filtered_participants
