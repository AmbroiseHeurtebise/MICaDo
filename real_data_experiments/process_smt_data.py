from pathlib import Path
import numpy as np
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
    n_components=None,
    n_batches=1,
    sfreq_envelope=None,
    tmin=-5.,
    tmax=5.,
    baseline=(-1.25, -1.0),
    metadata_tmin=-5.,
    metadata_tmax=0.,
    n_crop_edges=None,
    band_pass_beta=False,
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
        row_events="button", 
        keep_last="audiovis")
    
    # Create epochs
    epochs = mne.Epochs(
        raw, events, event_id, metadata=metadata,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True, verbose=verbose)
    
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
    cov = mne.compute_raw_covariance(raw)
    inv = make_inverse_operator(raw.info, fwd, cov, verbose=verbose)
    del fwd
    
    # Get the source time courses 
    # (inside the cortex; each source corresponds to a dipole)
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal",
        return_generator=False, verbose=verbose)

    # Get "aparc_sub" labels from fsaverage
    labels = mne.read_labels_from_annot(
        "fsaverage", "aparc_sub", subjects_dir=FREESURFER_DIR, verbose=verbose)
    
    # Remove labels that do not have any vertex in the source space inv["src"]
    filtered_labels = [
        label for label in labels
        if (
            (label.hemi == "lh" and len(set(label.vertices) & set(inv["src"][0]["vertno"])) > 0)
            or (label.hemi == "rh" and len(set(label.vertices) & set(inv["src"][1]["vertno"])) > 0)
        )
    ]
    
    # Compute average time course across all sources (dipoles) 
    # that belong to each label (region of interest, ROI),
    # so there is one time series per epoch and per label.
    label_ts = mne.extract_label_time_course(
        stcs, filtered_labels, inv["src"], return_generator=False, verbose=verbose)
    
    # Select a subset of labels
    # The general idea is that labels related to neural activity 
    # should have a higher variance
    var = np.mean(np.var(label_ts, axis=2), axis=0)  # shape (n_labels,)
    label_idx_good = np.argsort(var)[::-1][:n_components]
    np.random.shuffle(label_idx_good)  # XXX order is lost, although we need to keep labels
    label_ts_subset = [ts[label_idx_good] for ts in label_ts]

    # Band-pass ``label_ts`` at the beta waves range
    if band_pass_beta:
        label_ts_subset = mne.filter.filter_data(
            label_ts_subset, sfreq, 14, 30, verbose=verbose)

    # Compute the envelope.
    hilbert_ts = hilbert(label_ts_subset, axis=2)
    envelope = np.abs(hilbert_ts)
    
    # Crop envelope to remove edge effects 
    envelope_cropped = envelope[:, :, n_crop_edges:-n_crop_edges]

    # Perform batch-averaging
    envelope_batch_avg = batch_average(envelope_cropped, n_batches=n_batches)
    
    # Concatenate batches
    envelope_concat = envelope_batch_avg.swapaxes(0, 1).reshape(n_components, -1)

    # Downsample time points
    if sfreq_envelope is not None:
        factor = int(sfreq // sfreq_envelope)
    else:
        factor = 1
    envelope_reduced = envelope_concat[:, ::factor]
    
    return envelope_reduced


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
    save=False,
    subject=None,
):
    """Visualize envelope.
    """
    n_components, n_times = envelope.shape
    plt.subplots(figsize=(12, 6))
    if n_batches != 1:
        plt.plot(envelope.T)
        batch_length = n_times // n_batches
        tick_positions = batch_length * np.arange(n_batches) + batch_length // 2
        tick_labels = np.arange(n_batches)
        plt.xticks(tick_positions, tick_labels)
        ymin, ymax = np.min(envelope), np.max(envelope)
        for i in range(n_batches+1):
            plt.vlines(x=batch_length*i, ymin=ymin, ymax=ymax, linestyles="--", colors="black")
        plt.xlabel("Batch")
    else:
        t = np.linspace(tmin, tmax, n_times)
        plt.plot(t, envelope.T)
        plt.xlabel("Time (s)")
    plt.grid()
    plt.ylabel("Mean amplitude")
    plt.title(f"Envelope of shape {envelope.shape}")
    if save:
        save_dir = "/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments/"
        plt.savefig(save_dir + f"envelope_sub_{subject}.png")
    plt.show()


# Parameters
subject = "CC110033"
n_batches = 4  # used to batch-average epochs
sfreq_envelope = 10  # used to downsample envelope across timepoints dimension
n_components = 20
# We set metadata_tmin=-5 to remove button press events that occur within 5s of the previous event.
metadata_tmin, metadata_tmax = -5., 0. 
baseline = (-1.25, -1.0)
tmin, tmax = -5, 5
# List of 20 labels of the "aparc_sub" parcellation that are often related to motor processing
labels = ['precentral_1-lh', 'precentral_2-lh', 'precentral_3-lh', 'precentral_4-lh',
          'postcentral_1-lh', 'postcentral_2-lh', 'postcentral_3-lh', 'postcentral_4-lh',
          'superiorfrontal_1-lh', 'superiorfrontal_2-lh', 'precentral_1-rh', 'precentral_2-rh',
          'precentral_3-rh', 'precentral_4-rh', 'postcentral_1-rh', 'postcentral_2-rh',
          'postcentral_3-rh', 'postcentral_4-rh', 'superiorfrontal_1-rh', 'superiorfrontal_2-rh']

# check that trans file is available
trans = TRANS_DIR / f"sub-{subject}-trans.fif"
if not trans.exists():
    print("The ``trans`` file for this subject does not exist at this path. \
        Choose another subject.")
else:
    # Get envelope of shape (n_components, n_batches*(tmax-tmin)*sfreq_envelope)
    envelope_reduced = process_data_one_subject(
        subject,
        n_components=20,
        n_batches=4,
        sfreq_envelope=10,
        n_crop_edges=5,
    )
    # Plot
    plot_envelope(
        envelope_reduced, n_batches=n_batches, save=True,
        subject=subject)
