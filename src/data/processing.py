# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from constants import SAMPLING_RATE, FFT_SIZE, CENTER_FREQ # type: ignore

def load_dat_file(file_path: str) -> np.memmap:
    """Loads data from a .dat file using memory mapping.

    Args:
        file_path: The path to the .dat file.

    Returns:
        A numpy.memmap object representing the data in the file,
        assuming the data type is complex64.
    """
    return np.memmap(file_path, dtype=np.complex64, mode='r')

def length_corrector(signal: np.ndarray, correlation_length: int) -> np.ndarray:
    """Truncates a signal so its length is a multiple of correlation_length.

    Args:
        signal: The input signal as a NumPy array.
        correlation_length: The length to which the signal should be a multiple of.

    Returns:
        The truncated signal as a NumPy array.
    """
    number_of_vectors = signal.size // correlation_length
    croped_signal = signal[:number_of_vectors * correlation_length]
    return croped_signal

def get_sampling_rate(signal_size: int, recording_time: float) -> int:
    """Calculates the sampling rate given the signal size and recording time.

    Args:
        signal_size: The total number of samples in the signal.
        recording_time: The duration of the recording in seconds.

    Returns:
        The calculated sampling rate as an integer.
    """
    return int(signal_size // recording_time)

def visualize_signal(signal: np.ndarray, Fs: float = SAMPLING_RATE) -> matplotlib.figure.Figure:
    """Plots the magnitude of the complex signal over time.

    Args:
        signal: The input complex signal as a NumPy array.
        Fs: The sampling rate in Hz. Defaults to SAMPLING_RATE.

    Returns:
        A Matplotlib Figure object containing the plot. The figure is closed
        to prevent immediate display.
    """
    magnitude = np.abs(signal)
    time = np.arange(signal.size) / Fs

    fig, ax = plt.subplots()
    ax.plot(time, magnitude)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Signal Magnitude vs. Time")
    ax.grid(True)
    plt.tight_layout()
    plt.close(fig) # Close the figure to avoid displaying it
    return fig

def get_spectrogram(signal: np.ndarray, Fs: float = SAMPLING_RATE, NFFT: int = FFT_SIZE, Fc: float = CENTER_FREQ) -> matplotlib.figure.Figure:
    """Generates a spectrogram plot for a given signal.

    Args:
        signal: The input signal as a NumPy array.
        Fs: The sampling frequency in Hz. Defaults to SAMPLING_RATE.
        NFFT: The number of data points used in each block for the FFT.
              Defaults to FFT_SIZE.
        Fc: The center frequency in Hz. Defaults to CENTER_FREQ.

    Returns:
        A Matplotlib Figure object containing the spectrogram plot.
        The figure is closed to prevent immediate display.
    """
    fig, ax = plt.subplots(1)
    # Remove whitespace around the plot for a cleaner image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Generate the spectrogram using matplotlib's specgram function
    spectrum, freqs, t, im = ax.specgram(x=signal, Fs=Fs, NFFT=NFFT, Fc=Fc)
    # Turn off the axes (labels, ticks) for a pure image representation
    ax.axis('off')
    # Close the figure to prevent it from being displayed immediately
    plt.close(fig)
    return spectrum, fig

def get_stft_spectrogram(signal_sample: np.ndarray, window_size: int = FFT_SIZE, overlap: int = 128) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram.
    
    Parameters:
    -----------
    signal_sample : np.ndarray
        Input signal
    window_size : int, default=FFT_SIZE
        Size of the FFT window
    overlap : int, default=128
        Number of overlapping samples between windows
        
    Returns:
    --------
    stft_matrix : np.ndarray
        Complex STFT matrix with shape (window_size, num_time_bins)
    """
    step_size: int = window_size - overlap
    num_samples: int = signal_sample.size

    # Calculate the number of time bins (segments)
    num_time_bins: int = (num_samples - window_size) // step_size + 1

    # Initialize spectrogram matrix
    stft_matrix: np.ndarray = np.zeros((window_size, num_time_bins), dtype=np.complex64)

    # Define a window function to apply to each segment
    window: np.ndarray = np.hanning(window_size)

    for i in range(num_time_bins):
        start_index: int = i * step_size
        end_index: int = start_index + window_size
        segment: np.ndarray = signal_sample[start_index:end_index]

        # Apply window function
        windowed_segment: np.ndarray = segment * window

        # Perform FFT on the windowed segment
        fft_segment: np.ndarray = np.fft.fft(windowed_segment)
        # print(f"Type of fft_segment before fftshift: {type(fft_segment)}")
        # print(f"Content of fft_segment (first few elements): {fft_segment[:5] if isinstance(fft_segment, np.ndarray) else fft_segment}")
        fft_segment = np.fft.fftshift(fft_segment)  # Shift zero frequency to center
        
        stft_matrix[:, i] = fft_segment
    
    return stft_matrix

def get_centered_stft_matrix(signal_sample: np.ndarray, sampling_rate: float, fft_size: int,) -> np.ndarray:
    """
    Computes the Short-Time Fourier Transform (STFT) of a signal sample and centers the frequency bins.

    Args:
        signal_sample (np.ndarray): The input signal sample.
        sampling_rate (float): The sampling rate of the signal.
        fft_size (int): The FFT size to use for the STFT.

    Returns:
        np.ndarray: The centered STFT matrix.
    """
    # Compute the STFT
    _, _, stft_matrix = scipy.signal.stft(signal_sample, fs=sampling_rate, nperseg=fft_size, noverlap=128, window='hann', return_onesided=False)

    # Center the DC component (0 Hz)
    stft_matrix = np.fft.fftshift(stft_matrix, axes=0)

    return stft_matrix

def get_power_spectrogram_db(stft_matrix: np.ndarray, sampling_rate: float = SAMPLING_RATE, center_freq: float = CENTER_FREQ, 
                                window_size: int = FFT_SIZE, overlap: int = 128, step_size: int = None, eps: float = 1e-12) -> np.ndarray:
    """
    Compute power spectrogram in dB from STFT matrix.
    
    Parameters:
    -----------
    stft_matrix : np.ndarray
        Complex STFT matrix with shape (window_size, num_time_bins)
    sampling_rate : float, default=SAMPLING_RATE
        Sampling rate in Hz
    center_freq : float, default=CENTER_FREQ
        Center frequency for frequency axis
    window_size : int, default=FFT_SIZE
        FFT window size
    overlap : int, default=128
        Number of overlapping samples between windows
    step_size : Optional[int], optional
        Step size between windows (for time axis calculation)
    eps : float, default=1e-12
        Small value to avoid log(0)
        
    Returns:
    --------
    spectrum_db : np.ndarray
        Power spectrogram in dB scale
    """
    if step_size is None:
        step_size = window_size - overlap  # Default overlap of 128
    
    # Calculate power spectral density (magnitude squared)
    spectrum: np.ndarray = np.abs(stft_matrix) ** 2
    
    # Convert to dB for plotting
    spectrum_db: np.ndarray = 10 * np.log10(spectrum + eps)

    return spectrum_db

def get_color_img(spectrum_db: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
    """
    Convert power spectrogram in dB to color image.
    
    Parameters:
    -----------
    spectrum_db : np.ndarray
        Power spectrogram in dB scale
    colormap : str, default='viridis'
        Colormap to use for visualization
        
    Returns:
    --------
    color_img : np.ndarray
        Color image representation of the spectrogram
    """
    # Normalize the dB values to [0, 1] range for colormap
    norm_spectrum: np.ndarray = (spectrum_db - np.min(spectrum_db)) / (np.max(spectrum_db) - np.min(spectrum_db))
    
    # Apply colormap RGBA values ()
    rgba_img: np.ndarray = plt.get_cmap(colormap)(norm_spectrum)

    # Convert to RGB by removing the alpha channel
    alpha = rgba_img[..., 3:]       # shape (H, W, 1)
    rgb_channels = rgba_img[..., :3]     # shape (H, W, 3), floats in [0,1]
    bg_color = np.ones_like(rgb_channels)    # white background (1,1,1)

    # composite
    color_img = rgb_channels * alpha + bg_color * (1 - alpha)

    return color_img

def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal (can be complex).
    snr_db : float
        Desired Signal-to-Noise Ratio in dB.

    Returns:
    --------
    np.ndarray
        Signal with added AWGN.
    """
    # Calculate signal power
    signal_power: float = np.mean(np.abs(signal)**2)
    
    # Convert SNR from dB to linear scale
    snr_linear: float = 10**(snr_db / 10)
    
    # Calculate noise power
    noise_power: float = signal_power / snr_linear
    
    # Generate noise
    # For complex signals, noise should be complex (real and imaginary parts are Gaussian)
    noise: np.ndarray
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        
    noisy_signal: np.ndarray = signal + noise
    return noisy_signal

def save_color_img(color_img: np.ndarray, filename: str, dpi: int = 300, verbose: bool = True, overwrite: bool = False) -> None:
    """Saves a color image (numpy array) to a file.

    Args:
        color_img: The color image as a NumPy array.
        filename: The path and name of the file to save the image to.
        dpi: The resolution in dots per inch. Defaults to 300.
        verbose: If True, prints status messages. Defaults to True.
        overwrite: If True, overwrites the file if it already exists.
                   If False and the file exists, the function returns without saving.
                   Defaults to False.

    Returns:
        None
    """
    # Check if the file already exists and if overwriting is disallowed
    if not overwrite and os.path.exists(filename):
        if verbose:
            print(f"File {filename} already exists. Use overwrite=True to replace it.")
        return # Exit the function without saving

    # Save the color image using matplotlib
    plt.imsave(filename, color_img, dpi=dpi)

    if verbose:
        print(f"Color image saved to {filename}")

def save_spectrogram(fig: matplotlib.figure.Figure, filename: str, dpi: int = 300, verbose: bool = True, overwrite: bool = False) -> None:
    """Saves a Matplotlib Figure object, typically a spectrogram, to a file.

    Args:
        fig: The Matplotlib Figure object to save.
        filename: The path and name of the file to save the figure to.
        dpi: The resolution in dots per inch. Defaults to 300.
        verbose: If True, prints status messages. Defaults to True.
        overwrite: If True, overwrites the file if it already exists.
                   If False and the file exists, the function returns without saving.
                   Defaults to False.

    Returns:
        None
    """
    # Check if the file already exists and if overwriting is disallowed
    if not overwrite and os.path.exists(filename):
        if verbose:
            print(f"File {filename} already exists. Use overwrite=True to replace it.")
        return # Exit the function without saving

    # Save the figure to the specified file with the given DPI
    fig.savefig(filename, dpi=dpi)

    if verbose:
        print(f"Spectrogram saved to {filename}")

