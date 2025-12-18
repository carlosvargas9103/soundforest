import gc

gc.collect()

import os
import copy
import time
import random
import indices

from typing import List, Tuple

import joblib
from joblib import Parallel, delayed

import librosa
import noisereduce as nr

from maad import sound, features

import pandas as pd
import numpy as np

from enum import Enum, unique

@unique
class Metrics(Enum):  # The metrics to train the predictive models
    REGION = 'reg'
    SOUNDSCAPE_ID = 'sid'
    BAND_ID = 'ban'
    SECOND = 'sec'
    MEAN = 'men'
    MEDIAN = 'med'
    SUM = 'sum'
    MAX = 'max'
    MIN = 'min'
    ACOUSTIC_COMPLEXITY = 'aci'
    ACOUSTIC_COMPLEXITY_ALTERNATIVE = 'aca'
    ACOUSTIC_DIVERSITY = 'adi'
    BIOACOUSTIC_INDEX_BETA = 'bet'
    TEMPORAL_MEDIAN = 'mmm'
    NUMBER_PEAKS = 'npp'
    ENTROPY_FREQUENCY = 'hfq'
    ENTROPY_TEMPORAL = 'htp'
    ENTROPY = 'hhh'
    ACOUSTIC_EVENNESS = 'aei'
    SOUNDSCAPE_INDEX = 'dsi'
    VECTOR_VEC = 'vec'

    def __repr__(self):
        return self._value_

    def __str__(self):
        return self._value_

    def __hash__(self):
        return hash(self._value_)

    def __eq__(self, other):
        return self._value_==other._value_ if isinstance(other, Metrics) else self._value_==other.__str__()

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


t00 = time.time()

DICT_BANDAS = {
    0: (0, 1000),
    1: (1000, 2000),
    2: (2000, 3000),
    3: (3000, 4000),
    4: (4000, 5000),
    5: (5000, 6000),
    6: (6000, 7000),
    7: (7000, 8000),
    8: (8000, 9000),
    9: (9000, 10000)
}

random.seed("9103")


# def get_audio_indices(file_name: str = '',
def get_audio_indices(s=None, fs: int = 0,
                      apply_filter: bool = False, start_freqs: int = 0, end_freqs: int = 10000
                      ) -> Tuple[float]:
    """
    Load an audio file, optionally apply a bandstop filter, and calculate various acoustic indices.

    Parameters:
    - file_name: string, path to the audio file.
    - apply_filter: boolean, indicates if a bandstop filter should be applied.
    - start_freqs: list of floats, start frequencies for the bandstop filters, used if apply_filter is True.
    - end_freqs: list of floats, end frequencies for the bandstop filters, used if apply_filter is True.

    Returns:
    - indices: list, containing the file name and computed acoustic indices or np.nan in case of an error.
    """

    # Load the audio file and compute the spectrogram
    # s, fs = torchaudio.load(str(file_name))
    # s, fs = torchaudio.load(str(file_name))
    # if s.size()[0] != 1:
    #     s = torch.unsqueeze(s[0, :], 0)

    # If filtering is applied, filter the signal before computing the spectrogram
    if apply_filter and start_freqs is not None and end_freqs is not None:
        s_filtered = butter_bandstop_filter(s.numpy()[0, :], start_freqs, end_freqs, fs)
        s = torch.tensor(s_filtered.copy()[None, :])  # Add a new axis to make it 2D again
        rms = np.sqrt(np.mean(np.square(s.numpy()[0, :])))
        s = s / rms  # Normalize the signal
        # Sxx, tn, fn, ext = sound.spectrogram(s[0, :], fs, mode='amplitude')
        Sxx, tn, fn, ext = sound.spectrogram(s, fs, mode='amplitude')
        # Sxx_power, _, _, _ = sound.spectrogram(s[0, :], fs)
        Sxx_power, _, _, _ = sound.spectrogram(s, fs)
        ADI = 0
        AEI = 0
        for f1, f2 in zip(start_freqs, end_freqs):
            ADI += features.acoustic_diversity_index(Sxx, fn, f1, f2)
            AEI += features.acoustic_eveness_index(Sxx, fn, f1, f2)

    # Compute the spectrogram
    if apply_filter == False:
        # Sxx, tn, fn, ext = sound.spectrogram(s[0, :], fs, mode='amplitude')
        Sxx, tn, fn, ext = sound.spectrogram(s, fs, mode='amplitude')
        # Sxx_power, _, _, _ = sound.spectrogram(s[0, :], fs)
        Sxx_power, _, _, _ = sound.spectrogram(s, fs)
        ADI = features.acoustic_diversity_index(Sxx, fn)  # , fmax=int(fs / 2), dB_threshold=-40)
        AEI = features.acoustic_eveness_index(Sxx, fn)  # , fmax=int(fs / 2), dB_threshold=-40)
        # maad.features.acoustic_richness_index (PENDING)

    # Calculate acoustic indices
    ACIft_ = indices.ACIft(Sxx)
    _, _, ACI = features.acoustic_complexity_index(Sxx) # TODO: Include this into the return
    BETA = features.bioacoustics_index(Sxx, fn, flim=(2000, 8000))
    # M = features.temporal_median(s[0, :], mode='hilbert')
    M = features.temporal_median(s, mode='hilbert')
    NP = features.number_of_peaks(Sxx_power, fn)  # , slopes=6, min_freq_dist=100, display=False)
    Hf, _ = features.frequency_entropy(Sxx_power)
    # Ht = features.temporal_entropy(s.numpy()[0, :], mode='hilbert')
    # Ht = features.temporal_entropy(s.numpy()[:], mode='hilbert')
    Ht = features.temporal_entropy(s, mode='hilbert')
    H = Ht * Hf
    NDSI, _, _, _ = features.soundscape_index(Sxx_power, fn, flim_bioPh=(0, 10000), flim_antroPh=(0, 1000))

    # Compile all indices into a list
    # acoustic_indices = [ACIft_, ADI, BETA, M, NP, H, AEI, NDSI]
    # print('INDICES', ACIft_, ADI, BETA, M, NP, H, AEI, NDSI)
    return (ACIft_, ADI, BETA, M, NP, Hf, Ht, H, AEI, NDSI)

    # Include file name information in the indices list
    # file_info = [str(file_name).split('/')[-1], str(file_name).split('/')[-2]]
    # return file_info + acoustic_indices


def bootstrap_soundscape(audio_file: str = '',
                         region: str = '',
                         si: int = 1964, *,
                         sr: int = 48000,
                         b_band: int = 0,
                         u_band: int = 10000,
                         bandas: int = 10,
                         bandwidth: int = 1000,
                         path_data: str = '',
                         path_out: str = '',
                         samples_s: int = 1800, # this needs to be self-calculated
                         isamples_s: int = 3,
                         secs_b: int = 6,
                         secs_o: int = 1.9,
                         hanning: bool = True,
                         w_size_mins: float = 0.06,
                         n_jobs: int = 1,
                         job_id: str = 'NULL',
                         verbose: bool = False,
                         f_pattern_out: str = 'extraction',
                         windows_13: bool = True,
                         horas: int = 30,
                         metric_names: List[str] = Metrics.list()
                         ) -> None:
    d_re = {'NaturalRegeneration': 0, 'Pasture': 1, 'Plantation': 2, 'RefForest': 3}
    # print(f'{si} - audio_file: {audio_file} - region: {region}')
    # PARALLEL JOBS PER FILE
    y, y_c = None, None
    y, sr = librosa.load(audio_file, sr=None)  # , duration=1800)
    print("SAMPLE RATE => ", sr)
    # y = 86400000, sr = 48000
    y_c = copy.deepcopy(y)
    tt = int(len(y_c) / sr)
    # print(f'y: {y_c[:3]}')
    # print(f'total samples in y: {y_c.shape} in time: {tt} secs')
    # print(f'samples rate per second:  {sr}')
    print('####', 'ORIGINAL Y_C', len(y_c), y_c[:3], type(y_c))
    # [1800 / 1 => per 1 sec, 1800 / 3 => per 3 sec, 1800 / 30 => per 30 sec, 1800 / 60 => per 60 sec]
    # NUMBERS 48000 3 1800 6
    # print('NUMBERS', sr, isamples_s, samples_s, secs_b)
    # my_chunks = samples_s / secs_b
    # split_y = np.hsplit(y_c, my_chunks)
    #
    # print('####', 'CHUNKS Y_C',
    #       f'total_chunks: {len(split_y)}',
    #       f'chunk_size: {len(split_y[0])}',
    #       type(split_y), type(split_y[0])
    #       )

    # secs_b: int = 6, secs_o: int = 1.9,
    frame_size, hop_size = sr * secs_b, int(sr * (secs_b - secs_o))

    def hanning(y: np.array = None, frame_size: int = frame_size, hop_size: int = hop_size, hanning: bool = True):
        signal = np.array(y)
        # num_frames = 1 + (len(signal) - frame_size) // hop_size
        # print(num_frames) # 438 or 449
        if hanning:
            hanning_window = np.hanning(frame_size)
            return [signal[i:i + frame_size] * hanning_window for i in range(0, len(signal) - frame_size + 1, hop_size)]
        return [signal[i:(i + frame_size)] for i in range(0, len(signal) - frame_size + 1, hop_size)]

    split_y = hanning(y_c, frame_size, hop_size, hanning)

    # TODO: Place this print in code.
    print('####', 'HANNING Y_C',
          f'total_chunks: {len(split_y)}',
          f'chunk_size first: {len(split_y[0])}',
          f'chunk_size last: {len(split_y[-1])}',
          type(split_y), type(split_y[0])
          )

    split_y_indices, total_indices = None, 8
    indices = True
    if indices:
        # PARALLEL split_freq_band => ~6sec
        print('####', 'PARALLEL', 'INDICES', '####')
        t1 = time.time()
        split_y_indices = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(get_audio_indices)(y_i, sr) for y_i in split_y)
        )
        # split_y_indices = [get_audio_indices(y_i, sr) for y_i in split_y]

        print('####', 'PARALLEL INDICES Y_C', len(split_y_indices), len(split_y_indices[0]),
              split_y_indices[0][:3], f'{(time.time() - t1):.3}')

    # print('####', 'HANNING', '\n',
    #       f'hanning => max: {max(split_y[0])} min: {min(split_y[0])} mean: {(split_y[0].mean())} size: {len(split_y[250])}', '\n',
    #       f'hanning => max: {max(split_y[250])} min: {min(split_y[250])} mean: {split_y[250].mean()} size: {len(split_y[250])}'
    #       )
    # exit()

    # TODO: ATM, WE DO NOT CALL THESE METHODS
    # TODO: ATM, WE DO NOT CALL AMPLITUDE ENVELOPE
    # wrap methods audio_denoise with parameters
    # def audio_denoise_st(y=None, sr: int = 48000):
    def audio_denoise_st(y=None, sr: int = sr):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.9, stationary=True)

    # def audio_denoise_ns(y=None, sr: int = 48000):
    def audio_denoise_ns(y=None, sr: int = sr):
        return nr.reduce_noise(y=y, sr=sr, n_std_thresh_stationary=1.6, stationary=False)

    denoise = False
    if denoise:
        # PARALLEL split_freq_band => ~6sec
        print('####', 'PARALLEL audio_denoise')
        t1 = time.time()
        split_y_st = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_st)(y_i) for y_i in split_y)
        )
        split_y_ns = np.array(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(audio_denoise_ns)(y_i) for y_i in split_y)
        )
        # 438 288000 [2.19772187e-26 1.05804673e-26 5.78876953e-27] 18.265
        print('####', 'PARALLEL DENOISE Y_C', len(split_y_st), len(split_y_st[0]), split_y_st[0][:3], round(time.time() - t1, 3))

    def split_freq_band_per_frame(s: np.memmap = None,
                                  sr: int = sr,
                                  b_band: int = b_band,
                                  u_band: int = u_band,
                                  bandwidth: int = bandwidth) -> List[np.ndarray]:
        # Calculate the FFT
        # 2880000 48000 2.0833333333333333e-05 10000 0 1000 => 60 seconds
        # 288000 48000 2.0833333333333333e-05 10000 0 1000 => 6 seconds
        # print(len(s), sr, 1/sr, u_band, b_band, bandwidth)
        y_fft = np.fft.fft(s)
        # Calculate the frequencies for the FFT
        fft_freq = np.fft.fftfreq(len(s), 1.0 / sr)
        signals = []
        for f in range(b_band, u_band, bandwidth):
            mask = (fft_freq >= f) & (fft_freq < f + bandwidth)
            y_fft_band = y_fft[mask]
            fft_freq_band = fft_freq[mask]
            y_band = np.fft.ifft(y_fft_band)
            y_band = np.real(y_band)  # Take only the real part
            signals.append(y_band)
        return signals

    # PARALLEL split_freq_band => ~6sec
    print('####', 'PARALLEL split_freq_band')
    t1 = time.time()
    split_y_split_bands = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(split_freq_band_per_frame)(y_i) for y_i in split_y)
    )
    # PARALLEL SPLIT Y_C PER FREQUENCY 438 10 6000 [-3.44860133e-05 -3.45421108e-05 -3.45807478e-05] 2.368
    print('####', 'PARALLEL SPLIT Y_C PER FREQUENCY',
          len(split_y_split_bands), len(split_y_split_bands[0]),
          len(split_y_split_bands[0][0]), split_y_split_bands[0][0][:3],
          round(time.time() - t1, 3)
          )

    t1 = time.time()

    # # y_split_hours_bands = [split_y_split_bands[h, b, :].flatten() for b in range(0, bandas) for h in range(0, horas)]
    # y_split_seconds_bands = [split_y_split_bands[s, b, :].flatten() for b in range(0, bandas) for s in range(0, secs_b)]
    # # exit()
    #
    # print('####', 'REARRANGED THE SIGNAL PER (bands = 10) FREQUENCIES', type(y_split_seconds_bands), len(y_split_seconds_bands),
    #       type(y_split_seconds_bands[0]), len(y_split_seconds_bands[0]))

    # exit()

    def band_acift(s):
        intensity_vector = np.array(s)
        # Avoid division by zero
        if np.sum(intensity_vector) == 0:
            return 0.0
        # Calculate the numerator: sum of absolute differences between consecutive values
        numerator = np.sum(np.abs(np.diff(intensity_vector)))
        # Calculate the denominator: sum of intensity values
        denominator = np.sum(intensity_vector)
        # Compute ACIft
        return numerator / denominator

    # ARRANGE in dict per BANDAS, and HORAS
    # NOTE: Calculate ALL THE POSSIBLE INDICES. Then, subsample in MODELLING!!!!
    metrics_y_seconds_bandas = [
        {
            Metrics.REGION: d_re.get(region, 0),  # CLASS (INT) 4
            Metrics.SOUNDSCAPE_ID: si,  # audio_file_id
            Metrics.SECOND: s,  # TIME (int) 0 - 6 => DONT NEED!?? - SUNDAY (05.01.25)!
            Metrics.BAND_ID: b,  # BAND (int) 0 - 9  => TODO: Consider 10-bands at once - MONDAY (22.01.25)!
            Metrics.MEAN: np.mean(f := split_y_split_bands[s, b, :].flatten()),  # MEAN of the FRAME (float)
            Metrics.MEDIAN: np.median(f),  # MEDIAN of the FRAME (float)
            Metrics.SUM: np.sum(f),  # SUM of the FRAME (float)
            Metrics.MAX: np.max(f),  # MAX of the FRAME (float)
            Metrics.MIN: np.min(f),  # MIN of the FRAME (float)
            Metrics.ACOUSTIC_COMPLEXITY: band_acift(f),  # Acoustic Complexity Index per band-split (float)
            Metrics.ACOUSTIC_COMPLEXITY_ALTERNATIVE: split_y_indices[s, 0], # TODO: Check for the alternative
            Metrics.ACOUSTIC_DIVERSITY: split_y_indices[s, 1],
            Metrics.BIOACOUSTIC_INDEX_BETA: split_y_indices[s, 2],
            Metrics.TEMPORAL_MEDIAN: split_y_indices[s, 3],
            Metrics.NUMBER_PEAKS: split_y_indices[s, 4],
            Metrics.ENTROPY_FREQUENCY: split_y_indices[s, 5],
            Metrics.ENTROPY_TEMPORAL: split_y_indices[s, 6],
            Metrics.ENTROPY: split_y_indices[s, 7],
            Metrics.ACOUSTIC_EVENNESS: split_y_indices[s, 8],
            Metrics.SOUNDSCAPE_INDEX: split_y_indices[s, 9],
            Metrics.VECTOR_VEC: f  # FRAME (npArray[float]) 438 x 10 bands => 6000 each vec => 6 secs x 1000 samples_sec
         } for s in range(0, len(split_y_split_bands)) for b in range(0, bandas)
    ]

    print('####', 'TIMES', '####', 'DICT_VECTOR:', round(time.time() - t1, 3))

    # exit()

    # <class 'numpy.ndarray'> 10
    # <class 'numpy.ndarray'> 484 => 1800 (30min) / 10x484 = ~3.6 seconds
    df = pd.DataFrame(metrics_y_seconds_bandas)
    v_prefix = Metrics.VECTOR_VEC
    df_v = pd.DataFrame(df[v_prefix].to_list()).add_prefix(f'{v_prefix}_')
    # (60, 6009)
    # print(df_v.head(3), df_v.shape)
    df.drop(columns=[v_prefix], inplace=True)
    df_m = pd.concat([df, df_v], axis=1)
    # print(df_m.head(3), df_m.shape)
    # exit()

    # TODO: Confusion Matrix for classification => Table & Matrix - Pending!

    file_name_name = os.path.splitext(os.path.basename(audio_file))[0]
    df_m.to_pickle(
        f'{path_out}data/{f_pattern_out}/{region}/{file_name_name}_dict_y_split_{si}_{job_id}_{int(time.time())}.pkl'
    )
    print('#### TIMES #### extraction TOTAL TOTAL ==>>', round(time.time() - t00, 3))


if __name__ == '__main__':
    print('Mirá ve.. oís?? alles gut oder was??')
