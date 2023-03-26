import os

import numpy as np

from preprocessing import load_wav, save_wav, istft

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model


def aec(model_path, mixed_path, ref_path, output_path, gain=1.0):
    mic_sig, mic_sr = load_wav(tf.constant(mixed_path))
    ref_sig, ref_sr = load_wav(tf.constant(ref_path))

    total_sig_len = tf.minimum(len(mic_sig), len(ref_sig))

    mic_sig = mic_sig[:total_sig_len]
    ref_sig = ref_sig[:total_sig_len]

    model = load_model(model_path)

    f_len = 320
    f_step = 160
    l = 4000  # 250ms
    num_frames = total_sig_len // l

    delta = l - total_sig_len % l

    if delta < l:
        padding = np.zeros(delta)
        mic_sig = np.concatenate((mic_sig, padding))
        ref_sig = np.concatenate((ref_sig, padding))

    concat_sig = np.concatenate((mic_sig, ref_sig))
    frames = np.split(concat_sig, num_frames*2)

    z = tf.signal.stft(frames, frame_length=f_len, frame_step=f_step, fft_length=f_len)
    mic_z = z[:num_frames, :, :]
    ref_z = z[num_frames:, :, :]

    stacked_z = tf.concat([mic_z, ref_z], axis=-1)

    speech_masks = model(stacked_z)
    mask_prods = tf.multiply(mic_z, tf.cast(speech_masks, tf.complex64))

    rec_speech = istft(mask_prods)
    clean_signal = tf.reshape(rec_speech, [-1])
    clean_signal *= gain

    out_wav_path = os.path.join(output_path, f"tf_reconstructed_signal_x{gain}.wav")
    save_wav(tf.constant(out_wav_path), tf.cast(clean_signal, tf.float32), mic_sr)

    f, ax = plt.subplots(3)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("Noisy Signal")
    ax[0].plot(mic_sig)
    ax[1].set_ylim(-1, 1)
    ax[1].set_title("Reference Signal")
    ax[1].plot(ref_sig)
    ax[2].set_ylim(-1, 1)
    ax[2].set_title("Reconstructed Speech")
    ax[2].plot(clean_signal)
    f.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    pb_path = ""
    output_path = ""
    mixed_path = ""
    ref_path = ""
    gain = 1.0

    aec(pb_path, mixed_path, ref_path, output_path, gain)
