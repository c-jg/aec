import os
import shutil
from glob import glob
import random

import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt


@tf.function
def irm(background, speech, beta=0.5):
    background = tf.cast(background, tf.float32)
    speech = tf.cast(speech, tf.float32)

    sq_s = tf.math.square(speech)
    sq_b = tf.math.square(background)
    s_b = tf.math.add(sq_s, sq_b)
    mask = tf.divide(sq_s, s_b)

    return tf.pow(mask, beta)


@tf.function
def load_wav(file_path):
    file_contents = tf.io.read_file(file_path)

    sig, sr = tf.audio.decode_wav(file_contents, desired_channels=1)
    sig = tf.squeeze(sig, axis=-1)

    sig = tf.cond(sr != 16000,
                  true_fn=lambda: tfio.audio.resample(sig, rate_in=tf.cast(sr, dtype=tf.int64),
                                                      rate_out=16000),
                  false_fn=lambda: sig)

    return sig, sr


@tf.function
def load_stereo_wav(file_path):
    file_contents = tf.io.read_file(file_path)
    sig, sr = tf.audio.decode_wav(file_contents, desired_channels=2)

    return sig, sr


@tf.function
def save_wav(file_path, signal, sr):
    signal = tf.expand_dims(signal, -1)
    encoded_signal = tf.audio.encode_wav(signal, sample_rate=sr)
    tf.io.write_file(file_path, encoded_signal)

    return


@tf.function
def save_stereo_wav(file_path, signal, sr):
    encoded_signal = tf.audio.encode_wav(signal, sample_rate=sr)
    tf.io.write_file(file_path, encoded_signal)

    return


@tf.function
def istft(stft):
    frame_length = 320
    frame_step = 160

    i_s = tf.signal.inverse_stft(
        stft,
        frame_length,
        frame_step,
        fft_length=frame_length,
        window_fn=tf.signal.inverse_stft_window_fn(frame_step)
    )

    return i_s


@tf.function
def ibm(ref_z, mixed_z):
    snr = tf.divide(tf.abs(ref_z), tf.abs(mixed_z))
    mask2 = tf.math.round(snr)
    mask2 = tf.where(tf.math.is_nan(mask2), 1.0, mask2)
    mask2 = tf.where(mask2 > 1.0, 1.0, mask2)

    mask1 = 1.0 - mask2

    return mask1, mask2


def test_synthesis(lpb_path, speech_path, out_wav_path, seconds, target):
    signal_len = seconds * 16000
    mic_sig, mic_sr = load_wav(speech_path)[:signal_len]
    ref_sig, ref_sr = load_wav(lpb_path)[:signal_len]

    mixed_signal = tf.add(ref_sig, mic_sig)

    f_len = 320
    f_step = 160

    z1 = tf.signal.stft(mic_sig, frame_length=f_len, frame_step=f_step, fft_length=f_len)
    z2 = tf.signal.stft(ref_sig, frame_length=f_len, frame_step=f_step, fft_length=f_len)
    z3 = tf.signal.stft(mixed_signal, frame_length=f_len, frame_step=f_step, fft_length=f_len)

    if target == "ibm":
        speech_mask, ref_mask = ibm(z2, z3)
        z_masked_2 = tf.multiply(z3, tf.cast(ref_mask, tf.complex64))
        rec_music = istft(z_masked_2)
    elif target == "irm":
        speech_mask = irm(z2, z1)
    else:
        print(f"Target mask {target} not supported.")
        return

    z_masked_1 = tf.multiply(z3, tf.cast(speech_mask, tf.complex64))
    rec_speech = istft(z_masked_1)

    f, ax = plt.subplots(2, 1)
    ax[0].set_title("Mixed")
    ax[0].plot(mixed_signal)
    ax[1].set_title("Reconstructed Speech")
    ax[1].plot(rec_speech)
    f.tight_layout()
    plt.show()

    save_wav(out_wav_path, rec_speech, 16000)

    return


def split_ref_and_speech_data(input_path, output_path, valid_split):
    assert valid_split < 1.0, "Percentage of validation data must be less than 1.0"

    ref_path = os.path.join(input_path, "REF")
    speech_path = os.path.join(input_path, "SPEECH")

    split_files(ref_path, output_path, valid_split)
    split_files(speech_path, output_path, valid_split)

    return


def split_files(input_path, output_path, valid_split):
    data_type = os.path.basename(input_path)

    all_files = glob(os.path.join(input_path, "*.wav"))
    num_files = len(all_files)

    random.shuffle(all_files)

    valid_idx = int(valid_split * num_files)

    valid_files = all_files[:valid_idx]
    train_files = all_files[valid_idx:]

    move_files(train_files, "TRAIN", data_type, output_path)
    move_files(valid_files, "VALID", data_type, output_path)

    return


def move_files(files, dataset, data_type, output_path):
    dataset_output_path = os.path.join(output_path, dataset, data_type)

    if not os.path.exists(dataset_output_path):
        os.makedirs(dataset_output_path)

    for file in files:
        dst = os.path.join(dataset_output_path, os.path.basename(file))
        shutil.copyfile(file, dst)

    return


def split_wavs_into_1s_files(input_path, output_path):
    datasets = os.listdir(input_path)

    for dataset in datasets:
        dataset_path = os.path.join(input_path, dataset)
        signal_types = os.listdir(dataset_path)

        for signal_type in signal_types:
            wav_files = glob(os.path.join(dataset_path, signal_type, "*.wav"))

            out_dir = os.path.join(output_path, dataset, signal_type)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for wav in wav_files:
                sig, sr = load_wav(tf.constant(wav))
                len_sig = len(sig)
                end = len_sig - len_sig % sr
                cut_sig = sig[:end]

                num_splits = int(len(cut_sig) / sr)

                if num_splits == 0:
                    print(f"Not enough splits ({num_splits}) in {wav}. SKIPPING")
                    continue

                one_second_sigs = tf.split(cut_sig, num_splits)

                for i, one_second_sig in enumerate(one_second_sigs):
                    file_name = f"{os.path.splitext(os.path.basename(wav))[0]}_{i}.wav"
                    dst = os.path.join(out_dir, file_name)
                    save_wav(tf.constant(dst), one_second_sig, sr)

                print(f"Done splitting {dataset} - > {signal_type} -> {wav}")

    return


def combine_speech_and_ref(input_path, output_path):
    datasets = os.listdir(input_path)
    output_qty_dict = {"TRAIN": 1_000_000, "VALID": 100_000}

    for dataset in datasets:
        dataset_path = os.path.join(input_path, dataset)

        speech_dir = os.path.join(dataset_path, "SPEECH")
        ref_dir = os.path.join(dataset_path, "REF")

        speech_files = glob(os.path.join(speech_dir, "*.wav"))
        ref_files = glob(os.path.join(ref_dir, "*.wav"))

        random.shuffle(speech_files)
        random.shuffle(ref_files)

        out_dir = os.path.join(output_path, dataset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        num_output = output_qty_dict[dataset]
        num_speech_files = len(speech_files)
        num_to_mix = num_output // num_speech_files

        for i, speech_file in enumerate(speech_files):
            speech_sig, _ = load_wav(tf.constant(speech_file))
            speech_name = os.path.splitext(os.path.basename(speech_file))[0]

            for j, ref_file in enumerate(random.sample(ref_files, num_to_mix)):
                ref_name = os.path.splitext(os.path.basename(ref_file))[0]

                ref_sig, sr = load_wav(tf.constant(ref_file))

                mixed_sig = speech_sig + ref_sig

                stereo_sig = tf.stack((mixed_sig, ref_sig), axis=-1)

                save_name = f"{speech_name}_SS_{ref_name}_RS_{i}_{j}.wav"
                dst = os.path.join(out_dir, save_name)

                save_stereo_wav(tf.constant(dst), stereo_sig, sr)

    return


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_dataset(filenames, batch_size=64):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_spect_record, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def aec_example(spectrogram, target_mask):
    """Return TF example protobuf containing spectrogram and target mask."""
    serialized_spect = tf.io.serialize_tensor(spectrogram)
    serialized_mask = tf.io.serialize_tensor(target_mask)

    spect_feat_bytes = _bytes_feature(serialized_spect)
    mask_feat_bytes = _bytes_feature(serialized_mask)

    feature = {
        'spectrogram': spect_feat_bytes,
        'target_mask': mask_feat_bytes
    }

    example_proto = tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )

    return example_proto


def parse_spect_record(example):
    """Parses one example given feature description."""
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'target_mask': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(example, feature_description)

    spect = content['spectrogram']
    target_mask = content['target_mask']

    spect = tf.io.parse_tensor(spect, tf.complex64)
    target_mask = tf.io.parse_tensor(target_mask, tf.float32)

    return spect, target_mask


def write_examples_to_record(examples, output_path):
    """Write examples to tfrecords file until file reaches specified size."""
    tfrecord_file_size = 10.2 ** 8

    counter = 0
    tfrecord_save_path = os.path.join(output_path, f"{counter}.tfrecords")

    writer = tf.io.TFRecordWriter(tfrecord_save_path)

    for example in examples:
        if os.path.getsize(tfrecord_save_path) > tfrecord_file_size:
            counter += 1
            writer.close()

            tfrecord_save_path = os.path.join(output_path, f"{counter}.tfrecords")

            writer = tf.io.TFRecordWriter(tfrecord_save_path)

        writer.write(example.SerializeToString())

    writer.close()

    return


def generate_example(stereo_file):
    spectrogram, target_mask = extract_features(tf.constant(stereo_file))
    example = aec_example(spectrogram, target_mask)

    return example


def generate_spectrogram_and_mask_tfrecords(input_path, output_path):
    datasets = os.listdir(input_path)

    for dataset in datasets:
        dataset_output_path = os.path.join(output_path, dataset)

        if not os.path.exists(dataset_output_path):
            os.mkdir(dataset_output_path)

        dataset_input_path = os.path.join(input_path, dataset)

        stereo_files = glob(os.path.join(dataset_input_path, "*.wav"))

        examples = map(generate_example, stereo_files)
        write_examples_to_record(examples, dataset_output_path)

    return


@tf.function
def extract_features(stereo_wav_file_path):
    sigs, sr = load_stereo_wav(stereo_wav_file_path)

    duration = 4000  # 250ms
    mixed_sig = sigs[:, 0][:duration]
    ref_sig = sigs[:, 1][:duration]

    z_mixed = tf.signal.stft(mixed_sig, frame_length=320, frame_step=160, fft_length=320)
    z_ref = tf.signal.stft(ref_sig, frame_length=320, frame_step=160, fft_length=320)

    target_mask, _ = ibm(z_ref, z_mixed)  # (24, 161)

    f_m = tf.concat([z_mixed, z_ref], axis=-1)  # (24, 322)

    return f_m, target_mask
