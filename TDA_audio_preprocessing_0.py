from os import walk
import os
from pydub import AudioSegment
import librosa as lr
import numpy as np
from sklearn.preprocessing import StandardScaler


def from_mp3_to_wav(root_dir='covers80/', output_dir='covers80_wavs/'):
    """

    :param root_dir: str
        root directory, where all mp3 are stored
    :param output_dir: str
        directory, where all wavs will be stored
    """

    for (dirpath, dirnames, filenames) in walk(root_dir):
        if not dirnames:
            for i, filename in enumerate(filenames):
                wav_file = AudioSegment.from_mp3(dirpath + '/' + filename)
                output_subdir = output_dir + dirpath.split(sep='/')[1] + '/'
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                song_name = str(i) + '.wav'
                wav_file.export(output_subdir + song_name, format="wav")

    return 'Done!'


def save_ts(root_dir='covers80_wavs/', output_dir='covers80_ts/', sr=22050, duration=None, offset=0):
    """

    :param root_dir: str
        root directory, where all wavs are stored
    :param sr: number > 0 [scalar]
        target sampling rate
    :param duration: float
        only load up to this much audio (in seconds)
    :param offset: float
        start reading after this time (in seconds)
    """

    for (dirpath, dirnames, filenames) in walk(root_dir):
        print(dirpath)
        if not dirnames:
            for i, filename in enumerate(filenames):
                song_path = dirpath + '/' + filename
                ts, _ = lr.load(song_path, sr, offset=offset, duration=duration)
                output_subdir = output_dir + \
                                '_dur_{}__off_{}/'.format(duration, offset) + \
                                dirpath.split(sep='/')[1] + '/'
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                np.save(output_subdir + str(i), ts)

    return 'Done!'


def compute_audio_features(aw_ts, sr, features_list):
    """

    :param aw_ts: np.ndarray [shape=(n,) or (2, n)]
        analysis window's audio time series
    :param sr: number > 0 [scalar]
        sampling rate of aw_ts
    :param features_list: list of str
        list of used features to extract from time series
    :return: np.ndarray [shape=(stacked shape of each feature, len(aw_ts))]
        stacked features for one analysis window's timeseries
    """

    features = ()
    if 'mfcc' in features_list:
        mfcc = lr.feature.mfcc(aw_ts, sr, n_mfcc=12, n_fft=1024, hop_length=1024 // 4)
        features += (mfcc,)
    if 'chroma' in features_list:
        chroma = lr.feature.chroma_stft(y=aw_ts, sr=sr, n_fft=1024, hop_length=1024 // 4)
        features += (chroma,)
    if 'tonnetz' in features_list:
        chroma = lr.feature.chroma_stft(y=aw_ts, sr=sr, n_fft=1024, hop_length=1024 // 4)
        tonnetz = lr.feature.tonnetz(aw_ts, sr, chroma)
        features += (tonnetz,)
    features_arr = np.concatenate(features, axis=0)

    return features_arr


def from_ts_to_cloud_points(song_path='covers80_ts/_dur_None__off_0/Abracadabra/0.npy',
                            sr=22050,
                            texture_window_size=3,
                            analysis_window_size=0.2,
                            shift_size=1,
                            features_list=['mfcc', 'tonnetz', 'chroma']):
    """

    :param song_path: str
        path to the current song
    :param sr: number > 0 [scalar]
        target sampling rate
    :param features_list: list of str
        features to be used in cloud's point's formation
    :param texture_window_size: int
        window where cloud's point is calculated (in seconds)
    :param analysis_window_size: int
        window where features are calculated (in seconds)
    :param tw_shift_size: int
        shift of texture_window (in seconds)
    :return: np.ndarray [shape=(n features, n texture windows)]
        cloud points for one song
    """

    scl = StandardScaler()

    audio_ts = np.load(song_path)
    tw_sr_size = texture_window_size * sr
    aw_sr_size = int(analysis_window_size * sr)
    cloud_points = []
    for i in range(0, audio_ts.shape[0], shift_size * sr):
        if np.all(audio_ts[i:i + tw_sr_size]) == 0 or len(audio_ts[i:i + tw_sr_size]) < tw_sr_size:
            continue
        else:
            tw_ts = audio_ts[i:i + tw_sr_size]
            aw_features_list = []
            for j in range(0, tw_ts.shape[0], aw_sr_size):
                aw_ts = tw_ts[j:j + aw_sr_size]
                aw_features = compute_audio_features(aw_ts, sr, features_list)
                aw_features_list.append(aw_features)
            tw_features = np.concatenate(aw_features_list, axis=1)
            tw_feature_point = np.mean(tw_features, axis=1)
            tw_feature_point_norm = scl.fit_transform(tw_feature_point.reshape(-1, 1))
            cloud_points.append(tw_feature_point_norm)

    return np.concatenate(cloud_points, axis=1)


def make_cloud_points_all(root_dir='covers80_ts/_dur_None__off_0/', output_dir='cloud_points_all/',
                          features_list=['mfcc', 'tonnetz', 'chroma']):
    """

    :param root_dir: str
        root directory, where all time series are stored
    :param output_dir: str
        directory, where all cloud points will be stored
    :param features_list: list of str
        features to be used in cloud's point's formation

    :return: "Done!"
    """
    for (dirpath, dirnames, filenames) in walk(root_dir):
        if not dirnames:
            print(dirpath)
            for i, filename in enumerate(filenames):
                song_path = dirpath + '/' + filename
                output_subdir = output_dir + dirpath.split(sep='/')[1] + '/' + dirpath.split(sep='/')[2] + '/'
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                song_cloud_points = from_ts_to_cloud_points(song_path, features_list=features_list)
                np.savetxt(output_subdir + str(i) + '.csv', song_cloud_points)
    return "Done!"


# #all duration, no offset
make_cloud_points_all(output_dir='cloud_points_tonnetz/', features_list=['tonnetz'])
make_cloud_points_all(output_dir='cloud_points_mfcc/', features_list=['mfcc'])

# duration 60 sec, offset 10 sec
# make_cloud_points_all(root_dir='covers80_ts/_dur_60__off_10/', output_dir='cloud_points_all/')
make_cloud_points_all(root_dir='covers80_ts/_dur_60__off_10/', output_dir='cloud_points_tonnetz/', features_list=['tonnetz'])
make_cloud_points_all(root_dir='covers80_ts/_dur_60__off_10/', output_dir='cloud_points_mfcc/', features_list=['mfcc'])
