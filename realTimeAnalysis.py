import os
import numpy as np
import sys
import wave
from datetime import datetime
from pyaudio import PyAudio, paInt16
import librosa
from sklearn import svm
from test import getData


class Audioer(object):
    def __init__(self):
        # pyaudio内置缓冲大小
        self.num_samples = 2000
        # 取样频率
        self.sampling_rate = 8000
        # 声音保存的阈值
        self.level = 1500
        # num_samples个取样之内出现count_num个大于level的取样则记录声音
        self.count_num = 20
        # 声音记录的最小长度：save_length*num_samples
        self.save_length = 8
        # 记录时间，s
        self.time_count = 10

        self.voice_string = []

    # 保存录音文件
    def save_wave(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sampling_rate)
        wf.writeframes(np.array(self.voice_string).tostring())
        wf.close()

    def read_audio(self):
        pa = PyAudio()
        stream = pa.open(
            format=paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.num_samples)

        save_count = 0
        save_buffer = []
        time_count = self.time_count

        print("正在记录你的声音……")
        while True:
            time_count -= 1

            # 读入num_samples个取样
            string_audio_data = stream.read(self.num_samples)
            # 读入的数据转为数组
            audio_data = np.frombuffer(string_audio_data, dtype=np.short)
            # 计算大于level的取样的个数
            large_samples_count = np.sum(audio_data > self.level)

            # 如果个数大于count_num,则至少保存save_length个块
            if large_samples_count > self.count_num:
                save_count = self.save_length
            else:
                save_count -= 1
            if save_count < 0:
                save_count = 0

            if save_count > 0:
                save_buffer.append(string_audio_data)
            else:
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    # 成功记录一段语音
                    return True

            if time_count == 0:
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    # 成功记录一段语音
                    return True
                else:
                    print("记录结束")
                    return False

        print("记录结束")
        return True


if __name__ == '__main__':
    r = Audioer()
    audio = r.read_audio()
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = now + ".wav"
    r.save_wave(path)
    # 提取声音特征
    y, sr = librosa.load(path)

    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc=16)
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = librosa.feature.rmse(y)
    rms_feature = librosa.feature.rmse(y)

    mfcc_feature = mfcc_feature.T.flatten()[:48]
    zcr_feature = zcr_feature.flatten()
    energy_feature = energy_feature.flatten()
    rms_feature = rms_feature.flatten()

    zcr_feature = np.array([np.mean(zcr_feature)])
    energy_feature = np.array([np.mean(energy_feature)])
    rms_feature = np.array([np.mean(rms_feature)])

    data_feature = np.concatenate((mfcc_feature, zcr_feature, energy_feature,
                                   rms_feature))

    # 使用svm进行预测
    classfier = svm.SVC(
        decision_function_shape='ovo',
        kernel='rbf',
        C=10,
        gamma=0.0001,
        probability=True)
    train_data, train_labels = getData(48)
    # 训练模型
    classfier.fit(train_data, train_labels)
    print(classfier.predict([data_feature]))
    print(classfier.predict_proba([data_feature]))
