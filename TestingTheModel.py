import librosa
import numpy as np
from tensorflow.keras.models import load_model
from scipy.special import softmax

def extract_features(filepath):
    audio, rate = librosa.load(filepath, mono=True, duration=30)
    mfccs = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=15)
    mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, 500)))
    mfccs_padded_mean = np.mean(mfccs_padded, axis=1)
    mfccs_reshaped = mfccs_padded_mean.reshape(1, mfccs_padded_mean.shape[0], 1)
    return mfccs_reshaped

# Loading the model
model = load_model('myModel_withoutNormilizaing.keras')
audioPath = "test/Solo Guitar.mp3"
data = extract_features(audioPath)
prediction = model.predict(data)
normalized_probs = softmax(prediction)

class_labels = ['Piano', 'Guitar', 'Violin', 'Drum']

for prob, label in zip(normalized_probs[0], class_labels):
    print(f"{label}: {(prob * 100):.2f} %")



