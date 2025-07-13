import torch
import numpy as np
import librosa
from emotion_cnn import EmotionCNN
from emotion_bilstm import EmotionBiLSTMWithAttention
from emotion_transformer import EmotionTransformer
from emotion_dataset import EMOTION_LABELS

def extract_features_for_inference(filepath, sr=16000, n_mels=128, max_len=300):
    y, _ = librosa.load(filepath, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)

    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)

    T = mel_db.shape[1]
    if len(f0) != T:
        f0 = np.interp(np.linspace(0, len(f0), T), np.arange(len(f0)), f0)
    f0 = f0[np.newaxis, :]

    features = np.vstack([mel_db, f0])  # shape: (129, T)

    # 补齐 / 裁剪
    if features.shape[1] < max_len:
        pad = np.zeros((129, max_len - features.shape[1]), dtype=np.float32)
        features = np.concatenate((features, pad), axis=1)
    elif features.shape[1] > max_len:
        features = features[:, :max_len]

    # 转为 Tensor (1, 129, T)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

def load_model(model_type: str, model_path: str, device, max_len=300):
    if model_type == 'cnn':
        model = EmotionCNN(n_input=129)
    elif model_type == 'bilstm':
        model = EmotionBiLSTMWithAttention(input_dim=129)
    elif model_type == 'transformer':
        model = EmotionTransformer(input_dim=129, max_len=max_len)
    else:
        raise ValueError("Unknown model type")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_emotion(filepath, model_type='cnn', model_path='emotion_cnn.pt', max_len=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, model_path, device, max_len)
    features = extract_features_for_inference(filepath, max_len=max_len).to(device)

    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        emotion = EMOTION_LABELS[pred_idx]

    return emotion, probs.squeeze().cpu().numpy()

if __name__ == "__main__":
    # wav_path = "dataset/neutral/203-neutral-wangzhe.wav"  # 替换为你的音频文件路径

    # 遍历所有 wav 文件 in directory `dataset/neutral`
    import os
    for filename in os.listdir("dataset/neutral"):
        if filename.endswith(".wav"):
            wav_path = os.path.join("dataset/neutral", filename)


            for model_type in ['cnn']:
                emotion, prob = predict_emotion(
                    filepath=wav_path,
                    model_type=model_type,
                    model_path=f"emotion_{model_type}.pt"
                )
                print(f"[{model_type.upper()}] 文件: {wav_path}\n🧠 情感: {emotion} ｜ 概率分布: {np.round(prob, 3)}\n")
