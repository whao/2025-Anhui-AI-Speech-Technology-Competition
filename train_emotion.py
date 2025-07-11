import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from emotion_dataset import EmotionFeatureDataset, EMOTION_LABELS
from emotion_cnn import EmotionCNN
from emotion_bilstm import EmotionBiLSTMWithAttention
from emotion_transformer import EmotionTransformer

# -------------------- 配置项 --------------------
MODEL_TYPE = 'transformer'   # 可选：'cnn', 'bilstm', 'transformer'
BATCH_SIZE = 16
N_EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_LEN = 300
MODEL_PATH = f'emotion_{MODEL_TYPE}.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 数据加载 --------------------
train_dataset = EmotionFeatureDataset('dataset', split='train', max_len=MAX_LEN)
val_dataset = EmotionFeatureDataset('dataset', split='val', max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------- 模型选择 --------------------
if MODEL_TYPE == 'cnn':
    model = EmotionCNN(n_input=129)
elif MODEL_TYPE == 'bilstm':
    model = EmotionBiLSTMWithAttention(input_dim=129)
elif MODEL_TYPE == 'transformer':
    model = EmotionTransformer(input_dim=129, max_len=MAX_LEN)
else:
    raise ValueError("Unknown model type")

model.to(DEVICE)

# -------------------- 训练准备 --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 训练函数 --------------------
def train():
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"🧪 Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        evaluate()

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 模型已保存: {MODEL_PATH}")

def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        with record_function("forward_and_backward"):
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    print(f"✅ Epoch {epoch+1}: Loss = {running_loss / len(loader):.4f}")

def train_with_profiler(model, train_loader, device, epochs=1):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 使用 torch.profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(epochs):
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                with record_function("model_train_step"):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                prof.step()

    # 打印关键耗时操作
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# -------------------- 评估函数 --------------------
def evaluate():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("🎯 验证集分类报告:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))


# -------------------- 启动训练 --------------------
if __name__ == "__main__":
    print(f"🚀 使用模型: {MODEL_TYPE.upper()} 开始训练...")
    train()

# model = EmotionCNN().to(DEVICE)  # or EmotionBiLSTM / EmotionTransformer

# train_loader = DataLoader(
#     EmotionFeatureDataset('dataset', split='train'),
#     batch_size=8, shuffle=True, num_workers=16, pin_memory=True
# )

# train_with_profiler(model, train_loader, device=DEVICE, epochs=1)