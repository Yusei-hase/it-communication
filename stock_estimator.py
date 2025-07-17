import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# データの読み込みと前処理
# -------------------------------
def read_stock_data():
    stock_csv = pd.read_csv("AAPL.csv", index_col=0)
    stock_data = stock_csv.copy()

    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data['symbol'] = stock_data['symbol'].map({"AAPL": 0})
    return stock_data


def create_dataset_from_dataframe(data, target_tag="close"):
    input_data = data.drop([target_tag, "date"], axis=1)

    # スケーリング（入力とターゲット両方）
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(input_data)
    y_scaled = y_scaler.fit_transform(data[[target_tag]])

    input_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    return input_tensor, target_tensor, x_scaler, y_scaler


# -------------------------------
# モデル定義
# -------------------------------
class StockRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# -------------------------------
# モデル学習
# -------------------------------
def train_model(model, input_tensor, target_tensor, epochs=300):
    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.inference_mode():
                val_pred = model(input_tensor)
                val_loss = loss_fn(val_pred, target_tensor)
                print(f"Epoch {epoch}, Loss: {val_loss.item():.6f}")

def evaluate_model(nn_model, input, target):
    with torch.no_grad():
        prediction = nn_model(input).numpy()
        actual = target.numpy()

        mse = mean_squared_error(actual, prediction)
        rmse = mse ** 0.5
        mae = mean_absolute_error(actual, prediction)
        r2 = r2_score(actual, prediction)

        print(f"モデル評価:")
        print(f"  MSE  = {mse:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R²   = {r2:.4f}")

# -------------------------------
# 実行（学習とテスト）
# -------------------------------
if __name__ == "__main__":
    # データ準備
    stock_data = read_stock_data()
    input_tensor, target_tensor, x_scaler, y_scaler = create_dataset_from_dataframe(stock_data)

    # モデル構築・学習
    model = StockRegressor(input_size=input_tensor.shape[1])
    train_model(model, input_tensor, target_tensor)

    # テスト入力
    test_input_raw = [
        [
            0,     # symbol
            150,   # open
            155,   # high
            145,   # low
            42000000,  # volume
            110,   # adjClose
            112,   # adjHigh
            108,   # adjLow
            150,   # adjOpen
            36003540,  # adjVolume
            0,     # divCash
            1,     # splitFactor
        ]
    ]

    # 入力のスケーリング
    test_input_scaled = x_scaler.transform(test_input_raw)
    test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32)

    # 推論と逆スケーリング
    model.eval()
    with torch.inference_mode():
        pred_scaled = model(test_tensor).numpy()
        pred_actual = y_scaler.inverse_transform(pred_scaled)

    print(f"予測された終値: {pred_actual[0][0]:.2f} USD")
    evaluate_model(model, input_tensor, target_tensor)



