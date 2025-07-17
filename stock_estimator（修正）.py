import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# データの読み込みと前処理
def read_stock_data():
    stock_csv = pd.read_csv("AAPL.csv", index_col=0)
    stock_data = stock_csv.copy()
    stock_data.columns = stock_data.columns.str.lower()

    # 'date'列をdatetime型に変換
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data['symbol'] = stock_data['symbol'].map({"aapl": 0})
    stock_data = stock_data.sort_values(by='date').reset_index(drop=True)

    return stock_data


def create_dataset_from_dataframe(data, target_tag="close"):
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)

    # 予測に使用する特徴量を定義
    # 当日のopenと前日のhigh, low, volume
    features_to_use = ['open', 'prev_high', 'prev_low', 'prev_volume']

    target_data = data[[target_tag]]

    data = data.dropna(subset=features_to_use + [target_tag])
    target_data = target_data.loc[data.index]

    input_data = data[features_to_use]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(input_data)
    y_scaled = y_scaler.fit_transform(target_data)

    # NumPy配列をPyTorchテンソルに変換
    input_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    return input_tensor, target_tensor, x_scaler, y_scaler, input_data.columns


# -------------------------------
# モデル定義
# -------------------------------
class StockRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        # 全結合層を定義
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1) # 出力は単一の値（終値）

    def forward(self, x):
        # 活性化関数ReLUを適用
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# モデル学習
def train_model(model, input_tensor, target_tensor, epochs=300):
    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    loss_fn = nn.MSELoss() # 平均二乗誤差

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

def evaluate_model(nn_model, input_tensor, target_tensor, y_scaler, dataset_name=""):
    nn_model.eval() 
    with torch.no_grad(): 
        
        prediction_scaled = nn_model(input_tensor).numpy()
        
        prediction_actual = y_scaler.inverse_transform(prediction_scaled)

        actual_scaled = target_tensor.numpy()
        actual_actual = y_scaler.inverse_transform(actual_scaled)

        mse = mean_squared_error(actual_actual, prediction_actual)
        rmse = mse ** 0.5
        mae = mean_absolute_error(actual_actual, prediction_actual)
        r2 = r2_score(actual_actual, prediction_actual)

        print(f"モデル評価 ({dataset_name}):")
        print(f"   MSE  = {mse:.4f}")
        print(f"   RMSE = {rmse:.4f}")
        print(f"   MAE  = {mae:.4f}")
        print(f"   R²   = {r2:.4f}")

        return prediction_actual, actual_actual

# 実行（学習とテスト）
if __name__ == "__main__":
    # データ準備
    stock_data = read_stock_data()
    input_tensor_full, target_tensor_full, x_scaler, y_scaler, feature_columns = create_dataset_from_dataframe(stock_data)

    # 訓練用とテスト用にデータを分割
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        input_tensor_full.numpy(), target_tensor_full.numpy(), test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)


    # モデル構築・学習
    model = StockRegressor(input_size=X_train_tensor.shape[1])
    print("--- モデル学習開始 ---")
    train_model(model, X_train_tensor, y_train_tensor) # 訓練データでモデルを学習
    print("--- モデル学習完了 ---")

    # テスト入力 (当日のopenと前日のhigh, low, volume)
    test_input_raw = [
        [
            150,        # 当日のopen
            155,        # 前日のhigh
            145,        # 前日のlow
            42000000    # 前日のvolume
        ]
    ]

    # 入力のスケーリング
    test_input_scaled = x_scaler.transform(test_input_raw)
    test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32)

    # 推論と逆スケーリング
    model.eval() # モデルを評価モードに設定
    with torch.inference_mode(): # 勾配計算を無効化
        pred_scaled = model(test_tensor).numpy()
        pred_actual = y_scaler.inverse_transform(pred_scaled)

    print(f"\n予測された終値: {pred_actual[0][0]:.2f} USD")

    # テストデータでモデルを評価
    print("\n--- テストデータでの評価 ---")
    predicted_test_actual, actual_test_actual = evaluate_model(model, X_test_tensor, y_test_tensor, y_scaler, "テストデータ")

    # 訓練データでモデルを評価
    #print("\n--- 訓練データでの評価 ---")
    #predicted_train_actual, actual_train_actual = evaluate_model(model, X_train_tensor, y_train_tensor, y_scaler, "訓練データ")


    # 予測値と実際の値のプロット (テストデータ)
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_test_actual, predicted_test_actual, alpha=0.7)
    # 理想的な予測を示すy=xの線
    plt.plot([actual_test_actual.min(), actual_test_actual.max()],
             [actual_test_actual.min(), actual_test_actual.max()], 'r--', lw=2)
    plt.xlabel("Actual (test data)")
    plt.ylabel("Predict (test data)")
    plt.title("Actual vs Predict(test data)")
    plt.grid(True)
    plt.show()
    #plt.savefig('actual_vs_predicted_close_prices_nn_test_lagged_features.png')
