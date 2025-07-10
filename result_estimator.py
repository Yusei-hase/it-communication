import pandas as pd
import torch


def read_data() -> pd.DataFrame:
    # データの読み込み
    shootouts_csv = pd.read_csv("shootouts.csv", index_col=None, header=0)

    # NNで処理できるようにカテゴリカルデータを数値に変換
    # 全ての国名を列から抽出してユニークなものに変換
    all_countries = pd.unique(
    shootouts_csv[['home_team', 'away_team', 'winner']].values.ravel()
    )

# 国名に数値を割り当てる辞書を作成
    country_to_id = {country: idx for idx, country in enumerate(all_countries)}

    shootouts_data = shootouts_csv.copy()
    # 日付を datetime 型に変換
    shootouts_data["date"] = pd.to_datetime(shootouts_data["date"])

    #  年、月、日などを別の列に分ける
    shootouts_data["year"] = shootouts_data["date"].dt.year
    shootouts_data["month"] = shootouts_data["date"].dt.month
    shootouts_data["day"] = shootouts_data["date"].dt.day

    shootouts_data["home_team"] = shootouts_data["home_team"].map(country_to_id)
    shootouts_data["away_team"] = shootouts_data["away_team"].map(country_to_id)
    shootouts_data["winner"] = shootouts_data["winner"].map(country_to_id)
    shootouts_data["first_shooter"] = shootouts_data["first_shooter"].map(country_to_id)


    #shootouts_data["day"] = shootouts_data["day"].map({"Sun": 0, "Sat": 1, "Thur": 2, "Fri": 3})
    # 国名と対応する数値の一覧を表示
    for country, idx in country_to_id.items():
         print(f"{country}: {idx}")
    
    print(shootouts_data.head())


    # 数値の調整
    #shootouts_data["total_bill"] = shootouts_data["total_bill"] / 10

    # 変換後のデータの確認
    # print(tips_data.head())

    return shootouts_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(
    shootouts_data: pd.DataFrame, target_tag: str = "winner"
) -> tuple[torch.Tensor, torch.Tensor]:
    # "tip"の列を目的にする
    target = torch.tensor(shootouts_data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "tip"以外の列を入力にする
    input_data = shootouts_data.drop([target_tag, "date"], axis=1)
    #input = torch.tensor(shootouts_data.drop(target_tag, axis=1).values, dtype=torch.float32)
    input = torch.tensor(input_data.values, dtype=torch.float32)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = self.l3(h2)
        return o


def train_model(nn_model: FourLayerNN, input: torch.Tensor, target: torch.Tensor) -> None:
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=25として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=25, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(10000):
        # バッチごとに学習する
        for x, y_hat in train_loader:
            y = nn_model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回テストして誤差を表示
        if epoch % 1000 == 0:
            with torch.inference_mode():  # 推論モード（学習しない）
                y = nn_model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(epoch, loss)


# データの準備
shootouts_data = read_data()
input, target = create_dataset_from_dataframe(shootouts_data)

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
            2022,  # year
            12,  # month
            5,  # day
            96,  # home_team
            114,  # away_team
            96,  # first_shooter (home_team)
        ]
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))

