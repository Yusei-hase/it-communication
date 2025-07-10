from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# データセットの定義
df = pd.read_csv('shootouts.csv')

# 後で表示するためにオリジナルの機能を保存する（予測例を表示するため）
X_original = df[['home_team', 'away_team', 'winner']].copy()

# いらない情報を消す
df_for_model = df.drop(columns=['date', 'first_shooter'])

# X特徴量とｙターゲット変数の定義hoge
# 'home_team'と'away_team'を特徴量として、'winner'をターゲット変数として使用
X = df_for_model[['home_team', 'away_team']]
y = df_for_model['winner']

# カテゴリー特徴をエンコードする
encoder_X = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder_X.fit_transform(X)

# 変数をエンコードする
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# テストセットの例について、元のデータを再構築するための分割の列を取得する。
indices = np.arange(len(df_for_model))
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_encoded, y_encoded, indices, test_size=0.2, random_state=42
)

# 4層順方向ニューラルネットワークモデルの定義
# 入力層、2つの隠れ層（それぞれ100ニューロンと50ニューロン）、出力層の構成
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, verbose=True)

# モデルの検証
print("Training the neural network model...")
mlp.fit(X_train, y_train)
print("Training complete.")

# テストデータを予測する
y_pred_encoded = mlp.predict(X_test)

# 予想と実際のラベルを元のチーム名に予測する
y_pred_decoded = encoder_y.inverse_transform(y_pred_encoded)
y_test_decoded = encoder_y.inverse_transform(y_test)

# テストデータにオリジナルのホームチームとアウェイチームを取得する
X_test_original = X_original.iloc[indices_test]
X_test_original['predicted_winner'] = y_pred_decoded
X_test_original['actual_winner'] = y_test_decoded

# 予測（一例）の表示
print("\nテストセットからの予測例:")
print(X_test_original.to_string())

# 計算精度の評価
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"\nモデルの全体的な精度: {accuracy:.4f}")