①Kaggleの何のデータ（URLとデータの名前）を用いたのか。

＜使用したデータ＞

2015年から2020年までのアメリカにおけるApple社の株価データ（csv）

＜csvデータのカラム＞

close - 終値

high - 当日の最高値

low - 当日の最安値

open - 当日の始値

volume - 株式の取引量

adjClose - 他の株価属性/アクションとの関連における終値

adjHigh - 他の株価属性/アクションとの関連における最高値

adjOpen - 他の株価属性/アクションとの関連における始値

adjVolume - 他の株価属性/アクションとの関連における取引量

divCash - 現金配当

splitFactor - 株式分割

https://www.kaggle.com/datasets/suyashlakhani/apple-stock-prices-20152020?select=AAPL.csv

②何を入力として何を推定するのか。

＜入力量＞

・前日のLow,High,volume,当日のopen

＜推定量＞

・close

③ニューラルネットワークの構成（ニューロン数，層数など）

・ニューロン数・・10,32,32,1

入力層（特徴量の数である4）→隠れ層1（32）→隠れ層2（32）→出力層（目的変数なので1）

活性化関数（ReLU）

・層数・・3層の全結合層

入力層＋隠れ層2つ＋出力層＝計4層

④結果と考察

＜結果＞

・Epoch 0, Loss: 0.397222

　Epoch 50, Loss: 0.002327

　Epoch 100, Loss: 0.001903

　Epoch 150, Loss: 0.001566

　Epoch 200, Loss: 0.002312

　Epoch 250, Loss: 0.001493

　Epoch 299, Loss: 0.001468

・予測された終値: 152.73 USD

・モデル評価（テストデータ）: MSE = 6.2661

　　　　　　                RMSE = 2.5032

　　　　　　                MAE = 1.6863

　　　　　　                 R² = 0.9981

＜考察＞

・予測精度が良すぎる（そもそもデータが線形なので、それが原因かも）

・
