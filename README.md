①KaggleのKaggleの何のデータ（URLとデータの名前）を用いたのか。
＜使用したデータ＞
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
1872年から2017年までのナショナルチームにおけるサッカー国際試合の結果が入ったデータ（全部で4つのファイル）
その中でもshootouts.csvのデータを使用（PK戦の対戦データ）、項目は日付、home_team、away_team、winner、first_shooterの5つ

何を入力として何を推定するのか。
＜入力量＞
home_team,away_team,first_shooterの3つ
＜推定量＞
winner
ニューラルネットワークの構成（ニューロン数，層数など）
ニューロン数・・100,50
層数・・4

結果と考察
