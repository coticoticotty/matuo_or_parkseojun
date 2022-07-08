# チョコレートプラネットの松尾似か、梨泰院クラスのパク・セロイ似かを判別する顔識別アプリ

## 実装すること
1. 画像をWebから持ってくる。（目標：400枚ずつ）→Selenium インスタから
2. 持ってきた画像の顔部分を抽出 → openCV
3. データセットの水増しをする 
4. 画像の学習を行う（FineTuning、モデルはどれを使うべき？） 
5. 用意した画像の顔部分を抽出し、類似度の判別を行う。

## 具体的に
1. seleniumでスクレイピングにしてみる。→スクレイピング環境の立ち上げにはようやく成功！次はGoogleで画像検索して保存するところ→Instagramから取るのがよさそうな気がする
参考URL：https://sasuwo.org/get-images-automatically-for-python/
https://gurutaka-log.com/python-imgs-scraping-opencv
2. https://qiita.com/taptappun/items/b4bc01f812704642d399
instagramからの画像ダウンロードに使えそうなライブラリ
3. https://instaloader.github.io/
Instaloaderで上手くいきそうなのにできない。Githubのissueに上がってた。
https://github.com/instaloader/instaloader/issues/1588
分かりやすい→https://python.plainenglish.io/scrape-everythings-from-instagram-using-python-39b5a8baf2e5

顔検出は、opencvの学習済みモデルの方が有能？？？

ViTを使用。
参考：https://github.com/sorabatake/article_20454_transformer/blob/main/VisionTransformer.ipynb
