import pdb
import pandas as pd
import pdb
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
import glob
from natsort import natsorted
from PIL import Image
import pickle

image_title = []
Z = glob.glob('../data/images/*.png')
for str in natsorted(Z):
    image_title.append(str)

df = pd.read_csv("../../corporationClassifier/data/data_all.csv") #load csv file
y = df['class'] #0 or 1

Y = np.array(y) #labelをnumpyに変換
image_title = np.array(image_title)
# 全データ数
nData = len(Z)
# 学習データの割合
trainRatio = 0.7

# 学習データとテストデータ数
nTrain = np.floor(nData * trainRatio).astype(int)
nTest = nData - nTrain

# ランダムに全データのインデックスをシャッフル
randInd = np.random.permutation(nData).tolist()

# 学習データ(入力)

xTrain = image_title[:10]
yTrain = Y[:10]

xTest = image_title[10:15]
yTest = Y[10:15]


"""
xTrain = image_title[randInd[0:nTrain]]
# 以下実装! 学習データ(出力、正解データ)
yTrain = Y[randInd[0:nTrain]]
# テストデータ
xTest = image_title[randInd[nTrain:]]
# 以下実装! テストデータ(出力、正解データ)
yTest = Y[randInd[nTrain:]]
"""
# 学習データ保存 -----------------------------------------------------------------
label_list = []
flag = False
for i,j in zip(xTrain,yTrain): # i:入力データ, j:出力データ
    # 元となる画像の読み込み
    img = Image.open(i)
    #オリジナル画像の幅と高さを取得
    width, height = img.size
    # オリジナル画像と同じサイズのImageオブジェクトを作成する
    img2 = Image.new('RGB', (width, height))
    if not flag:
        imgs = np.array(img)[np.newaxis]
        # 以下実装! labelデータをimgsと同じ要領で
        label_list = np.append(label_list,j)
        flag = True
    else:
        # [Num.of files, height, width, channel]
        imgs = np.concatenate([imgs,np.array(img)[np.newaxis]])
        # 以下実装! labelデータをimgsと同じ要領で
        label_list = np.append(label_list,j)

# 学習データを保存 imgs,labels
with open('../data/out/data_z.pickle','wb') as f:
        pickle.dump(imgs,f)
        pickle.dump(label_list,f)


# 以下に、学習データ(xTrain)と同じ要領でテストデータ(xTest)をpickle保存 ------
# 学習データ保存 -----------------------------------------------------------------
test_label_list = []
flag = False
for i,j in zip(xTest,yTest): # i:入力データ, j:出力データ
    # 元となる画像の読み込み
    img = Image.open(i)
    #オリジナル画像の幅と高さを取得
    width, height = img.size
    # オリジナル画像と同じサイズのImageオブジェクトを作成する
    img2 = Image.new('RGB', (width, height))
    if not flag:
        imgs = np.array(img)[np.newaxis]
        # 以下実装! labelデータをimgsと同じ要領で
        test_label_list = np.append(test_label_list,j)
        flag = True
    else:
        # [Num.of files, height, width, channel]
        imgs = np.concatenate([imgs,np.array(img)[np.newaxis]])
        # 以下実装! labelデータをimgsと同じ要領で
        test_label_list = np.append(test_label_list,j)

#pdb.set_trace()
#------データをpickleで保存
with open('../data/out/data_w.pickle','wb') as f:
        pickle.dump(imgs,f)
        pickle.dump(test_label_list,f)
