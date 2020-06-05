import pandas as pd
import pdb
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
import pickle
import glob
from natsort import natsorted
from PIL import Image

options = Options()
options.add_argument('headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1280,1024')

driver = webdriver.Chrome(options=options)

df = pd.read_csv("../../corporationClassifier/data/data_all.csv")

link = df['link']

'''
for i in range(len(link)):
    driver = webdriver.Chrome(options=options)
    driver.get(link[i])
    s = str(i).zfill(5)
    driver.save_screenshot('../data/images/{0}.png'.format(s))
    driver.quit()
    print(i)
'''

'''
driver = webdriver.Chrome(options=options)
driver.get(link[5])
s = str(5).zfill(5)
driver.save_screenshot('../data/images/{0}.png'.format(s))
driver.quit()
print(5)
'''

image_title = []
Z = glob.glob('../data/images/images/*.png')
for str in natsorted(Z):
    image_title.append(str)
#pdb.set_trace()

df = pd.read_csv("../../corporationClassifier/data/data_all.csv") #load csv file

y = df['class'] #0 or 1
Y = np.array(y) #labelをnumpyに変換
pdb.set_trace()

trainRatio = 0.7
nTrain = np.floor(nData * trainRatio).astype(int)
nTest = nData - nTrain


# 学習データとテストデータ数
nTrain = np.floor(nData * trainRatio).astype(int)
nTest = nData - nTrain

# ランダムにインデックスをシャッフル
randInd = np.random.permutation(nData)

# 学習データ
xTrain = image_title[randInd[0:nTrain]]
y1Train = Y[randInd[0:nTrain]]

# 評価データ
xTest = image_title[randInd[nTrain:]]
y1Test = Y[randInd[nTrain:]]


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
        labels = Y[np.newaxis]
        flag = True
    else:
        # [Num.of files, height, width, channel]
        imgs = np.concatenate([imgs,np.array(img)[np.newaxis]])
        # 以下実装! labelデータをimgsと同じ要領で
        labels = np.concatenate([labels,Y[np.newaxis]])

with open('../data/out/data_z.pickle','wb') as f:
        pickle.dump(imgs,f)

#------データをpickleで保存
#with open('../data/out/data_z.pickle','wb') as f:
    #pickle.dump("../data/images/images",f)
