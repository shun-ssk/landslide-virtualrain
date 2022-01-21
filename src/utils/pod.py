# POD (Principal Orthogonal Decomposeition)固有直行分解

# 入力ファイル
#   配列サイズ=>m*n(m<n) numpy file
#   m=>観測地点数
#   n=>標本数

import numpy as np
from numpy.linalg import svd, matrix_rank
import matplotlib.pyplot as plt

#--------------------------
#  Constant Numbers
#--------------------------
UsageModeNum = 10


#--------------------------
#  Loading Files
#--------------------------
print("loading file...")
X = np.load("../input/utils/rain.npy")
loc = np.loadtxt("../raindata/coordinate.txt")

print("matrix size: ", X.shape)
print("rank: ", matrix_rank(X))


#--------------------------
#  Analyze
#--------------------------
# SVD (Singular Value Decomposition)特異値分解
U, S, Vt = svd(X, full_matrices=False)
print("\n ######## SVD result ######")
print("shape of U, S, Vt: ", U.shape, S.shape, Vt.shape)

# 再構築
reX = (U @ np.diag(S) @ Vt)

# 再構築誤差
### neX shoud be neary 0
neX = (X - reX).round(2)
print("\nNumerical error between X and reX: ", neX)

# 寄与率
CR = S**2
ConRate = CR / CR.sum()

# モード係数
Alp = (np.diag(S)@Vt).round(2)


#--------------------------
# output
#--------------------------
print("saveing results")
np.savetxt('../output/pod/txt/eigenValue.txt', U.round(4))
np.savetxt('../output/pod/txt/numericalError.txt', neX)
np.savetxt('../output/pod/txt/contributionRate.txt', ConRate.round(4))
np.savetxt('../output/pod/txt/modeCoffecient.txt', Alp)

# モードの空間構造を出力
for i in range(UsageModeNum):
    print("make img (mode"+str(i+1)+")")
    EigenVector = U[:,i]
    # 自分の研究対象地域に合うように調整
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111)
    scat = ax.scatter(loc[:,0], loc[:,1], s=300, marker=",", c=EigenVector, cmap="jet")
    plt.colorbar(scat)
    fig.savefig("../output/pod/png/mode/mode" + str(i+1) + ".png", dpi=600)

# 寄与率を出力
print("make img (contribution rate)")
sumRate = 0
ConRates = []
for rate in ConRate:
    sumRate += rate
    ConRates.append(sumRate)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot( [i for i in range(len(ConRate[:UsageModeNum]))], ConRate[:UsageModeNum])
fig.savefig("../output/pod/png/contributionRate.png", dpi=600)

# 平均降雨強度と第1モード係数との関係
x_aves = []
coff1s = []
for i in range(X.shape[-1]):
    x = X[:,i]
    coff = Alp[:,i]
    x_ave = np.average(x)
    x_aves.append(x_ave)
    coff1s.append(coff[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_aves, coff1s)
fig.savefig("output/png/intencityAndMode1.png", dpi=600)

# 第nモードまでで再構築した時のエラー率
err_aves = []
mode_x = [i+1 for i in range(UsageModeNum)]
for i in range(UsageModeNum):
    modes = U[:,:i]
    coffs = Alp[:i,:]
    X_hat = modes @ coffs
    err = np.abs( X - X_hat )
    err_aves.append( np.average(err) )
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mode_x, err_aves)
fig.savefig("output/png/errorAverage.png", dpi=600)

# 平均降雨強度のヒストグラム
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(x_aves, bins=30, range=(0,12))
plt.show()
