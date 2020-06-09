import numpy as np
import matplotlib.pyplot as plt
import copy
def rand_sample(m, n):
    label = np.ones(100)
    label[-40:] = -1

    score = np.zeros(100)
    score[:60] = 0.7
    score[-40:] = 0.3
    # 随机挑选m n修改值
    index_pos = np.random.choice(np.arange(60), size=m, replace=False)
    index_neg = np.random.choice(np.arange(60, 100), size=n, replace=False)

    index = np.concatenate((index_pos, index_neg))
    score[index] = np.random.random(m+n)
    return score, label

def FPR_TPR(thr,s,l):
    score = copy.deepcopy(s)
    label = copy.deepcopy(l)
    N_neg = label[label == -1].size
    N_pos = label[label == 1].size

    score[score>thr] = 1
    score[score<=thr] = -1

    TP = (score > 0) * (label > 0)
    FP = (score > 0) * (label < 0)

    TP_num = TP[TP].size
    FP_num = FP[FP].size

    FPR = FP_num / N_neg
    TPR = TP_num / N_pos

    return FPR, TPR

for m, n in [[50,30], [40,20], [35,20], [35,15], [20,20], [10,10], [30, 5], [30, 30], [45, 20], [10, 35]]:
    s, l = rand_sample(m, n)
    thr_points = np.linspace(0, 1, 1000) 
    FPR_points = [] 
    TPR_points = []
    for thr in thr_points:
        FPR, TPR = FPR_TPR(thr, s, l)
        FPR_points.append(FPR)
        TPR_points.append(TPR)
    plt.plot(FPR_points, TPR_points, label='m={},n={}'.format(m,n))
    plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("FPR-TPR")


















