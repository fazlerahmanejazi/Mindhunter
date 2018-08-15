# Loading libraries
import sys
import pandas as pd
import numpy as np

res = pd.read_csv('/home/fazle/Desktop/Innovation Lab/CS299_13_43/Source FIles - MindHunter/Random Decision Forest/res.csv', index_col=False)
test = pd.read_csv('/home/fazle/Desktop/Innovation Lab/CS299_13_43/Source FIles - MindHunter/data/test.csv', index_col=False)

res = res.drop('Id', 1)
test = test['Category']

nlargest = 6
order = np.argsort(-res.values, axis=1)[:, :nlargest]
result = pd.DataFrame(res.columns[order], columns=['top{}'.format(i) for i in range(1, nlargest+1)])

cnt = 0

try:
    x = int(sys.argv[1])
except IndexError:
    x = 3

a = [result['top1'], result['top2'], result['top3'], result['top4'], result['top5'], result['top6']]
for i in range(1, len(res)):
    for j in range(0, x):
        if a[j][i]==test[i]:
            cnt = cnt+1
print (cnt/len(res))
