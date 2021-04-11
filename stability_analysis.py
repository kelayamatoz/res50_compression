import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

n_iters = 10

def get_all_data(n):
    r = []
    for i in range(n):
        data = pd.read_csv("cuda_{}.csv".format(i))
        runtime_ms = data.iloc[:, -1:].to_numpy().flatten()
        r.append(runtime_ms)
    return r

all_data = get_all_data(10)
layers = [_ for _ in range(len(all_data[0]))]
for idx, data in enumerate(all_data):
    print(idx)
    plt.plot(layers, data, label='run_{}'.format(idx))
plt.legend(loc="upper right")

plt.savefig('result.pdf')

import code
code.interact(local=locals())