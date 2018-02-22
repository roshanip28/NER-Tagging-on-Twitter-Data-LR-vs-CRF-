import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
means_frank = (92.33,37.78,26.40,31.08)
means_guido = (92.37,44.09,31.83,36.97)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Log-Reg')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='y',
                 label='CRF')

plt.xlabel('Labels')
plt.ylabel('FB1 Score')
plt.title('Log-Reg Vs CRF for twitter_dev_test.ner (With additional features)')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall','FB1'))
plt.legend()

plt.tight_layout()
plt.show()