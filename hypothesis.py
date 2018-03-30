import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t
from random import randint



df = pd.read_csv("ks2018.csv")
df = df[df.goal < 200000]
df = df.sample(n=2000)

alpha = 0.05

del df["ID"]
del df["name"]
del df["category"]
del df["main_category"]
del df["pledged"]
del df["usd_pledged_real"]
del df["usd pledged"]
del df["deadline"]
del df["launched"]
del df["currency"]
del df["backers"]
del df["goal"]
del df["country"]

df_s = df[df.state == "successful"]

df_f = df[df.state == "failed"]


print(df_s)
print(df_f)
tvalue,pvalue = stats.ttest_ind(df_s["usd_goal_real"],df_f["usd_goal_real"], equal_var=False)

print("Pvalue:",pvalue)
print("Tvalue", tvalue)

if pvalue <= alpha:
    # we reject null hypothesis
    print ('Null hypothesis is rejected')
else:
    # we reject alternative hypothesis
    print ('Null hypothesis cannot be rejected')


deg_freedom = df_s["usd_goal_real"].shape[0] - 1

print(np.var(df_s["usd_goal_real"]))
print(np.var(df_f["usd_goal_real"]))

t_dist = t.stats(deg_freedom)

print(deg_freedom)
print(t_dist)

critical_value = stats.t.ppf(q = 1 - alpha, df = deg_freedom)

plt.style.use('ggplot')
x = np.linspace(-10, 10, 1000)
plt.plot(x, stats.t.pdf(x,deg_freedom))
plt.annotate('Critical Value = {0:.2f}'.format(critical_value), xy=(critical_value, 0.10), xytext=(critical_value, 0.12),
        arrowprops=dict(facecolor='black', shrink=0.5), verticalalignment='bottom' )
plt.annotate('Current Value = {0:.2f}'.format(tvalue), xy=(tvalue, 0.02), xytext=(tvalue, 0.01))
plt.fill_between(x, stats.t.pdf(x,deg_freedom), where= x > critical_value, facecolor='grey', interpolate=True)
plt.show()
