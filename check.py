import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ks2018.csv")

num_proj = 0
num_success = 0
num_failed = 0
num_canceled = 0
num_live = 0
others = []

for i in range(0,len(df["ID"])-1):

	num_proj += 1

	state = df["state"][i]

	if(state == "successful"):
		num_success += 1
	elif(state == "failed"):
		num_failed += 1



print("Total:", num_proj, "successful:", num_success, "failed:", num_failed, "canceled:", num_canceled)
print("All usable projects:" , num_success +  num_failed )

labels = ("Successful", "Failed", "Canceled")
values = (num_success, num_failed, num_canceled)
colors = ("green", "red", "orange")
explode = (0.1,0,0)

plt.style.use('ggplot')

plt.figure(1, figsize=(15,10))
plt.pie(values,labels=labels,colors = colors,autopct='%1.1f%%', startangle=140, explode=explode)
plt.savefig("pie.png")
