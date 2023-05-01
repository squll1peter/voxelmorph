from utils import labelDicePath, densewarpPath
import random, os
import json
import numpy as np

model_epoch_list = [0, 300,500,880, 1000, 1500, 2000]
oasis_case_list = [1,2,3,4,5,6,7,9,10,11]
OASIS_PATH = "data/OASIS/"
RESULT_PATH = "result/"
random.seed(0)
case_pairs = [random.sample(oasis_case_list, k=2) for i in range(0,10)]
print(case_pairs)
pair_pth = {}
dices = {}
def get_pths(fixed_id, moving_id, OASIS_PATH, RESULT_PATH, model_epoch, pair_id):
    fixed_id = pair[0]
    moving_id = pair[1]
    moving_pth = os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(moving_id),  "aligned_norm.nii.gz")
    fixed_pth =  os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(fixed_id),  "aligned_norm.nii.gz")
    moved_pth =  os.path.join(RESULT_PATH, "Pair{:04d}".format(pair_id), f"{model_epoch:04d}", "aligned_norm.nii.gz")
    moving_seg35_pth = os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(moving_id), "aligned_seg35.nii.gz")
    fixed_seg35_pth  = os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(fixed_id),  "aligned_seg35.nii.gz")
    moved_seg35_pth  = os.path.join(RESULT_PATH, "Pair{:04d}".format(pair_id), f"{model_epoch:04d}" , "aligned_seg35.nii.gz")
    moving_seg4_pth = os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(moving_id), "aligned_seg4.nii.gz")
    fixed_seg4_pth  = os.path.join(OASIS_PATH, "OASIS_OAS1_{:04d}_MR1".format(fixed_id),  "aligned_seg4.nii.gz")
    moved_seg4_pth  = os.path.join(RESULT_PATH, "Pair{:04d}".format(pair_id), f"{model_epoch:04d}" , "aligned_seg4.nii.gz")
    warp_pth  = os.path.join(RESULT_PATH, "Pair{:04d}".format(pair_id), f"{model_epoch:04d}" , "warp.nii.gz")
    model_pth = os.path.join("models", "{:05d}.h5".format(model_epoch))
    pair_pth = {"epoch":model_epoch,
                "moving_pth":moving_pth, 
                "fixed_pth":fixed_pth, 
                "moved_pth":moved_pth, 
                "moving_seg35_pth":moving_seg35_pth, 
                "fixed_seg35_pth":fixed_seg35_pth, 
                "moved_seg35_pth":moved_seg35_pth, 
                "moving_seg4_pth":moving_seg4_pth, 
                "fixed_seg4_pth":fixed_seg4_pth, 
                "moved_seg4_pth":moved_seg4_pth, 
                "warp_pth":warp_pth,
                "model_pth":model_pth}
    return pair_pth

if not os.path.isfile("result.json"):
    from register_and_evaluate import register128
    for epoch in model_epoch_list:
        pair_pth[epoch] = {}
        dices[epoch] = {}
        for i, pair in enumerate(case_pairs):
            print("epoch {}, pair {}".format(epoch, i))
            pths = get_pths(pair[0], pair[1], OASIS_PATH, RESULT_PATH, epoch, i)
            pair_pth[epoch][i] = pths
            if(not os.path.isdir(os.path.dirname(pths["warp_pth"]))):
                os.makedirs(os.path.dirname(pths["warp_pth"]))
            if(not os.path.isfile(pths["warp_pth"]) or not os.path.isfile(pths["moved_pth"])):
                register128(pths["moving_pth"], pths["fixed_pth"], pths["moved_pth"],  pths["model_pth"], pths["warp_pth"], gpu=None, multichannel=False, method="shrink")
            dices[epoch][i] = labelDicePath(pths["fixed_seg35_pth"], densewarpPath(pths["moving_seg35_pth"], pths["warp_pth"], pths["moved_seg35_pth"], interp_order=0))

    with open("result.json", "w") as outfile:
        json.dump(dices, outfile)
else:
    with open("result.json", "r") as outfile:
        dices = json.load(outfile)

dices2 = {}
for i, pair in enumerate(case_pairs):
    pths = get_pths(pair[0], pair[1], OASIS_PATH, RESULT_PATH, 0, i)
    dices2[i] = labelDicePath(pths["fixed_seg35_pth"], pths["moving_seg35_pth"])

with open("result2.json", "w") as outfile:
    json.dump(dices2, outfile)



def print_pretty_list(list):
    for i in list:
        print(f"{i} ", end='')
    print("")
# ========================

with open("result2.json", "w") as outfile:
    json.dump(dices2, outfile)

import pandas as pd
df = pd.read_csv("data/OASIS/seg35_labels.txt",sep='\s+', header=0)
label_list = df['Label'].tolist()
cols = ["epoch", "Pair"]+label_list
df = pd.DataFrame(columns=cols)



pair_avg_dice = np.zeros([36])
for i, pair in dices2.items():
    row = {"epoch":"Unregistered", "Pair":i}
    for i, label in enumerate(label_list):
        row[label] = pair[i]
    df.loc[len(df)] = row

for idx, epoch_pairs in dices.items():
    pair_avg_dice = np.zeros([36])
    for i, pair in epoch_pairs.items():
        row = {"epoch":f"{idx}", "Pair":i}
        for i, label in enumerate(label_list):
            row[label] = pair[f"{i}"]
        df.loc[len(df)] = row

print(df)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
df2=pd.melt(df, id_vars=['epoch', 'Pair'], value_vars=label_list,
              var_name='label', value_name='Dice')
flierprops = dict(marker='x', markersize=4,
                  markeredgecolor='none')
boxprops = dict(linewidth=0)
sns.boxplot(x="label", y="Dice", hue="epoch", data=df2, palette="Set1", width=1, linewidth=0, boxprops=boxprops, flierprops=flierprops)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.gca().set_aspect(6)
plt.legend(bbox_to_anchor =(0.65, 1.25), fontsize=6)
plt.savefig('result.png', dpi=600)
