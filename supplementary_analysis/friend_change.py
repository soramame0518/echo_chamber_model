import pandas as pd
import json
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib


def read_in_data(f):
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)  
    return df

def ratio(t1, t2, f1, f2):
    if (t2 - t1) == 0:
      return None
    tmp = (f2 - f1) / float((t2 - t1))
    if tmp > 1e308 or tmp < -1e308: # inf check
      return None
    return tmp

def compute_changes(g_df):
  change_over_time_lst = []
  change_over_tweet_count_lst = []
  number_used_ids = 0
  for iid, sub_df in g_df:
      if len(sub_df) < 2:
          continue
      number_used_ids += 1
      sub_df = sub_df.sort_values(by='timestamp', ascending=True)
      sub_df = sub_df.reset_index()
      ratio1 = []
      ratio2 = []
      for i in range(len(sub_df) - 1):
          t1 = time.mktime(sub_df.iloc[i]['timestamp'].timetuple()) / (24*3600.0)  # represented as days
          t2 = time.mktime(sub_df.iloc[i+1]['timestamp'].timetuple()) / (24*3600.0)
          fr1 = sub_df.iloc[i]['num_friends']
          fr2 = sub_df.iloc[i+1]['num_friends']
          c1 = sub_df.iloc[i]['tweet_count']
          c2 = sub_df.iloc[i+1]['tweet_count']
          tmp_ratio1 = ratio(t1, t2, fr1, fr2)
          tmp_ratio2 = ratio(c1, c2, fr1, fr2)
          if tmp_ratio1:
            ratio1.append(tmp_ratio1)
          if tmp_ratio2:
            ratio2.append(tmp_ratio2)
      if ratio1:
          change_over_time_lst.append(np.mean(ratio1))
      if ratio2:
          change_over_tweet_count_lst.append(np.mean(ratio2))
  return (np.array(change_over_time_lst),
          np.array(change_over_tweet_count_lst),
          number_used_ids)

def clean_stat(arr):
    print(arr)
    print('=========')
    tmp_arr = arr[~np.isnan(arr)]
    # remove inf
    tmp_arr = tmp_arr[tmp_arr < 1e308]
    tmp_arr = tmp_arr[tmp_arr > -1e308]
    # only use numbers in 99% CI
    a_min, a_max = st.t.interval(0.99, len(tmp_arr)-1, loc=np.mean(tmp_arr), scale=st.sem(tmp_arr))
    tmp_arr = tmp_arr[tmp_arr > a_min]
    tmp_arr = tmp_arr[tmp_arr < a_max]
    return tmp_arr

df = read_in_data('tweet_summary')
g_df = df.groupby('id')
change_over_time_lst, change_over_tweet_count_lst, number_used_ids = compute_changes(g_df)
change_over_tweet_count_lst = clean_stat(change_over_tweet_count_lst)
change_over_time_lst = clean_stat(change_over_time_lst)

font = {'size': 15}
matplotlib.rc('font', **font)
fig, axes = plt.subplots(1,2,sharey=True, figsize=(15,5))

weights = np.ones_like(change_over_tweet_count_lst) / len(change_over_tweet_count_lst)
axes[0].hist(
  change_over_tweet_count_lst, bins=20, color='royalblue', weights=weights)

weights = np.ones_like(change_over_time_lst) / len(change_over_time_lst)
axes[1].hist(
  change_over_time_lst, bins=20, color='royalblue', weights=weights)
axes[0].locator_params(axis='x', nbins=10)
axes[1].locator_params(axis='x', nbins=10)
axes[0].set_ylabel('Frequency', size=20)
axes[0].set_xlabel(r'$\Delta f/tweet$', size=20)
axes[1].set_xlabel(r'$\Delta f/day$', size=20)
# axes[0].ticklabel_format(axis='x', useMathText=True)
plt.tight_layout()
plt.savefig('friend_change_days.pdf')

