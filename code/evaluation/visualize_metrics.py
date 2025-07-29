# Copyright 2025 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Load all scores
score_fpath = "HowToDIVEvals/scores.json"
with open(score_fpath, "r") as json_file:
  score_list = json.load(json_file)

#BLEU Score histogram
bleu_score_list = []
for file in score_list:
  bleu_score_list.extend(score_list[file]['bleu_score'])
sns.set_style("whitegrid")
data = bleu_score_list
plt.figure(figsize=(10, 6))
sns.histplot(
    data,
    bins=30,
    kde=True,
    color='yellowgreen',
    edgecolor='black',
    alpha=0.7,
    line_kws={'linewidth': 2, 'color': 'darkblue'}
)
plt.xlabel('BLEU Score', fontsize=14)
plt.ylabel('Count of Turns', fontsize=14)
plt.title('Histogram of BLEU Scores', fontsize=16, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12)
mean_value = np.mean(data)
plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.legend(fontsize=12)
sns.despine(top=True, right=True)
plt.show()

# Turn-wise BLEU Score analysis
turnwise_bleu_score_list = [0]*10
for file in score_list:
  for idx in range(len(score_list[file]['bleu_score'])):
    turnwise_bleu_score_list[idx] += score_list[file]['bleu_score'][idx]
turnwise_bleu_score_list = [sc/len(score_list) for sc in turnwise_bleu_score_list]
data = turnwise_bleu_score_list
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8, markeredgecolor='darkblue', markerfacecolor='white')
ax.set_title('Variation of BLEU Score with Turn', fontsize=16, fontweight='bold')
ax.set_xlabel('Turn Number', fontsize=12)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.show()

#Task-wise Score Analysis
taskwise_bleu_score_list = defaultdict(list)
for file in score_list:
  task = file.split('/')[-2]
  if 'parse' in task or 'mod' in task:
    task = task.split('_')[0]
  taskwise_bleu_score_list[task].extend(score_list[file]['bleu_score'])

taskwise_avg_bleu_score = {}
for task in taskwise_bleu_score_list:
  taskwise_avg_bleu_score[task] = np.mean(taskwise_bleu_score_list[task])

values = list(taskwise_avg_bleu_score.values())
plt.figure(figsize=(10, 6))
plt.bar(list(taskwise_avg_bleu_score.keys()), values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'darkgrey', 'b', 'coral'])
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('BLEU Scores for different tasks', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a subtle grid
plt.tight_layout()
plt.show()

# Parse vs Non-Parse data scores
category_wise_bleu_list = defaultdict(list)
category_wise_llmj_list = defaultdict(list)
for file in score_list:
  task = file.split('/')[-2]
  if 'parse' in task:
    category_wise_bleu_list['parse'].extend(score_list[file]['bleu_score'])
    category_wise_llmj_list['parse'].extend(score_list[file]['llmj_score'])
  else:
    category_wise_bleu_list['non-parse'].extend(score_list[file]['bleu_score'])
    category_wise_llmj_list['non-parse'].extend(score_list[file]['llmj_score'])

  if 'mod' in task:
    category_wise_bleu_list['mod'].extend(score_list[file]['bleu_score'])
    category_wise_llmj_list['mod'].extend(score_list[file]['llmj_score'])
  else:
    category_wise_bleu_list['non-mod'].extend(score_list[file]['bleu_score'])
    category_wise_llmj_list['non-mod'].extend(score_list[file]['llmj_score'])

data = {
    'Criteria': ['Parse', 'Normal'],
    'BLEU': [np.mean(category_wise_bleu_list['parse']), np.mean(category_wise_bleu_list['non-parse'])],
    'LLM as Judge': [np.mean(category_wise_llmj_list['parse']), np.mean(category_wise_llmj_list['non-parse'])],
}
df = pd.DataFrame(data)
df_melted = df.melt(id_vars='Criteria', var_name='Category', value_name='Score')
print('Criteria: ', data['Criteria'])
print('BLEU: ', data['BLEU'])
print('LLM as Judge: ', data['LLM as Judge'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Criteria', y='Score', hue='Category', data=df_melted, palette='viridis')
plt.title('Scores for Short vs Normal Turns', fontsize=16)
plt.xlabel('Criteria', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Category', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
