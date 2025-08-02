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


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools


def get_ue_from_dialogues(cur_dialogue):
  ue_text, next_turn_window = cur_dialogue.split('\n\n')
  expert_turn_text = ue_text.split('\nExpert: ')[-1]
  user_turn_text = ue_text.split('\nExpert: ')[0].split('\nUser: ')[-1]
  return next_turn_window, user_turn_text, expert_turn_text

# Load all ids
test_split_fpath = 'HowToDIV/data/test_ids.txt'
with open(test_split_fpath, 'r') as file:
  test_ids = file.readlines()

train_split_fpath = 'HowToDIV/data/train_ids.txt'
with open(train_split_fpath, 'r') as file:
  train_ids = file.readlines()

all_ids = test_ids + train_ids
all_ids = [f.strip() for f in all_ids]
len(all_ids)

# Load dialogues and video data for all turns
file_to_turns = {}
file_to_dial_lens = {}
file_to_video_lens = {}
for file_idx, file_name in enumerate(all_ids):
  fpath = file_name
  print(file_idx, file_name)
  with open(fpath, 'r') as file:
    file_content = file.read()

  instructions, dialogues = file_content.split('\n\nDialogue\n')
  instructions = instructions.split('Instructions\n')[-1].split('\n')
  dialogues = dialogues.split('\nTurn ')
  num_turns = len(dialogues)-1
  file_to_turns[file_name] = num_turns

  file_to_dial_lens[file_name] = {'user': [], 'expert': []}
  file_to_video_lens[file_name] = [dialogues[0]]
  for i in range(1, len(dialogues)):
    next_window, user_text, expert_text = get_ue_from_dialogues(dialogues[i])
    num_user_words = len(user_text.split(' '))
    num_expert_words = len(expert_text.split(' '))
    file_to_dial_lens[file_name]['user'].append(num_user_words)
    file_to_dial_lens[file_name]['expert'].append(num_expert_words)
    file_to_video_lens[file_name].append(next_window)

# histogram for number of turns per conversation
turns_counter = defaultdict(int)
taskwise_turn_counter = defaultdict(int)
taskwise_session_counter = defaultdict(int)
for file in file_to_turns:
  num_turns = file_to_turns[file]
  turns_counter[num_turns] += 1
  task = file.split('/')[-2]
  taskwise_turn_counter[task] += num_turns
  taskwise_session_counter[task] += 1

turns_list = []
for key in turns_counter:
  turns_list.extend([key] * turns_counter[key])
print('Total count of turns in dataset: ', sum(turns_list))
print('Total number of sessions: ', len(turns_list))
print(f'Average number of turns per session: {np.mean(turns_list)}, median:'
      f'{np.median(turns_list)}')

plt.bar(turns_counter.keys(), turns_counter.values(), width=1.0, color='g')
plt.title('Histogram for Count of turns per session')
plt.xlabel('Number of Turns')

# pie-chart for task-wise distribution
modifiers = ['mod', 'parse']
keys_to_del = set()
for key in taskwise_turn_counter:
  if any(m in key for m in modifiers):
    base_key = key.split('_')[0]
    taskwise_turn_counter[base_key] += taskwise_turn_counter[key]
    taskwise_session_counter[base_key] += taskwise_session_counter[key]
    keys_to_del.add(key)
for key in keys_to_del:
  del taskwise_turn_counter[key]
  del taskwise_session_counter[key]

taskwise_turn_counter['repot plant'] = taskwise_turn_counter.pop('repot')
taskwise_turn_counter['change tire'] = taskwise_turn_counter.pop(
    'changing_tire')
taskwise_turn_counter['jump-start car'] = taskwise_turn_counter.pop('jump_car')

taskwise_session_counter['repot plant'] = taskwise_session_counter.pop('repot')
taskwise_session_counter['change tire'] = taskwise_session_counter.pop(
    'changing_tire')

taskwise_session_counter['jump-start car'] = taskwise_session_counter.pop(
    'jump_car')

sorted_keys = sorted(taskwise_turn_counter.keys())[::-1]
sorted_dict = {key: taskwise_turn_counter[key] for key in sorted_keys}
labels = sorted_keys
sizes = sorted_dict.values()
plt.subplots(figsize=(8, 8))
colors = sns.color_palette('Set2')

plt.pie(sizes, labels=labels, colors=colors, shadow=True, autopct='%1.1f%%',
        startangle=90, radius=1.2, textprops={'fontsize': 18})
plt.title('Data split across procedural task turns', fontsize=22, y=1.05)
plt.axis('equal')
plt.show()


# Duration histogram
def time_to_seconds(t):
  t = t.split(',')[0]
  h, m, s = map(int, t.split(':'))
  return h * 3600 + m * 60 + s

file_to_video_lens_nos = {}
for file in file_to_video_lens:
  intervals_str = file_to_video_lens[file]
  intervals = []
  for s in intervals_str:
    if s.strip():  # skip empty strings
      if '\n' in s:
        s = s.split('\n')[0]
      t1, t2 = s.strip('[]').split(', ')
      if 'NIV' not in file:
        intervals.append([float(t1), float(t2)])
      else:
        intervals.append([time_to_seconds(t1), time_to_seconds(t2)])
  file_to_video_lens_nos[file] = intervals

# Concat durations into single list
durations = list(file_to_video_lens_nos.values())
all_times = list(itertools.chain(*durations))
all_durations = [
    max(1, interval[1] - interval[0])
    for interval in all_times
    if (interval[1] - interval[0] < 120)
]

# Video length per task
modifiers = ['mod', 'parse']
taskwise_video_length = defaultdict(int)
for file in file_to_video_lens_nos:
  task = file.split('/')[-2]
  if any(m in task for m in modifiers):
    task = task.split('_mod')[0].split('_parse')[0]
  intervals = file_to_video_lens_nos[file]
  for interval in intervals:
    dur = interval[1] - interval[0]
    taskwise_video_length[task] += dur/60


taskwise_video_length['repot plant'] = taskwise_video_length.pop('repot')
taskwise_video_length['change tire'] = taskwise_video_length.pop(
    'changing_tire')
taskwise_video_length['jump-start car'] = taskwise_video_length.pop('jump_car')

total_duration = sum(taskwise_video_length.values())
print(f'Total duration: {total_duration},  mins i.e. {total_duration/60} hours')

print(taskwise_video_length)

# chart for taskwise data details
avg_task_length = {}
for task in taskwise_turn_counter:
  total_num_turns = taskwise_turn_counter[task]
  total_num_sessions = taskwise_session_counter[task]
  avg_task_length[task] = total_num_turns/total_num_sessions

sns.set_style("whitegrid")
categories = list(avg_task_length.keys())
values = list(avg_task_length.values())

fig, ax = plt.subplots(figsize=(14, 6))
colors = sns.color_palette("viridis", len(categories))
bars = ax.bar([c.capitalize() for c in categories], values, 
              color=colors, width=0.7)
ax.set_xlabel('Task Categories', fontsize=16, labelpad=10)
ax.set_ylabel('Avg. session length (# turns)', fontsize=16, labelpad=10)
ax.set_title('Data Distribution across Tasks', fontsize=17,
             fontweight='bold', pad=20)

for idx, bar in enumerate(bars):
  yval = bar.get_height()
  ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1,
          f'{round(yval, 1)} turns', ha='center', va='bottom', fontsize=13)
  ax.text(bar.get_x() + bar.get_width()/2, yval - 1.7,
          f'{taskwise_session_counter[categories[idx]]} records',
          ha='center', va='bottom', fontsize=12, color='white')
  ax.text(bar.get_x() + bar.get_width()/2, yval - 2.7,
          f'{int(taskwise_video_length[categories[idx]])} mins',
          ha='center', va='bottom', fontsize=12, color='white')
  ax.text(bar.get_x() + bar.get_width()/2, yval - 3.7,
          f'{int(taskwise_turn_counter[categories[idx]])} total ',
          ha='center', va='bottom', fontsize=12, color='white')
  ax.text(bar.get_x() + bar.get_width()/2, yval - 4.5, f'      turns',
          ha='center', va='bottom', fontsize=12, color='white')

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
sns.despine(ax=ax, top=True, right=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Number of Expert words/turn histogram
user_words, expert_words = [], []
for file in file_to_dial_lens:
  user_words.extend(file_to_dial_lens[file]['user'])
  expert_words.extend(file_to_dial_lens[file]['expert'])
print('Avg number of user words per turn: ', sum(user_words) / len(user_words))
print('Avg numbr of expt wrds per turn:', sum(expert_words) / len(expert_words))

# Parse vs regular vs error comparison
parse_sessions = {'user': [], 'expert': []}
regular_sessions = {'user': [], 'expert': []}
error_sessions = {'user': [], 'expert': []}
num_files = {'parse': 0, 'regular': 0, 'error': 0}
num_turns = {'parse': 0, 'regular': 0, 'error': 0}
for file in file_to_dial_lens:
  if '_parse' in file:
    parse_sessions['user'].extend(file_to_dial_lens[file]['user'])
    parse_sessions['expert'].extend(file_to_dial_lens[file]['expert'])
    num_files['parse'] += 1
    num_turns['parse'] += len(file_to_dial_lens[file]['user'])
  elif '_mod' in file:
    error_sessions['user'].extend(file_to_dial_lens[file]['user'])
    error_sessions['expert'].extend(file_to_dial_lens[file]['expert'])
    num_files['error'] += 1
    num_turns['error'] += len(file_to_dial_lens[file]['user'])
  else:
    regular_sessions['user'].extend(file_to_dial_lens[file]['user'])
    regular_sessions['expert'].extend(file_to_dial_lens[file]['expert'])
    num_files['regular'] += 1
    num_turns['regular'] += len(file_to_dial_lens[file]['user'])


print(
    'Parse Sessions\nAverage number of user words:'
    f' {sum(parse_sessions["user"]) / len(parse_sessions["user"])} over'
    f' {num_files["parse"]} recordings and {num_turns["parse"]} turns')
print(
    'Average number of expert words: ',
    sum(parse_sessions['expert']) / len(parse_sessions['expert']),)

print(
    'Regular Sessions\nAverage number of user words:'
    f' {sum(regular_sessions["user"]) / len(regular_sessions["user"])} over'
    f' {num_files["regular"]} recordings and {num_turns["regular"]} turns')
print(
    'Average number of expert words: ',
    sum(regular_sessions['expert']) / len(regular_sessions['expert']),)

print(
    'Error Sessions\nAverage number of user words:'
    f' {sum(error_sessions["user"]) / len(error_sessions["user"])} over'
    f' {num_files["error"]} recordings and {num_turns["error"]} turns')
print(
    'Average number of expert words: ',
    sum(error_sessions['expert']) / len(error_sessions['expert']),)

sns.set_style('whitegrid')
categories = [
    'Concise, Following Steps',
    'Regular, Following Steps',
    'Regular, Making errors',]
values = [3.37, 10.93, 9.97]

fig, ax = plt.subplots(figsize=(8.9, 6))
colors = sns.color_palette('viridis', len(categories))
bars = ax.bar(categories, values, color=colors, width=0.7)

ax.set_xlabel(
    'User categories (Speech style, Action type)', fontsize=16, labelpad=10)
ax.set_ylabel('Average word count for user turns', fontsize=16, labelpad=10)
ax.set_title(
    'Distribution of user turn lengths', fontsize=17, fontweight='bold', pad=20)

sessions = [180, 252, 75]
turns = [2419, 3326, 891]
for idx, bar in enumerate(bars):
  yval = bar.get_height()
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 0.1,
      f'{round(yval, 1)} words',
      ha='center',
      va='bottom',
      fontsize=15,
  )
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      yval - 1.7,
      f'{sessions[idx]} sessions',
      ha='center',
      va='bottom',
      fontsize=15,
      color='white',
  )
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      yval - 2.5,
      f'{turns[idx]} turns',
      ha='center',
      va='bottom',
      fontsize=15,
      color='white',
  )

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
sns.despine(ax=ax, top=True, right=True)

ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


def plot_turn_length_histogram(data, xlabel, title):
  """Plots a histogram of turn lengths.

  Args:
    data: A list of turn lengths.
    xlabel: The label for the x-axis.
    title: The title of the plot.
  """
  sns.set_style("whitegrid")
  plt.figure(figsize=(10, 6))
  sns.histplot(
      data,
      bins=30,
      kde=True,
      color='skyblue',
      edgecolor='black',
      alpha=0.7,
      line_kws={'linewidth': 2, 'color': 'darkblue'}
  )

  plt.xlabel(xlabel, fontsize=14)
  plt.ylabel('Count', fontsize=14)
  plt.title(title, fontsize=16, fontweight='bold')
  plt.tick_params(axis='both', which='major', labelsize=12)

  mean_value = np.mean(data)
  plt.axvline(mean_value, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_value:.2f}')
  plt.legend(fontsize=12)
  sns.despine(top=True, right=True)
  plt.show()

# User turn length histogram
plot_turn_length_histogram(
    data=user_words,
    xlabel='Number of words in User turns',
    title='Histogram of User turn lengths',
)

# Expert turn length histogram
plot_turn_length_histogram(
    data=expert_words,
    xlabel='Number of words in Expert turns',
    title='Histogram of Expert turn lengths',
)

# Plot video length histogram
plot_turn_length_histogram(
    data=all_durations,
    xlabel='Video Duration (in seconds)',
    title='Histogram of video lengths',
)
