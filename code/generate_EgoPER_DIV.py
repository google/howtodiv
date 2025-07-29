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
import jax
import jax.numpy as jnp
from gemma import gm
from kauldron import kd
import json


def add_steps_order(step_list):
  ordered_steps = []
  for i in range(len(step_list)):
    ordered_steps.append(f"{i+1}. {step_list[i]}")
  ordered_steps.append(f"{len(ordered_steps)+1}. Done")
  return ordered_steps

def call_model(task_command, task_steps):
  generic_sampler = gm.text.Sampler(model=model, params=params,)
  prompt_intermediate = task_command + task_steps
  prompt_final = f"""<start_of_turn>user
  {prompt_intermediate}<end_of_turn>
  <start_of_turn>model
  """
  turn_response = generic_sampler.sample(prompt_final, sampling=gm.text.RandomSampling(temperature=1.5),).split('<end_of_turn>')[0]
  return turn_response

def get_turns(generated_transcript):
  generated_transcript_split = generated_transcript.split('User:')
  turns = []
  for ct_idx, g_i in enumerate(generated_transcript_split):
    if 'Expert:' not in g_i:
      continue
    user_expert_parts = g_i.split('Expert:')
    user_part = user_expert_parts[0].strip()
    expert_part = user_expert_parts[1].split('Turn')[0].strip()
    turns.append(f'Turn {ct_idx}\nUser: {user_part}\nExpert: {expert_part}')
  return turns

def save_transcript(turn_content, step_times, ordered_steps, fname, modifications=None):
  fpath = f"HowToDIV/EgoPER_DIV/{config}_modification/{fname}.txt"
  turns = get_turns(turn_content)
  if len(turns) != len(step_times):
    print(f'ERROR = {len(turns)}!={len(step_times)}. FILE: ', fname)
    return False

  formatted_list = [f'Instructions\n{ordered_steps}\n\nDialogue\n']
  for i in range(len(turns)):
    t, s = step_times[i], turns[i]
    if modifications is not None:
      m = modifications[i]
      formatted_list.append(f"[{t[0]}, {t[1]}]\n{m}\n{s}\n\n")
    else:
      formatted_list.append(f"[{t[0]}, {t[1]}]\n{s}\n\n")
  formatted_list = ''.join(formatted_list)
  with open(fpath, "wt") as file:
    file.write(formatted_list)
  return True


# Generate DIV for Normal Sessions
def generate_DIV(config, task_file_list, cur_config_command):
  for video in task_file_list:
    fname = video['video_id']
    if '_normal_' not in fname:
      continue
    time_stamps, error_description = video['labels']['time_stamp'], video['labels']['error_description']
    step_times, step_titles = [], []
    # add time for step 1
    if error_description[0] == 'BG':
      step_times.append([time_stamps[0][0], time_stamps[0][1]/2 ])
      step_times.append([time_stamps[0][1]/2, time_stamps[0][1] ])
    else:
      start, stop = 0, time_stamps[0][0]
      step_times.append([start, stop//2 ])
      step_times.append([stop//2, stop ])
  
    reached_floss_step = False # pinwheels
    for t, s in zip(time_stamps, error_description):
      if s == 'BG':
        continue
      
      if config == 'oatmeal':
        if s == 'Measure 4 Tablespoons of quick-cook oats':
          s = 'Measure 4 Tablespoons of quick-cook oats and put in bowl'
        if s in ['Put bowl in microwave', 'Remove bowl from microwave']:
          continue

      if config == 'coffee': # merge same steps    
        if step_titles and s == step_titles[-1]:
          step_times[-1][1] = t[1]
        else:
          step_times.append(t)
          step_titles.append(s)
        continue

      if config == 'pinwheels':
        if reached_floss_step:
          if s in ['Slice using floss', 'Put floss under tortilla']:
            if step_titles[-1] != 'Similarly slice entire tortilla':
              step_titles.append('Similarly slice entire tortilla')
              step_times.append(t)
            else:
              step_times[-1][1] = t[1]
          else:
            step_times.append(t)
            step_titles.append(s)
          continue
        if s == 'Slice using floss':
          reached_floss_step = True
      step_times.append(t)
      step_titles.append(s)

    step_titles.insert(0, ingreds[config])
    ordered_steps = add_steps_order(step_titles)
    ordered_steps = "\n".join(ordered_steps)
    generated_transcript = call_model(cur_config_command.format(config=config), ordered_steps)
    save_transcript(generated_transcript, step_times, ordered_steps, fname)


# Generate DIV for Error Sessions
def generate_error_DIV(config, task_file_list, cur_config_command):
  for video_idx, video in enumerate(task_file_list):
    fname = video['video_id']
    if ('_error_' not in fname) or 
     (1 in video['labels']['action_type'] and 2 not in video['labels']['action_type']) or 
      (1 not in video['labels']['action_type'] and 4 not in video['labels']['action_type']):
      continue

    time_stamps, error_description = video['labels']['time_stamp'], video['labels']['error_description']
    actions, action_types = video['labels']['action'], video['labels']['action_type']
    step_times, step_titles, modifications = [], [], []

    # add time for step 1
    if error_description[0] == 'BG':
      step_times.append([time_stamps[0][0], time_stamps[0][1]/2 ])
      step_times.append([time_stamps[0][1]/2, time_stamps[0][1] ])
    else:
      start = 0
      stop = time_stamps[0][0]
      step_times.append([start, stop//2 ])
      step_times.append([stop//2, stop ])
    modifications.append(['None', 'NA'])
    modifications.append(['None', 'NA'])

    reached_floss_step = False # pinwheels
    found_discrepancy = False
    step_titles.insert(0, ingreds[config])
    local_action_type_counts = {0:0, 1:0, 2:0, 4:0}
    for step_idx in range(len(time_stamps)):
      t, s, ac, acty = time_stamps[step_idx], error_description[step_idx],  actions[step_idx], action_types[step_idx]
      if s == 'BG' or acty == 3:
        continue
      if acty == 1 and action_types[step_idx+2]!=2 and action_types[step_idx+1]!=2:
        print('Error, found Error_Slip however next step not Error_correction', )
        found_discrepancy = True
        break
      if config == 'pinwheels' and s == 'Place tortilla on table':
        acty = 0

      if config == 'oatmeal':
        if s == 'Measure 4 Tablespoons of quick-cook oats':
          s = 'Measure 4 Tablespoons of quick-cook oats and put in bowl'
        if s in ['Put bowl in microwave', 'Remove bowl from microwave']:
          continue

      if step_titles and s == step_titles[-1]: # coffee # pinwheels # merge same steps
        step_times[-1][1] = t[1]
        continue

      if config == 'pinwheels':
        if reached_floss_step:
          if s in ['Slice using floss', 'Put floss under tortilla', 'Slice using knife']:
            if step_titles[-1] != 'Similarly slice entire tortilla':
              step_titles.append('Similarly slice entire tortilla')
              step_times.append(t)
              local_action_type_counts[acty] +=1
              s1, modifications = add_to_modifications(acty, s, modifications)
            else:
              step_times[-1][1] = t[1]
          else:
            step_times.append(t)
            step_titles.append(s)
            local_action_type_counts[acty] +=1
            s1, modifications = add_to_modifications(acty, s, modifications)
          continue
        reached_floss_step = True if s in ['Slice using floss', 'Slice using knife'] else reached_floss_step


      s1, modifications = add_to_modifications(acty, s, modifications)
      step_times.append(t)
      step_titles.append(s1)
      local_action_type_counts[acty] +=1

    if local_action_type_counts[1] + local_action_type_counts[2] + local_action_type_counts[4] == 0:
      print('No Modification found'); continue
    if found_discrepancy:
      continue
    ordered_steps = add_steps_order(step_titles)
    ordered_steps_string = "\n".join(ordered_steps)
    generated_transcript = call_model(cur_config_command, ordered_steps_string)
    result = save_transcript(generated_transcript, step_times, ordered_steps_string, fname, modifications)


# mappings
ingreds = {
    'tea': "Ingredients needed: Kettle, water, tea bag, honey, mug",
    'coffee': "Ingredients needed: coffee beans, water, paper filter, mug, kettle, dripper, coffee grinder",
    'oatmeal': "Ingredients needed: oats, raisins, banana, honey, cinnamon, water, a bowl, microwave",
    'pinwheels': "Ingredients needed: tortilla, nut butter, butter knife, cutting board, floss",
    'quesadilla': "Ingredients needed: tortilla, nutella, banana, cinammonm, cutting board, butter knife",
}

errors_to_correction_map = {"Drop tea bag": "**previously user dropped tea bag** Discard tea bag and place a new one",
      "Pour water into another mug": "**previously user poured water into a different mug** Transfer the tea bag to this mug",
      "Drop tortilla on floor": "**previously user mistakeny dropped tortilla on floor** Discard it and place a new tortilla",
      "Fold tortilla": "**previously user mistakeny folded the tortilla** Unfold tortilla and roll it",
      "Add water to a different bowl": "**previously user added water into a different bowl** Transfer the water to this bowl",
      "Add bananas to another empty bowl": "**previously user added bananas into a different bowl** Transfer the bananas into this bowl",
      "Drop tortilla": "**previously user mistakeny dropped tortilla on floor** Discard it and place a new tortilla",
      "Fold tortilla into quarter-circle": "**previously user mistakeny folded tortilla into quarter-circle** Unfold tortilla and roll it",
      "Knock over dripper": "**previously user mistakeny knocked over dripper** Put the dripper back to the mug",
      "Tear paper filter": "**previously user mistakenly tore the paper filter** Fold paper filter in half to create semi-circle",
      }

cur_config_command = ("Given the following set of steps, write a dialogue between an Expert and User where the Expert teaches User 'How to make {config_c}' observing him, "
      "while User performs the actions and asks for questions or next steps. Only output the dialogues, not the actions. Start by the User asking how to make {config}. "
      "Make sure the number of turns is same as number of steps. Only the Expert can see instructions while the User does not know the next step. {user_turn_spec}"
      "Output should be in format Turn X\n User: <<user dialogue>> \n Expert: <<expert dialogue>>")

def add_to_modifications(acty, s, modifications):
  if acty == 4:
    modifications.append(['modification', s])
    s1 = 'Measure 12 ounces of cold water' if s == 'Directly pour water to kettle' else config_action_map[ac]
  elif acty == 1:
    modifications.append(['error', s])
    s1 = config_action_map[ac]
  elif acty == 2:
    modifications.append(['correction', s])
    s1 = errors_to_correction_map[modifications[-2][1]]
  else:
    modifications.append(['follow_instruction', 'NA'])
    s1=s
  return s1, modifications


# Load Gemma3
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
print(jax.device_count())
model = gm.nn.Gemma3_27B()
params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    sharding=kd.sharding.FSDPSharding(),
)

# Load EgoPER Annotations
all_configs = ['tea', 'coffee', 'oatmeal', 'pinwheels', 'quesadilla']
filename = 'EgoPER/annotation.json'
with open(filename, 'rt') as f:
  data = json.load(f)


# Choose user dialogue preference
user_turn_spec = ""
# user_turn_spec = "Most of the times User dialogues are short. "


# Generate dialogues, instructions and videostep annotations for normal conversations
for config in all_configs:
  task_file_list = data[config]['segments']
  generate_DIV(config, task_file_list, cur_config_command.format(config=config, config_c = config.capitalize(),user_turn_spec=user_turn_spec))


# Generate dialogues, instructions and videostep annotations for error conversations
for config in all_configs:
  config_action_map = {data[config]['action2idx'][k]:k for k in data[config]['action2idx']}
  if config == 'quesadilla':
    config_action_map[6] = 'Fold tortilla in half'  
  task_file_list = data[config]['segments']
  generate_error_DIV(config, task_file_list, cur_config_command.format(config=config, config_c = config.capitalize(),user_turn_spec=user_turn_spec))