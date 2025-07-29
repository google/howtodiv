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
import jax
import jax.numpy as jnp
from gemma import gm
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

# Load Gemma3
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

#Generate Responses i.e. Expert turns using model
test_split_fpath = "HowToDIV/test_ids.txt"
with open(test_split_fpath, 'r') as file:
  test_ids = file.readlines()

def get_prompt_versions(task, instructions):
  return [
      {'prompt': f'Assist the user on how to perform the specified task. Narrate only one step at a time in a conversative manner\n',
      'prompt_version': 'h'
      },
      {'prompt': (
          f'You are given the STEPS for {task}. Using these steps assist the user by describing instructions. '
          f'Narrate only one step at a time. \nSTEPS: {'\n'.join(instructions)}\n\n'
        ),
      'prompt_version': 'tsh'
      },
    ]

def get_ue_from_dialogues(cur_dialogue):
  ue_text, next_window = cur_dialogue.split('\n\n')
  expert_text = ue_text.split('\nExpert: ')[-1]
  user_text = ue_text.split('\nExpert: ')[0].split('\nUser: ')[-1]
  return next_window, user_text, expert_text

def save_results(test_file, logs):
  base_dir = "HowToDIV/"
  out_fpath = f"HowToDIVEvals/{test_file.replace(base_dir, '')}"
  with open(out_fpath, "w") as json_file:
    json.dump(logs, json_file, indent=4)

def draft_initial_prompt(howto_prompt, query):
  return f"""<start_of_turn>user
  {howto_prompt + query}<end_of_turn>
  <start_of_turn>model
  """

def draft_intermediate_prompt(prompt, prev_response, query):
  return (f"{prompt}{prev_response}<end_of_turn>\n"
  f"<start_of_turn>user\n{query}<end_of_turn>\n"
  "<start_of_turn>model\n")

subdir_to_task = {'repot': 'How to repot a plant', 'jump_car': 'How to jump start a car', 'coffee': 'How to make pour over coffee',
                  'changing_tire': 'How to change a car tire',
                  'oatmeal': 'How to make oatmeal', 'oatmeal_mod': 'How to make oatmeal', 'oatmeal_parse': 'How to make oatmeal',
                  'pinwheels': 'How to make pinwheels', 'pinwheels_mod': 'How to make pinwheels', 'pinwheels_parse': 'How to make pinwheels',
                  'quesadilla': 'How to make quesadilla', 'quesadilla_mod': 'How to make quesadilla', 'quesadilla_parse': 'How to make quesadilla',
                  'tea': 'How to make tea', 'tea_mod': 'How to make tea', 'tea_parse': 'How to make tea',
                  'coffee': 'How to make pour over coffee', 'coffee_mod': 'How to make pour over coffee', 'coffee_parse': 'How to make pour over coffee'}

for testf_idx, test_file in enumerate(test_ids):
  test_file = test_file.strip()
  test_fpath = test_file
  print(testf_idx, test_fpath)

  with open(test_fpath, 'r') as file:
      test_file_content = file.read()

  instructions, dialogues = test_file_content.split('\n\nDialogue\n')
  instructions = instructions.split('Instructions\n')[-1].split('\n')
  dialogues = dialogues.split('\nTurn ')

  dialogue_data = [[dialogues[0]]]
  for i in range(1, len(dialogues)):
    next_window, user_text, expert_text = get_ue_from_dialogues(dialogues[i])
    dialogue_data[-1].extend([user_text, expert_text])
    dialogue_data.append([next_window])

  task = subdir_to_task[test_fpath.split('/')[-2]]
  prompt_versions = get_prompt_versions(task, instructions)

  logs = {}
  sampler = gm.text.Sampler(model=model, params=params,)
  for prompt_idx in range(len(prompt_versions)):
    print(f'Processing prompt {prompt_idx+1}')
    howto_prompt, prompt_version  = prompt_versions[prompt_idx]['prompt'], prompt_versions[prompt_idx]['prompt_version']
    prompt = draft_initial_prompt(howto_prompt, dialogue_data[0][1])

    model_response = sampler.sample(prompt).split('<end_of_turn>')[0]
    if 0 not in logs:
      logs[0] = {'query': dialogue_data[0][1], 'gt': dialogue_data[0][2], f'model_response_{prompt_version}': model_response}
    else:
      logs[0][f'model_response_{prompt_version}'] = model_response

    for i in range(len(dialogue_data)-2):
      prompt = draft_intermediate_prompt(prompt, dialogue_data[i][2], dialogue_data[i+1][1])
      model_response = sampler.sample(prompt).split('<end_of_turn>')[0]
	  print(f'Computed response for {i+1}th turn')
      if (i+1) not in logs:
        logs[i+1] = {'query': dialogue_data[i+1][1], 'gt': dialogue_data[i+1][2], f'model_response_{prompt_version}': model_response}
      else:
        logs[i+1][f'model_response_{prompt_version}'] = model_response

  save_results(test_file, logs)
