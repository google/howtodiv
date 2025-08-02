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
from typing import List
from gemma import gm
from kauldron import kd
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.00'

# Load Gemma3
print(jax.device_count())
model = gm.nn.Gemma3_27B()
params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    sharding=kd.sharding.FSDPSharding(),)


# Utils for generating Instructions and Video-Annotations
def save_instructions(cfg, fname, instructions):
  fpath = f'HowToDIV/NIV_DIV/{cfg}/{fname.split('.')[0]}.txt'
  with open(fpath, 'wt') as file:
    file.write(instructions)
  return

steps_command_template = (
    "You will be given a transcript with timestamps. Go through the transcript"
    " and convert it into a list of instructions for 'How to {task}'. Also"
    " predict the start and stop times for each step, based on the transcript"
    " timestamps. Please ensure that the instruction steps are high quality and"
    " accurate, and involve one action. Every instruction should be a single"
    " sentence which is informative and concise. Output should be in format:\n"
    "How to {task}\nStep X: <<instructional step>>\nStart: <<HH:mm:ss>>\nStop:"
    " <<HH:mm:ss>>"
)


def generate_instructions(cfg: str):
  """Generates instructions for a given config.

  Args:
    cfg: The config to use.
  """
  files = os.listdir(ego_vid_directory_template.format(config=cfg))
  command = steps_command_template.format(task=keyword_mapping[cfg])
  for i in range(len(files)):
    fpath = f'{subtitle_directory_template.format(config=cfg)}{files[i].split(".")[0]}.srt'
    with open(fpath, 'r') as f:
      task_subt = f.read()
    generic_sampler = gm.text.Sampler(model=model, params=params,)
    prompt_instruction = command + task_subt
    prompt_final = f"""<start_of_turn>user
    {prompt_instruction}<end_of_turn>
    <start_of_turn>model
    """
    turn_response = generic_sampler.sample(
        prompt_final, sampling=gm.text.RandomSampling(temperature=1.5)
    )
    turn_content = turn_response.split('<end_of_turn>')[0]
    save_instructions(cfg, files[i], turn_content)


# Utils for Generating Dialogues
def read_instruction_file(cfg, file):
  """Reads an instruction file.

  Args:
    cfg: The config to use.
    file: The file to read.

  Returns:
    The contents of the file.
  """
  fpath = f'{instruction_dir_template.format(config=cfg)}/{file}'
  with open(fpath, 'r') as f:
    instructions_i = f.read()
  return instructions_i


def parse_instruction_into_steps(instructions_i):
  """Parses instructions into steps.

  Args:
    instructions_i: The instructions to parse.

  Returns:
    A tuple of the steps and their times.
  """
  steps_list = instructions_i.split('Step ')[1:]
  steps, times = [], []
  for step_idx, raw_step in enumerate(steps_list):
    step_parts = raw_step.split('Start:')
    step_desc = step_parts[0].split(': ')[-1].strip()
    start, stop = step_parts[1].split('Stop:')
    start = start.strip()
    stop = stop.strip()
    step = str(step_idx+1) + '. '+ step_desc
    times.append([start, stop])
    steps.append(step)
  steps.append(str(len(steps)+1) + '. Done')
  first_step_start = times[0][1]
  times.insert(0, ['00:00:00', first_step_start])
  return steps, times


def convert_steps_to_dialogue(cfg, steps_string):
  """Converts steps to a dialogue.

  Args:
    cfg: The config to use.
    steps_string: The steps to convert.

  Returns:
    The dialogue.
  """
  dialogue_command_template = (
      'Given the following set of steps, write a dialogue between an Expert and'
      " User where the Expert teaches User 'How to {task}' observing him, while"
      ' User performs the actions and asks for questions or next steps. Only'
      ' output the dialogues, not the actions. Start by the User asking how to'
      ' {task}. Make sure the number of turns is same as number of steps. Only'
      ' the Expert can see instructions while the User does not know the next'
      ' step. Output should be in format Turn X\n User: <<user dialogue>> \n'
      ' Expert: <<expert dialogue>>\n'
  )
  command = dialogue_command_template.format(task=keyword_mapping[cfg])

  generic_sampler = gm.text.Sampler(
      model=model,
      params=params,
  )
  prompt_intermediate = command + steps_string
  prompt_final = f"""<start_of_turn>user
  {prompt_intermediate}<end_of_turn>
  <start_of_turn>model
  """
  turn_response = generic_sampler.sample(
      prompt_final,
      sampling=gm.text.RandomSampling(temperature=1.5),
  )
  turn_content = turn_response.split('<end_of_turn>')[0]
  return turn_content


def get_turns(generated_transcript):
  """Gets the turns from a generated transcript.

  Args:
    generated_transcript: The transcript to get the turns from.

  Returns:
    A list of turns.
  """
  generated_transcript_comps = generated_transcript.split('User:')
  turns = []
  for ct_idx, cur_turn in enumerate(generated_transcript_comps):
    if 'Expert:' not in cur_turn:
      continue
    user_expert_parts = cur_turn.split('Expert:')
    user_part = user_expert_parts[0].strip()
    expert_part = user_expert_parts[1].split('Turn')[0].strip()
    turns.append(f'Turn {ct_idx}\nUser: {user_part}\nExpert: {expert_part}')
  return turns


def save_dialogue(turn_content, step_times, ordered_steps, fname):
  """Saves a dialogue to a file.

  Args:
    turn_content: The dialogue to save.
    step_times: The times for each step.
    ordered_steps: The ordered steps.
    fname: The name of the file to save to.

  Returns:
    True if the dialogue was saved successfully, False otherwise.
  """
  fpath = f'HowToDIV/NIV_DIV/{config}/{fname}_dialogue.txt'
  turns = get_turns(turn_content)
  if len(turns) != len(step_times):
    print(f'ERROR = {len(turns)}!={len(step_times)}. FILE: ', fname)
    return False

  formatted_list = [f'Instructions\n{ordered_steps}\n\nDialogue\n']
  for i in range(len(turns)):
    t, s = step_times[i], turns[i]
    formatted_list.append(f'[{t[0]}, {t[1]}]\n{s}\n\n')
  formatted_list = ''.join(formatted_list)
  with open(fpath, 'wt') as file:
    file.write(formatted_list)
  return True


# Define configs and mappings
all_configs = ['repot', 'jump_car', 'changing_tire', 'coffee']
keyword_mapping = {
    'repot': 'Repot A Plant',
    'coffee': 'Make coffee',
    'jump_car': 'Jump Start a Car',
    'changing_tire': 'Change a Tire',
}


# Step 1: Generate Instructions and Video-Annotations
ego_vid_directory_template = 'NIV Dataset/{config}/videos'
subtitle_directory_template = (
    'NIV Dataset/data_release/{config}/subtitles/manual/')
for config in all_configs:
  generate_instructions(config)

# Step 2: Generate Dialogues
instruction_dir_template = 'HowToDIV/NIV_DIV/{config}'
for config in all_configs:
  files = os.listdir(instruction_dir_template.format(config=config))
  total_num_turns, num_parsed_videos = 0, 0
  for i in range(1, len(files)):
    if '_dialogue' in files[i]:
      continue
    current_instruction = read_instruction_file(config, files[i])
    steps, times = parse_instruction_into_steps(current_instruction)
    steps_str = '\n'.join(steps)
    dialogue = convert_steps_to_dialogue(config, steps_str)
    result = save_dialogue(dialogue, times, steps_str, files[i].split('.')[0])
    if result:
      total_num_turns += len(steps)
      num_parsed_videos += 1
  print(
      f'Total num turns for {config} = {total_num_turns} across'
      f' {num_parsed_videos} conversations')
