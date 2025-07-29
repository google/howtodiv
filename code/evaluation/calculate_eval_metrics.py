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

import jax
import nltk
import os
import json
import jax.numpy as jnp
from gemma import gm
from rouge import rouge_scorer

# Load Gemma3
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)


def evaluate_with_llm_as_judge(q, gt, mr ):
  eval_prompt = (
    'You will be given a user Query along with the expected Ground-truth Answer and Sample Response. '
    'Evaluate the quality of Sample Response by outputing a Score between 1 to 5 and a Reasoning explaining the rationale briefly in a sentence.\n'
    'Judge on the basis of accuracy, alignment with Ground-truth Answer, conciseness, accuracy and relevance of the Sample Response. '
    'Penalize long answers, those that include details not part of the Ground-truth Answer by giving a lower score.\nOutput in the format:\n'
    'Reasoning: <explanation in a sentence> \nScore: <number between 1 to 5>\n\n'
)

  prompt = f"""<start_of_turn>user
{eval_prompt}Query: {q}\nGround-truth Answer: {gt}\nSample Response: {mr}
    <start_of_turn>model
    """
  sampler = gm.text.Sampler( model=model, params=params,)
  out = sampler.sample(prompt).split('<end_of_turn>')[0]
  return out

#Load test ids
test_split_fpath = "HowToDIV/test_ids.txt"
with open(test_split_fpath, 'r') as file:
  test_ids = file.readlines()

#Compute all scores
score_list = {}
rouge_scorer_module = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
for testf_idx, test_file in enumerate(test_ids):
  test_file = test_file.strip()
  print(testf_idx, test_file)
  base_dir = "HowToDIV/"
  eval_fpath = f"HowToDIVEvals/{test_file.replace(base_dir, '')}"
  with gfile.Open(eval_fpath, "r") as json_file:
    session_data = json.load(json_file)
  score_list[test_file] = {'bleu_score':[], 'llmj_score':[], 'llmj_reason':[], 'rouge1': [], 'rouge2': []}
  for turn_idx in session_data:
    turn = session_data[turn_idx]
    print(turn_idx)
    if int(turn_idx)>=10:
      break
    query, gt, response1, response2 = turn['query'], turn['gt'], turn['model_response_h'], turn['model_response_tsh']
    used_response = response1 #change this based on the response needed

    llm_judge_output = evaluate_with_llm_as_judge(q=query, gt=gt, mr=used_response)
    llmj_score = int(llm_judge_output.split('Score: ')[-1])
    llmj_reason = llm_judge_output.split('Score: ')[0].split('Reasoning: ')[-1].strip()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([gt], used_response)
    rouge_score = rouge_scorer_module.score(gt, used_response)
    rouge1 = rouge_score['rouge1'].fmeasure
    rouge2 = rouge_score['rouge2'].fmeasure

    score_list[test_file]['bleu_score'].append(BLEUscore)
    score_list[test_file]['llmj_score'].append(llmj_score)
    score_list[test_file]['llmj_reason'].append(llmj_reason)
    score_list[test_file]['rouge1'].append(rouge1)
    score_list[test_file]['rouge2'].append(rouge2)

out_fpath = f"HowToDIVEvals/scores_prompt2.json"
with open(out_fpath, "w") as json_file:
  json.dump(score_list, json_file, indent=4)