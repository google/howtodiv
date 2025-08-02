# Generating Dialogues from Egocentric Instructional Videos: HowToDIV Dataset and Benchmark

This repo contains the code and data for building HowToDIV dataset and benchmark evaluations. The repo implements the core components, for generating dialogues from egocentric narrated instructional videos along with evaluating the performance of Gemma3 on HowToDIV to establish a benchmark. 

## Overview
Many everyday tasks ranging from fixing appliances, to cooking new recipes to car maintenance require expert knowledge, especially when tasks are complex and multi-step. Despite growing interest in AI agents, there is a dearth of dialogue-video datasets grounded for real world task assistance. In this paper, we propose a simple yet effective approach that transforms single-person instructional videos into structured two-person dialogues, aligned with fine grained video-shots and task steps. Our approach, powered by large language models, offers a highly efficient alternative to the substantial cost and effort required for manual data collection. Using this technique, we build HowToDIV, a large-scale dataset containing 507 sessions and 6636 question-answer pairs across diverse tasks in cooking, mechanics, and planting. Each session includes multi-turn conversation between an expert and novice, with synchronized novice video clips, sourced from publicly available NIV and EgoPER datasets. We benchmark the performance of Gemma3-4B model on HowToDIV, establishing a baseline for future research on this new multimodal instruction-following dialogue task.


HowToDIV dataset consists of 507 sessions covering 6636 novice - expert dialogue turns ranging over 9 tasks: Mechanics - Jump start a car, Change car tire, Cooking - Prepare coffee using a moka pot, Cook a tortilla, Prepare pinwheels, Make tea, Prepare filter coffee, Make Quesadilla, and Planting - Repot a plant. For each session, the dataset provides multi-turn user-expert dialogues, task instructions and per-turn video annotations. Our video annotations consist of start and stop times; these timestamps correspond to videos in the given original datasets.

Please download original EgoPER Dataset from ([https://github.com/robert80203/EgoPER_official](https://github.com/robert80203/EgoPER_official)) and NIV Dataset from ([https://www.di.ens.fr/willow/research/instructionvideos/](https://www.di.ens.fr/willow/research/instructionvideos/))

More details can be found on the project page ([https://github.com/google/howtodiv](https://github.com/google/howtodiv)).


## Contributors
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Andrea Colaco (Google)**

## Contribute
To learn how to contribute to this project, read [CONTRIBUTING.md](docs/contributing.md).

## License
The code is released with Apache 2.0 License [LICENSE.txt](LICENSE).
The data is released with CC-BY-4.0 License [CC-BY-4.0](data/CC-BY-4.0).

 ```
@article{laggarwal2025howtodiv,
  author    = {Aggarwal, Lavisha and Colaco, Andrea},
  title     = {Generating Dialogues from Egocentric Videos: Dialogue-Video Dataset (HowToDIV) for HowTo Procedural tasks},
  journal   = {arxiv},
  year      = {2025},
}
```






