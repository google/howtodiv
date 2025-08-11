# Generating Dialogues from Egocentric Instructional Videos for Task Assistance: Dataset, Method and Benchmark

This repo contains the code and data for building HowToDIV dataset and benchmark evaluations.

## Overview
Many everyday tasks ranging from fixing appliances, cooking recipes to car maintenance require expert knowledge,  especially when tasks are complex and multi-step. Despite growing interest in AI agents, there is a dearth of dialogue-video datasets grounded for real world task assistance. In this paper, we propose a simple yet effective approach that transforms single-person instructional videos into task-guidance two-person dialogues, aligned with fine grained steps and video-clips. Our fully automatic approach, powered by large language models, offers an efficient alternative to the substantial cost and effort required for manual data collection. Using this technique, we build HowToDIV, a large-scale dataset containing 507 conversations, 6636 question-answer pairs and 24 hours of videoclips across diverse tasks in cooking, mechanics, and planting. Each session includes multi-turn conversation where an expert teaches a novice user how to perform a task step by step, while observing user's surrounding through a camera and microphone equipped wearable device. We establish the baseline benchmark performance on HowToDIV dataset through Gemma-3 model for future research on this new task of dialogues for procedural-task assistance.


## Getting Started
Please download original EgoPER Dataset from ([https://github.com/robert80203/EgoPER_official](https://github.com/robert80203/EgoPER_official)) and NIV Dataset from ([https://www.di.ens.fr/willow/research/instructionvideos/](https://www.di.ens.fr/willow/research/instructionvideos/))

HowToDIV dataset consists of 507 conversations covering 6636 novice - expert dialogue turns ranging over 9 tasks: Mechanics - Jump start a car, Change car tire, Cooking - Prepare coffee using a moka pot, Cook a tortilla, Prepare pinwheels, Make tea, Prepare filter coffee, Make Quesadilla, and Planting - Repot a plant. For each session, the dataset provides multi-turn user-expert dialogues, task instructions and per-turn video annotations. Our video annotations consist of start and stop times; these timestamps correspond to videos in the given original datasets.

More details can be found on the project page ([https://github.com/google/howtodiv](https://github.com/google/howtodiv)).


## Contributors
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Andrea Colaco (Google)**

## Contribute
To learn how to contribute to this project, read [CONTRIBUTING.md](docs/contributing.md).

## License
The code is released with Apache 2.0 License [LICENSE.txt](LICENSE) and the data is released with CC-BY-4.0 License [CC-BY-4.0](data/CC-BY-4.0).

 ```
@article{laggarwal2025howtodiv,
  author    = {Aggarwal, Lavisha and Bahirwani, Vikas and Li, Lin and Colaco, Andrea},
  title     = {Generating Dialogues from Egocentric Instructional Videos for Task Assistance: Dataset, Method and Benchmark},
  journal   = {arxiv},
  year      = {2025},
}
```

