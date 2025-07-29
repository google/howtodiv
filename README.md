# Generating Dialogues from Egocentric Videos: Dialogue-Video (HowToDIV) Dataset for HowTo Procedural tasks

This repo contains the code and data for the HowToDIV dataset. The repo implements the core components, for generating dialogues from egocentric narrated instructional videos along with evaluating the performance of Gemma3 on HowToDIV to establish a benchmark. 

## Overview
On a daily basis we often need to perform tasks related to home improvement, fixing and using appliances, cooking, or mechanical needs that are outside of our knowledge and expertise. Some of these are quite complex involving multiple steps, requiring expert knowledge. Hence there has been a need to develop multimodal AI agents which can help users perform new tasks more easily.

This dataset consists of dialogues between a user and an expert where the expert instructs the user on how to perform and finish a task, while the user inquires for next steps or any questions. For each session, the dataset provides multi-turn user-expert dialogues, task instructions and per-turn video annotations. This dataset is built on top of the publically available NIV and EgoPER datasets. Our video annotations consist of start and stop times; these timestamps correspond to videos in the given original datasets.

HowToDIV dataset consists of 507 sessions covering 6627 user - expert dialogue turns ranging over 9 tasks: Mechanics - Jump start a car, Change car tire, Cooking - Prepare coffee using a moka pot, Cook a tortilla, Prepare pinwheels, Make tea, Prepare filter coffee, Make Quesadilla, and Planting - Repot a plant.

More details can be found on the project page ([https://github.com/google/howtodiv](https://github.com/google/howtodiv)).


## Contributors
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Andrea Colaco (Google)**

## Contribute
To learn how to contribute to this project, read [CONTRIBUTING.md](../CONTRIBUTING.md).

## License
The code is released with Apache 2.0 License [LICENSE.txt](../LICENSE.txt).
The data is released with CC-BY-4.0 License [CC-BY-4.0](../data/CC-BY-4.0).

 ```
@article{laggarwal2025howtodiv,
  author    = {Aggarwal, Lavisha and Colaco, Andrea},
  title     = {Generating Dialogues from Egocentric Videos: Dialogue-Video Dataset (HowToDIV) for HowTo Procedural tasks},
  journal   = {arxiv},
  year      = {2025},
}
```






