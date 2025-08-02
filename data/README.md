### README
## HowToDIV Dataset
This archive contains dialogues, instructions and associated video-clips for HowToDIV Dataset. It consists of 507 sessions covering 6636 novice - expert dialogue turns and contains data for 9 tasks: preparing tea, coffee, quesadilla, pinwheels, oatmeal, Changing a car tire (changing_tire), making coffee (coffee), jump car battery (jump_car), and repotting plant (repot).


## Organization of folder

1. test_ids.txt: Data recording session ID's corresponding to test split
2. train_ids.txt: Data recording session ID's corresponding to train split

3. NIV DIV: Data for the tasks corresponding to NIV Dataset - changing tire, making coffee, jump car battery , and repotting plant. Within each subdirectory, there are 2 files for a recording:
  a. *.txt contains only the instruction steps
  b. *_dialogue.txt contains the dialogues, instructions and videosteps for the given session. Videosteps are represented as intervals corresponding to videos present in NIV dataset which can be downloaded from https://www.di.ens.fr/willow/research/instructionvideos/

4. EgoPER DIV: Data for the tasks corresponding to EgoPER Dataset - preparation pinwheels, oatmeal, quesadilla, coffee, tea. For each task there are 3 subdirs (named for the speech-style and user-type):
  a. *_concise_followsteps: Contains data corresponding to concise and short user ialogues and when the user follows all instruction steps accurately without making mistakes
  b. *_regular_followsteps: Contains data corresponding to normal user ialogues and when the user follows all instruction steps accurately without making mistakes
  c. *_usererror: Contains data corresponding to normal user ialogues and when the user makes mistakes
  Each *.txt contains the dialogues, instructions and videosteps for the given session. Videosteps are represented as intervals corresponding to videos present in EgoPER dataset which can be downloaded from https://github.com/robert80203/EgoPER_official

More details can be found on the project page ([https://github.com/google/howtodiv](https://github.com/google/howtodiv)).


## Contribute
To learn how to contribute to this project, read [CONTRIBUTING.md] in the github repo (../CONTRIBUTING.md).


## License
The data is released with CC-BY-4.0 License [CC-BY-4.0](../data/CC-BY-4.0).


## Citation
If you plan to use these data in publications please cite the following paper:
@article{laggarwal2025howtodiv,
  author    = {Aggarwal, Lavisha and Colaco, Andrea},
  title     = {Generating Dialogues from Egocentric Videos: Dialogue-Video Dataset (HowToDIV) for HowTo Procedural tasks},
  journal   = {arxiv},
  year      = {2025},
}


## Contact
For all questions please feel free to contact the first author of the paper: lavishaaggarwal1995@gmail.com


## Contributors
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Andrea Colaco (Google)**
