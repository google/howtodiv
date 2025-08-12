### README
## HowToDIV Dataset
This archive contains dialogues, instructions and associated video-clips for HowToDIV Dataset. It consists of 507 sessions covering 6636 novice-expert dialogue turns and contains data for 9 tasks: preparing tea, coffee, quesadilla, pinwheels, oatmeal, changing a car tire (changing_tire), making coffee (coffee), jump car battery (jump_car), and repotting plant (repot).


## Organization of folder

1. test_ids.txt: Data recording session ID's corresponding to testing split
2. train_ids.txt: Data recording session ID's corresponding to training split
3. val_ids.txt: Data recording session ID's corresponding to validation split

4. NIV DIV: Data for the tasks corresponding to narrated instruciton video dataset - changing tire, making coffee, jump car battery , and repotting plant. Within each subdirectory, there are 2 files for a recording:
    1. *.txt contains only the instruction steps
    2. *_dialogue.txt contains the dialogues, instructions and videosteps for the given session. Videosteps are represented as intervals corresponding to videos present in NIV dataset (can be downloaded from https://www.di.ens.fr/willow/research/instructionvideos/)

5. EgoPER DIV: Data for the tasks corresponding to EgoPER Dataset - preparation pinwheels, oatmeal, quesadilla, coffee, tea. For each task there are 3 subdirs (named for the speech-style and user-type):
    1. *_concise_followsteps: Contains data corresponding to concise and short user dialogues and the user follows all instruction steps accurately without making mistakes
    2. *_regular_followsteps: Contains data corresponding to normal user dialogues and when the user follows all instruction steps accurately without making mistakes
    3. *_usererror: Contains data corresponding to normal user dialogues and when the user either makes modifications to steps OR makes mistakes and corrects them in the following turns
    4. Each *.txt contains the dialogues, instructions and videosteps for the given session. Videosteps are represented as intervals corresponding to videos present in EgoPER dataset (can be downloaded from https://github.com/robert80203/EgoPER_official)

More details can be found on the project page ([https://github.com/google/howtodiv](https://github.com/google/howtodiv)).


## Contribute
To learn how to contribute to this project, read [CONTRIBUTING.md](../docs/contributing.md).


## License
The data is released with CC-BY-4.0 License [CC-BY-4.0](CC-BY-4.0).


## Citation
If you plan to use these data in publications please cite the following paper:
```
@article{laggarwal2025howtodiv,
  author    = {Aggarwal, Lavisha and Colaco, Andrea},
  title     = {Generating Dialogues from Egocentric Videos: Dialogue-Video Dataset (HowToDIV) for HowTo Procedural tasks},
  journal   = {arxiv},
  year      = {2025},
}
```


## Contact
For all questions please feel free to contact the first author of the paper: lavishaaggarwal1995@gmail.com


## Contributors
- **Lavisha Aggarwal (Google)**
- **Vikas Bahirwani (Google)**
- **Lin Li (Google)**
- **Andrea Colaco (Google)**
