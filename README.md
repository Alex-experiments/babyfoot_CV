# :soccer: Babyfoot Computer Vision :soccer:


## :book: Description

This project aims at computing statistics of a foosball game out of a video taken from above. The field, the ball and the players are detected using YOLO and ohter computer vision techniques. Then a 2D flatenned view of the field is produced to overcome the perspective effects of the image. Finally, some statistics are computed, such as the score, the maximum speed of the ball, the possession, etc. Also, some punctual events are registered: the shots and the goals. Further details can be found in the report [here](Projet_Babyfoot.pdf)

This project is part of the final year project of CentraleSupélec.


## :busts_in_silhouette: Team Members

- Vivien Conti
- Guillaume Dugat
- Alexandre Gautier


## :japanese_castle: Structure of the repository

...


## :hammer: Installation

To manage the Python dependencies and the use of a virtual environment, we have chosen to use poetry. Further information about this tool can be found on the following [website](https://python-poetry.org/docs/basic-usage/).

To install the project, run the following command in a terminal at the root of the project:

```
make install
```

## :ferris_wheel: Usage

#### Pipeline

To run the whole pipeline of the project (the video acquisition, the detection of the field, the ball and the players, the 3D reconstruction and the computation of the statistics), run the following command :

```
make run
```

#### Tests

We implemented some tests, that can be run using pytest :

```
make pytest
```
