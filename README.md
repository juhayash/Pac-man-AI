# Pac-man-AI
### Team: brostas
Members: Jun Hayashida, Ryan Wang
## Project Description
This project aims to develop a multi-player capture-the-flag variant of Pacman while agents contril both Pacman and ghosts in coordinated team-based strategies. My goal was to create agents capable of efficiently eating food on the opponent's side while defending our side from the opponent. The project was originally completed for the CSE140-01 course at UCSC, under the guidance of Professor Gilpin in the Spring of 2023.
## Features
### Offensive Agent
Focuses on maximizing food intake and capsules, avoiding ghosts, and increasing the overall game score.
### Defensive Agent
Prioritizes protecting our side from invaders, strategically positions itself based on the game state, and penalizing game state with invaders.
Both agents were built as reflex agents, utilizing a cross product of features and self-assigned weights to evaluate fame state and select optimal actions.
## Technical Approach
I implemented reflex agents, evaluating game stats based on a set of features and associated weights. The decision-making for the agents is based on a cross product of these features and their weights, with adjustments made through trial and error.
