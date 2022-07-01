# OADM
Obstacle Avoidance by Dangerous Map
<img src="https://raw.githubusercontent.com/ge95net/OADM/main/pictures/Picture1.png" width="500" />
## Abstract
Obstacle avoidance in dense crowd is a very important but challenging task. Previous had presented a lot of approaches to address this issue. However, There are still some deficiencies in these work. (1) classical method like ORCA and learning-based methods like CADRL or SARL all requires global-state, it is hard to realize in real world. (2) These methods performs badly in real dense crowd with large number of humans because of high computation cost and low robustness. We propose a novel method for robot to better avoid dynamic obstacles without global-state, robot detects surroundings by using Lidar and generate a ``dangerous map'' to indicate both dynamic and static obstacles, which is a basis for robot to choose its best action. The proposed method is high real-time , low computation cost and do not need global states of humans.
## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. install crowd_sim and crowd_nav into pip

## Test the Algorithm
```
python test.py --policy cadrl_with_lidar
```



