import random
import numpy as np


def point_distributor(in_weights, in_points, in_useseed=False, in_seed=42):
    """
    Distribute in_points points over len(in_weights) groups, considering the weights.
    At least one point is given to each group.
    Then, points are assigned one by one to the group that has the highest weight per point
     if the extra point is assigned.
    If weights are equal, a random choice is made.

    in_weights: list of non-negative floats
    in_points: int larger than or equal to len(in_weights)

    Returns: list of points for each group 
    """
    if len(in_weights) <= 0:
        return None
    if in_points < len(in_weights):
        return None
    if min(in_weights) < 0:
        return None

    if in_useseed:
        random.seed(in_seed)   # Used by random.choice

    cur_pointlist = [1] * len(in_weights)
    for _ in range(in_points - len(in_weights)):
        virtual_pointlist = [j + 1 for j in cur_pointlist]
        virtual_weightperpoint = [in_weights[k] / virtual_pointlist[k] for k in range(len(in_weights))]
        virtual_choiceindexarray = np.where(np.array(virtual_weightperpoint) == max(virtual_weightperpoint))[0]
        if len(virtual_choiceindexarray) > 1:
            new_choice = random.choice(virtual_choiceindexarray)   # Deterministic if in_useseed is True
        elif len(virtual_choiceindexarray) == 1:
            new_choice = virtual_choiceindexarray[0]
        else:
            return None
        cur_pointlist[new_choice] += 1
    return cur_pointlist
