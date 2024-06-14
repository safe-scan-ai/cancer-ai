import torch

scores = torch.tensor([0.0008, 0.0000, 0.6000, 1.0000])
scattered_rewards = torch.tensor([0.0000, 0.0000, 0.6000, 1.0000])
mask = [True, True, False, True]
def update_scores():
    new_scores = 0.1 * scattered_rewards[mask] + (1 - 0.1) * scores[mask]
    print(new_scores)

update_scores()
