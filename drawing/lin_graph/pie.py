import random

def generate_win_tie_loss(n_games):
    results = {'model_1_win': 0, 'tie': 0, 'model_2_win': 0}
    
    for _ in range(n_games):
        outcome = random.choices(['model_1_win', 'tie', 'model_2_win'], weights=[0.6, 0.1, 0.3])
        results[outcome[0]] += 1
    
    return results

# 生成100场比赛的结果
n_games = 100
results1 = generate_win_tie_loss(n_games)
results2 = generate_win_tie_loss(n_games)
results3 = generate_win_tie_loss(n_games)

print(results1)
print(results2)
print(results3)

