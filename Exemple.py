from src.mlp import MLP

network = MLP([3,8,1], 0.2,20000)

dataset = [
    [0,0,0,0],
    [0,0,1,1],
]

test_dataset = [
    [1,1,1,1],
    [1,1,0,0],
    [1,0,1,1],
    [0,1,1,1],
    [1,0,0,0],
    [0,1,0,0]
]


network.train(dataset)

output = network.try_network(test_dataset)

for res in output:
    print("pred:", res[2], "expected:", res[1], "👍" if abs(res[1][0] - res[2][0]) < 0.1 else "👎")

