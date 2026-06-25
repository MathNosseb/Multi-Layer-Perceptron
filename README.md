# Multi-Layer-Perceptron
Une libraire capable de créer des réseaux de neurones

## Le MLP

le mlp est un type de réseau de neurone capable d'apprendre et de reconnaitre
des paternes.<br>
Le réseau de neurones est capable d'apprendre un schéma ou une logique (ex XOR) et de le retrouver
et de le reproduire, il peut analyser des jeux de données

## Le projet

Ce projet est basé sur un code que j'ai entièrement fait il y a plusieurs années mais que je 
publie que récemment.<br>
Le projet est fait en python et permet de créer des réseaux de neurones utilisant une fonction d'activation:
la **sigmoïd**.

## Utilisation

Pour créer son réseau de neurones il faut télécharger le projet puis:

```python
from src.mlp import MLP #importer la class réseau MLP
```

```python
network = MLP([3,8,1], 0.2,20000) #créer un réseau de neurones
#[3,8,1] correspond au réseau -> 3 neurones d'entrée, 8 neurones en couche caché, 1 neurone en sortie
#0.2 correspond à la valeur d'exploration
#20000 correspond aux nombres d'itération d'entrainement
```
On fourni un dataset d'entrainement
```python
dataset = [
    [0,0,0,0], #les premières valeurs sont pour l'entrée du réseau, les suivantes la sortie
    [0,0,1,1], #le nombre de paramètre dépend du nombre d'entrée et de sorties que l'on a definit
]
```

```python
network.train(dataset) #entrainement sur le dataset
```

```python
output = network.try_network(test_dataset) #test sur un autre dataset et capture de la sortie
#output est une liste de liste de données
```

```python
#affichage des résultats
#pour chaque donnée dans le dataset test_dataset
#res[0] -> liste de valeurs d entrée
#res[1] -> valeur attendu
#res[2] -> valeur prédite round() à 1 décimale
for res in output:
    print("pred:", res[2], "expected:", res[1], "👍" if abs(res[1][0] - res[2][0]) < 0.1 else "👎")

```