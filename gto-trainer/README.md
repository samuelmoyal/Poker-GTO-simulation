# GTO Trainer

App locale qui te fait jouer des mains heads-up contre un bot GTO et compare chacune de tes décisions
à la stratégie du solveur (TexasSolver, `../TexasSolver-v0.2.0-MacOs`).

## Lancer

```bash
cd gto-trainer
python3 server.py            # ouvre http://localhost:8765
```

Python 3.13, aucune dépendance à installer. Le solveur est un binaire x86_64 qui tourne via Rosetta.

## Fonctionnement

- **Spots** : pots simples relancés (SRP) à 100bb, heads-up, ranges préflop tirées de
  `ranges/6max_range` (UTG/MP/CO/BTN vs BB, UTG/MP/CO vs BTN, SB vs BB). Voir `gto/config.py`.
- **Flop** : résolu **en avance** et rangé dans `cache/flops/` (bibliothèque). Une même solution sert pour
  n'importe quelle permutation de couleurs : le flop affiché est tiré au hasard dans les couleurs.
- **Turn et river** : résolus **à la volée** (~15-30 s chacun), avec les ranges rétrécies par les actions
  réellement jouées. Le résultat est mis en cache (`cache/streets/`).
- **Le bot** joue en tirant ses actions dans la stratégie mixte du solveur pour sa main exacte.
- **Notation** : le solveur ne donne que des fréquences (pas d'EV). Ta décision est notée selon la
  fréquence GTO de l'action choisie : ✅ ≥ 30 % · 🟡 10–30 % · 🟠 3–10 % · ❌ < 3 %.
  Score = moyenne de `min(1, fréquence / 30 %)`. Historique dans `data/history.jsonl`.

## Remplir la bibliothèque de flops

```bash
python3 precompute.py -m BTN_vs_BB -n 30          # 30 flops aléatoires pour un spot (~2-3 min chacun)
python3 precompute.py -n 10                       # 10 flops pour chaque spot
python3 precompute.py -m CO_vs_BB -f AsKd7c Qh8h3d
```

Reprenable à tout moment (les flops déjà calculés sont ignorés). Mode « Nouveau flop » dans l'app :
résout un flop aléatoire à la demande (~2-3 min d'attente).

## Limites à connaître

- **Heads-up uniquement** : TexasSolver ne résout pas le multiway.
- **Arbres de mises réduits** pour rester rapide : flop 50 % (+ relance 100 %), turn/river 66 % (+ relance
  150 % et all-in dans les re-solves). Ce n'est pas le solve « complet » d'un logiciel du commerce.
- **Approximation aux streets suivantes** : la CLI du solveur n'accepte que des poids par classe de main
  (`AKs:0.4`), pas par combo exact. Les ranges du turn/river sont donc moyennées par classe (on perd
  l'information de couleur exacte des combos).
- **Précision** : ~3 % du pot d'exploitabilité au flop, ~1 % au turn/river. Les fréquences très basses
  (< 3 %) sont du bruit de solveur.
- **Le chemin du dossier contient des espaces** : le solveur découpe ses commandes sur les espaces, donc
  `dump_result` reçoit un chemin relatif (voir `gto/solver.py`).
