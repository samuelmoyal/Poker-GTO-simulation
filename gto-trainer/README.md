# GTO Trainer

App locale qui te fait jouer des mains heads-up contre un bot GTO et compare chacune de tes décisions
à la stratégie du solveur (postflop-solver, vendorisé dans `../tools/postflop-solver`).

## Lancer

```bash
cd gto-trainer
python3 server.py            # ouvre http://localhost:8765
```

Python 3.13. Le solveur est un binaire Rust à compiler une fois :

```bash
(cd ../tools/turn-labels && cargo build --release)      # produit target/release/street_tree
```

Les fichiers de ranges préflop viennent encore du dossier `../TexasSolver-v0.2.0-MacOs/ranges/6max_range`
(voir le README à la racine) ; le solveur TexasSolver lui-même n'est plus utilisé.

Pour le mode « Main IRL » (voir plus bas), lance le serveur avec l'environnement du modèle, qui a torch :

```bash
../model/.venv/bin/python server.py
```

Sans torch, le serveur démarre quand même et seuls les modes « Bibliothèque » et « Nouveau flop » marchent.

## Fonctionnement

- **Spots** : pots simples relancés (SRP) à 100bb, heads-up, ranges préflop tirées de
  `ranges/6max_range` (UTG/MP/CO/BTN vs BB, UTG/MP/CO vs BTN, SB vs BB). Voir `gto/config.py`.
- **Flop, trois modes** (menu de l'interface) :
  - **Main IRL** : aucun précalcul. Le flop est tiré au hasard et résolu **sur le moment par le réseau de valeur**
    (`gto/netflop.py`, ~3 s sur M2) : un CFR tronqué au flop dont les feuilles sont valorisées par `net_turn`.
    C'est une approximation (perte d'EV réelle mesurée : ~0,9 % du pot sur un flop de validation, autres flops en cours de mesure), avec l'arbre de
    flop que le réseau connaît : une taille de mise (50 %), ni relance ni all-in. Réglages : `GTO_NET_CKPT`
    (`tiny_v2` par défaut, `small_v2` plus précis et plus lent), `GTO_NET_DEVICE` (`mps` ou `cpu`), `GTO_NET_ITERS`.
  - **Bibliothèque** : flop résolu en avance par le solveur exact et rangé dans `cache/flops/`.
  - **Nouveau flop, solveur exact** : résolution à la demande (~20-60 s).
  Détail de la bibliothèque : Une même solution sert pour
  n'importe quelle permutation de couleurs : le flop affiché est tiré au hasard dans les couleurs.
- **Turn et river** : résolus **à la volée** (0,1 à 1 s chacun), avec les ranges rétrécies **combo par combo** par les
  actions réellement jouées. Le résultat est mis en cache (`cache/streets/`).
- **Le bot** joue en tirant ses actions dans la stratégie mixte du solveur pour sa main exacte.
- **Notation** : le solveur ne donne que des fréquences (pas d'EV). Ta décision est notée selon la
  fréquence GTO de l'action choisie : ✅ ≥ 30 % · 🟡 10–30 % · 🟠 3–10 % · ❌ < 3 %.
  Score = moyenne de `min(1, fréquence / 30 %)`. Historique dans `data/history.jsonl`.

## Remplir la bibliothèque de flops

```bash
python3 precompute.py -m BTN_vs_BB -n 30          # 30 flops aléatoires pour un spot (~15-20 s chacun)
python3 precompute.py -n 10                       # 10 flops pour chaque spot
python3 precompute.py -m CO_vs_BB -f AsKd7c Qh8h3d
```

Reprenable à tout moment (les flops déjà calculés sont ignorés). Mode « Nouveau flop » dans l'app :
résout un flop aléatoire à la demande (~15-20 s d'attente).

## Limites à connaître

- **Heads-up uniquement** : le solveur ne résout pas le multiway.
- **Arbres de mises réduits** pour rester rapide : flop 50 % (+ relance 100 %), turn/river 66 % (+ relance
  150 % et all-in dans les re-solves). Ce n'est pas le solve « complet » d'un logiciel du commerce.
- **Précision** : le solveur s'arrête à 0,5 % du pot d'exploitabilité (0,3 % à la river). Les fréquences très
  basses (< 3 %) restent du bruit de solveur.
- **Bibliothèque de flops à régénérer** : les flops calculés avec l'ancien solveur (TexasSolver) sont ignorés
  (l'identifiant de profil a changé) ; `precompute.py` les recalcule avec le solveur actuel.
