# CLAUDE.md

Contexte permanent pour Claude Code sur ce projet. Les détails sont dans
`architecture.md` (racine du projet) — le lire avant toute modification structurelle.

## Objectif

Un réseau qui approxime la valeur d'équilibre au poker (NLHE heads-up), permettant de
**résoudre des flops en temps réel sur un Mac Apple Silicon**. Entraîné par distillation
des sorties d'un solver CFR.

Cas d'usage : un trainer qui simule des mains, compare le jeu de l'utilisateur à la
référence GTO, et explique les écarts.

## Ce que ce projet n'est pas

**Ce n'est pas du reinforcement learning.** C'est de la régression supervisée sur une
fonction déterministe. Pas d'exploration, pas de policy gradient, pas de TD error, pas de
replay buffer. Besoins réels : PyTorch, un dataloader, une perte Huber.

**Ne pas importer de bibliothèque de RL.** Le terme « value network » est employé au sens
de DeepStack — un réseau qui prédit des valeurs — pas au sens du RL.

## Vocabulaire

Les réseaux sont nommés par ce qu'ils **évaluent**, jamais par ce qu'ils servent à résoudre.

| Nom | Évalue | Sert à résoudre |
|---|---|---|
| `net_river` | un river-root | des **turns** |
| `net_turn` | un turn-root | des **flops** |

**Objectif du projet : `net_turn`.** Il n'existe pas de « réseau du flop » — le réseau
remplace toujours ce qui vient *après* la street résolue. La river est le seul cas sans
réseau, puisque rien ne vient après.

Autres termes :
- **street-root** — début du round de mises d'une street, board distribué, aucune mise
  encore faite. C'est le seul domaine du réseau.
- **CFV** — valeur en fraction du pot. Notre cible est une **EV conditionnelle**
  (divisée par `Z(h)`), pas une CFV au sens strict. Voir « décisions » ci-dessous.
- **masse de range** `w_h` — probabilité normalisée du combo `h` au nœud courant.

## Décisions tranchées — ne pas réinventer

1. **Tête partagée par main, jamais de sortie globale.** Le trunk produit un contexte `c`,
   puis un MLP partagé est appliqué en parallèle aux 1326 mains via un batched matmul.
2. **Pas de bucketing.** DeepStack et Supremus compressent à 1000 buckets ; on sort les
   1326 mains. Le bucketing impose un plancher d'erreur incompressible. C'est la tête
   partagée qui rend ce choix viable (un seul trunk, pas 1326 forwards).
3. **Pas de convolution, nulle part.** Ni sur la grille 13×13, ni sur un tenseur
   rang × couleur. Débat tranché en §8.3 : le rang n'est pas invariant par translation, la
   couleur n'a pas de structure ordinale, l'interaction main↔board est non locale. Aucun
   travail publié comparable n'en utilise. En cas de plateau → attention structurée sur les
   1326 mains, pas un CNN.
4. **La cible est une EV conditionnelle** — `v_i(h)` divisé par `Z(h)`. Ne jamais
   « simplifier » en retirant cette division : prédire des EV plutôt que des CFV améliore
   la précision jusqu'à 15 % et permet des réseaux plus petits (AAAI 2023).
5. **Couche zero-sum en dur.** `s = r1·v1 + r2·v2`, soustraire `s/2` aux deux vecteurs.
   Les marginales doivent être **corrigées du card removal** avant, sinon biais systématique.
6. **Normalisation des cibles par `(1 + SPR)`.** Sans elle, les spots profonds dominent le
   gradient et le réseau devient mauvais en jeu court sans signe visible.
7. **Encodage de Fourier du SPR** (13 dims). La valeur a des coudes en SPR là où une ligne
   de mise devient possible ; un scalaire seul les capture mal.
8. **Décomposition de la sortie** : baseline équité (exact, non appris) + branche bilinéaire
   (linéaire en `r2`) + tête MLP. Le réseau n'apprend que la réalisation d'équité.
9. **Features par main calculées, pas apprises** : rang de force sur le board, équité
   corrigée du removal, masse adverse bloquée, outs. Ce sont les sous-fonctions exactes du
   calcul de la cible ; les faire réapprendre est du gaspillage.
10. **Perte pondérée par `w_h`** (avec plancher `ε = 0.01`), Huber et non MSE.
11. **Ranges d'entraînement on-policy.** Les turn-roots doivent venir du simulateur ou de
    solutions flop existantes, jamais d'un tirage aléatoire. Des ranges aléatoires
    produisent un réseau inutilisable (résultat mesuré par ReBeL).
12. **Pas d'échantillonnage de mains à l'entraînement.** Le raccourci valable pour un policy
    net est interdit ici : la couche zero-sum exige les 1326 sorties.
13. **CFR entièrement vectorisé** sur les mains, tenseurs `(n_infosets, 1326, n_actions)`.
    Toute boucle Python sur les combos est un bug de performance.
14. **Batcher les appels réseau aux feuilles.** Un forward sur `(n_feuilles * 49, D)`,
    jamais une boucle. Rafraîchir toutes les `k` itérations (k = 5–10).

## Pièges MPS

- Entraîner sur `mps`, pas sur CPU (~0,2 s/pas contre 3–6 s).
- Lancer avec `PYTORCH_ENABLE_MPS_FALLBACK=1` et traiter tout warning de fallback comme un
  bug : un seul fallback dans la boucle chaude coûte un facteur 10.
- bf16 sur le trunk, **fp32 sur la tête de valeur et la couche zero-sum**.
- `torch.compile` est partiel sur MPS. Mesurer, ne pas supposer.

## Tests de validation obligatoires

- **Extraction des labels** : sur un échantillon, `|r1·v1 + r2·v2| < 1e-6`. Sinon il y a un
  bug, presque toujours dans le card removal au showdown.
- **Sensibilité aux ranges** : remplacer la range adverse par une range très différente
  (polarisée vs linéaire) et vérifier que la sortie bouge substantiellement. Mode d'échec le
  plus courant et le plus silencieux — un réseau qui a appris une moyenne conditionnée au
  board et ignore les ranges aura une perte correcte à l'entraînement et sera inutile au
  runtime.
- **Permutation de couleurs** : vérifier que board permuté + indices réindexés donne la même
  sortie. Une incohérence ici n'a aucun symptôme visible.

## Évaluation

**Ne jamais utiliser l'accuracy de match d'action.** **Ne jamais utiliser le résidu
zero-sum** comme métrique — la couche l'impose par construction.

```
primaire   : MAE pondérée par range, en % du pot        cible < 0,3 %
secondaire : perte d'EV du resolving flop, en bb/100
diagnostic : erreur par texture de board et bucket de SPR
```

Split par `solve_id` **et** hash de board, jamais par nœud. Les nœuds d'un même solve sont
fortement corrélés. **Raisonner en solves**, pas en labels : la taille d'échantillon
effective est le nombre de solves distincts.

## Conventions

- Tout composant coûteux vient avec un micro-benchmark dans `bench/`.
- Les spots du holdout n'entrent jamais à l'entraînement — vérifier par hash.
- Les explications en langage naturel ne citent que des nombres présents dans le JSON de
  faits. Le vérificateur qui l'impose est obligatoire.
- Avant d'entraîner quoi que ce soit : mesurer la MAE du **baseline équité seul**. C'est le
  plancher à battre et ça valide le pipeline de labels.

## État du projet

<!-- à tenir à jour : étape courante de la roadmap de architecture.md §15 -->
Étape 4/4b (voir §15, « Avancement »). Étapes 0–1 levées : le solver retenu
(`tools/postflop-solver`, wrapper `tools/turn-labels`) renvoie les EV par main directement.
Génération des données en cours (`model/scripts/gen_flops.py`, `gen_labels.py`). Code :
`model/gtonet`, tests `model/tests`, bench `bench/`.
