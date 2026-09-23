# Architecture — solveur flop rapide par réseau de valeurs

Document de référence. Remplace la version initiale : la nomenclature des réseaux et
le sens du dépassement de profondeur y étaient ambigus.

---

## 1. Cadrage

Le problème n'est pas du reinforcement learning. C'est de la **régression supervisée sur
une fonction déterministe** : un solver CFR produit la vérité terrain, un réseau apprend
à la reproduire en quelques millisecondes.

Pas d'exploration, pas de policy gradient, pas de TD error, pas de replay buffer.
Besoins réels : PyTorch, un dataloader, une perte Huber. **Ne pas importer de
bibliothèque de RL**, ce serait du poids mort.

Le terme « value network » reste juste au sens de DeepStack et GTO Wizard — un réseau
qui prédit des valeurs — pas au sens du RL.

### 1.1 Pourquoi un réseau est plus rapide qu'un solver

Le solver **cherche**, le réseau **se souvient**. Un solve, c'est ~500 itérations de CFR
traversant l'arbre entier, soit 10^11–10^12 opérations pour un turn. Un forward du réseau
coûte ~1,5 GFLOP. Le solver repart de zéro à chaque fois et n'a aucune mémoire ; le
réseau a compressé la structure commune de millions de solves dans ses poids.
C'est de l'**inférence amortie**.

Le coût du réseau est **constant** : il ne grandit ni avec la profondeur de l'arbre ni
avec le nombre de tailles de mise. Celui du solver explose.

### 1.2 Le gain décisif n'est pas la vitesse brute

Le réseau ne remplace pas le solver, il lui **raccourcit l'horizon**. On fait tourner un
vrai CFR, mathématiquement correct, sur un arbre 1000× plus petit. On n'échange pas
« recherche » contre « prédiction », mais « chercher loin » contre « chercher près +
estimer le reste ».

Ordre de grandeur publié par GTO Wizard : un flop exact avec plusieurs tailles prendrait
plusieurs jours et jusqu'à des téraoctets de RAM. Précision atteignable avec le réseau :
0,1 à 0,3 % du pot de Nash distance.

---

## 2. Nomenclature — à ne pas confondre

**Le réseau évalue l'état futur ; il remplace tout ce qui vient APRÈS la street résolue.**

```
résoudre le flop  → CFR sur le flop,  réseau pour turn+river
résoudre le turn  → CFR sur le turn,  réseau pour river
résoudre la river → CFR exact, aucun réseau  (rien ne vient après)
```

Les réseaux sont donc nommés par ce qu'ils **évaluent** :

| Réseau | Évalue | Sert à résoudre |
|---|---|---|
| `net_river` | un river-root | des **turns** |
| `net_turn` | un turn-root | des **flops** |

Il n'existe pas de « réseau du flop ». Un réseau ne prédit jamais une stratégie, il prédit
la valeur d'un état futur.

**Objectif du projet : `net_turn`.** C'est lui qui permet de résoudre des flops en temps réel.

---

## 3. Convention d'état : le street-root

Le domaine du réseau est un type d'état précis : **le début du round de mises d'une
street**, board de cette street distribué, aucune mise encore faite sur cette street.

C'est exactement ce dont le CFR tronqué a besoin à ses feuilles, et c'est un état bien
plus simple qu'un nœud quelconque : pas d'historique de betting à encoder puisqu'il n'y
en a pas encore. Tout le passé est résumé par les ranges et le pot.

---

## 4. Entrées et sorties de `net_turn`

### 4.1 Entrées

```
board_flop : 52    multi-hot (3 bits actifs)
board_turn : 52    multi-hot (1 bit)
r1         : 1326  float, normalisé à somme 1
r2         : 1326  float, normalisé à somme 1
log_spr    : 1     float, log(stack effectif / pot)
```

Trois points non évidents :

**Le pot disparaît.** La sortie étant en fraction du pot, le problème est invariant
d'échelle : seul le SPR compte. Pot 10 bb / 30 bb derrière et pot 100 bb / 300 bb derrière
sont le même état. Une dimension entière de l'espace d'entrée éliminée gratuitement.

**Les ranges sont normalisées.** Les reach probabilities brutes ne somment pas à 1 ; leur
échelle absolue n'affecte pas la stratégie d'équilibre. La normalisation est aussi une
précondition de la couche zero-sum.

**Le turn est séparé du flop.** `A♠7♦2♣ / K♥` et `A♠K♥2♣ / 7♦` n'ont pas la même
signification stratégique — quelle carte est arrivée en dernier change tout.

### 4.2 Sorties

```
v1, v2 : 1326 float chacun
```

`v_i[h]` = valeur de la main `h` pour le joueur `i`, conditionnellement à la détenir, en
supposant le jeu à l'équilibre à partir d'ici.

```
EV_i(h) = Σ_{h' ∩ h = ∅}  [ r_j(h') / Z(h) ] · EV(h contre h' sous σ*)
          Z(h) = Σ_{h' ∩ h = ∅} r_j(h')

v_i(h)  = [ EV_i(h) / P − 1/2 ] / (1 + SPR)
```

- `Z(h)` est le **card removal** : renormalisation de la range adverse par main, différente
  pour chaque `h`.
- Le `− 1/2` transforme le jeu à somme constante (`EV_1 + EV_2 = P`) en somme nulle.
  Sans lui la couche zero-sum est fausse.
- La division par `1 + SPR` empêche les spots profonds de dominer le gradient. Sans elle,
  le réseau devient bon en jeu profond et mauvais en jeu court sans qu'on comprenne pourquoi.

**La cible est une EV conditionnelle, pas une CFV — c'est délibéré.** La division par `Z(h)`
fait que `v_i(h)` est la valeur *sachant qu'on détient `h`*, et non la valeur contrefactuelle
pondérée par la reach probability. Zarick & Tejwani (AAAI 2023) mesurent que prédire des EV
puis les multiplier par les reach probabilities, plutôt que prédire directement des CFV,
réduit le bruit, améliore la précision jusqu'à 15 % et permet à des réseaux plus petits et
plus rapides de dépasser le standard. Un réseau à 2 couches y fait aussi bien ou mieux qu'un
réseau à 7 couches sur la cible CFV — signe que l'état de l'art sur-dimensionne ses modèles
pour compenser une cible mal définie. **Ne pas « simplifier » en retirant la division par
`Z(h)`**, ce serait une régression.

### 4.3 Couche zero-sum

```python
s  = (r1 * v1).sum(-1, keepdim=True) + (r2 * v2).sum(-1, keepdim=True)
v1 = v1 - s / 2
v2 = v2 - s / 2
```

Contrainte imposée en dur, pas espérée du réseau. Astuce de DeepStack ; réduit l'erreur
d'un facteur notable pour zéro paramètre.

**Subtilité qui mord** : l'identité `r1·v1 + r2·v2 = 0` n'est exacte que si `r1` et `r2`
sont les marginales **corrigées du card removal**, pas les ranges normalisées naïvement.
La différence est petite mais systématique ; non corrigée, elle introduit un biais que le
réseau apprendra à compenser n'importe comment.

---

## 5. Obtenir les valeurs quand le solver ne donne que des fréquences

> **Statut — obsolète pour les labels.** Le solver retenu (`postflop-solver`, §13.1) renvoie
> directement `expected_values(joueur)` par main : c'est l'EV conditionnelle du §4.2, vérifiée
> (`Σ_h c₁(h)·EV₁(h) + Σ_h c₂(h)·EV₂(h) = pot` à 1e-6, avec `c` = marginales corrigées du card
> removal). L'extraction ci-dessous n'est plus nécessaire pour les cibles. Le calcul de
> showdown avec card removal reste utile pour les **features** et le **baseline équité** (§8.6).

Les EV ne sont pas une information supplémentaire détenue par le solver. **Elles sont une
fonction déterministe des fréquences.** Une fois `σ` connu partout dans l'arbre, il ne
reste qu'un calcul d'espérance.

```
passe avant  : propager les reach probabilities depuis la racine
               r_i(nœud·a) = r_i(nœud) · σ_i(a|nœud)   si i agit
               r_i(nœud·a) = r_i(nœud)                  sinon

passe arrière :
  terminal fold     : ± pot, pondéré par la range adverse (card removal)
  terminal showdown : somme sur la range adverse selon la comparaison
  nœud où i agit    : v_i = Σ_a σ_i(a|nœud) · v_i(nœud·a)
  nœud où j agit    : v_i = Σ_a v_i(nœud·a)   (les reach de j portent déjà σ_j)
```

**Coût : une traversée**, soit ~1 itération de CFR contre les ~500 du solve. 0,2 % du budget.

**Condition bloquante** : il faut les fréquences de **tout l'arbre**, pas seulement du nœud
racine. À vérifier dans l'export du solver avant toute autre chose.

**Le morceau pénible — le showdown.** Comparer 1326 mains contre 1326 avec card removal
fait 1,7 M de comparaisons par board en naïf. Astuce standard : trier les mains par force,
sommes cumulées, puis soustraire les combos bloqués → O(n log n).

**Test de validation obligatoire** : sur un échantillon de labels, `|r1·v1 + r2·v2|` doit
être sous `1e-6`. Sinon il y a un bug dans l'extraction — presque toujours le card removal
au showdown.

---

## 6. Deux chemins de labellisation

### 6.1 Chemin retenu — turn+river exact

```
solver : CFR depuis un turn-root, arbre développé turn ET river
         jusqu'aux nœuds terminaux réels
    ↓ CFV à la racine du turn (via §5)
net_turn : entraîné dessus
    ↓ utilisé aux feuilles
CFR flop tronqué au turn-root → résout le flop au runtime
```

Un seul réseau, labels **exacts**, pas de bootstrap, pas de dérive à mesurer. L'unique
source d'erreur est l'approximation du réseau, directement mesurable sur un holdout.

Le flop n'entre dans aucun solve. Il n'est résolu qu'au runtime, tronqué.

### 6.2 Chemin alternatif — bootstrap par street

```
vague 1 : river exacte              → net_river   (labels exacts)
vague 2 : turn, CFR + net_river gelé → net_turn    (labels approximatifs)
```

À la vague 2, chaque feuille de l'arbre turn est évaluée en moyennant `net_river` sur les
**48** cartes river possibles (49 depuis un flop). Les cartes dans les mains des joueurs
ne sortent pas du deck : elles sont gérées par le card removal dans la pondération.

### 6.3 Arbitrage

Ce n'est pas « exact contre approximatif », c'est **exactitude par label contre nombre de
labels**, à budget de calcul constant.

| | Turn+river exact | Turn bootstrapé |
|---|---|---|
| Coût par label | ~2 s (mesuré, `postflop-solver`) | secondes |
| Labels par jour-CPU | ~40 000 (mesuré) | dizaines de milliers |
| Erreur du label | nulle | celle de `net_river` |
| Réseaux à entraîner | 1 | 2 |

L'erreur totale a deux composantes : erreur de label et erreur de généralisation. La
seconde dépend du **nombre de solves distincts**, pas du nombre de labels — les 1326 mains
d'un nœud sont fortement corrélées, et les nœuds d'un solve aussi. **Raisonner en solves.**

**Décision** : commencer par 6.1. Plus simple, système complet de bout en bout, et les
solves exacts produits servent de toute façon de holdout.

**Mise à jour (mesures)** : avec `postflop-solver`, un label exact coûte ~2 s, du même ordre
que le bootstrap. Le coût ne justifie donc plus 6.2 ; le facteur limitant est le nombre de
**flops distincts** (donc de distributions de ranges), pas le nombre de labels par flop.

**Critère de bascule** : si la perte d'EV stagne quand on ajoute des solves → limité par le
label, inutile d'en générer plus. Si elle continue de baisser → limité par la quantité, le
bootstrap devient rentable.

**Préparation gratuite** : lors des solves turn+river, enregistrer aussi les CFV aux
river-roots rencontrés. Déjà calculées par la passe arrière, coût marginal nul, et le
dataset `net_river` est prêt si on bascule.

---

## 7. Distribution des turn-roots — le piège numéro un

On ne résout jamais de flop, donc on ne sait pas d'où viennent les ranges turn. Or elles ne
sortent pas de nulle part : ce sont des ranges flop propagées à travers un round de mises.

**Des ranges tirées au hasard produisent un réseau inutilisable.** ReBeL l'a mesuré
explicitement : un value net entraîné sur des public belief states aléatoires n'apprend
rien d'utile. DeepStack contournait le problème avec une distribution construite à partir
de connaissance experte.

Sources de turn-roots réalistes, par ordre de qualité :

1. **Le simulateur** — le faire jouer et enregistrer les états et ranges à l'entrée du turn.
   C'est son meilleur usage : générateur de distribution, pas de récompenses.
2. **Solutions flop pré-calculées** — seules leurs ranges turn importent, pas leur précision.
3. **Stratégie flop heuristique** — c-bet raisonnable sur ranges preflop standard, propagé
   au turn. Grossier, mais très supérieur à l'aléatoire.

Mélanger 70 % on-policy / 30 % perturbations (bruit multiplicatif, mélange avec uniforme)
pour couvrir les ranges nodelockées et les adversaires non-GTO du trainer.

---

## 8. Architecture de `net_turn`

### 8.0 Dériver l'architecture du calcul exact

Écrire le calcul exact de la cible, puis le lire comme un cahier des charges.

```
v₁(h) = (1/Z(h)) · Σ_{h'} r₂(h')·𝟙[h∩h'=∅] · [
             Σ_{a ∈ arbre turn} σ*(a|h,h') · (
                 si terminal : payoff(a, h, h')
                 sinon       : (1/48) · Σ_{c ∈ deck} V_river(h, h', c, pot_a, r_a)
             )
         ]
```

Point crucial : `σ*` est **l'équilibre**, qui n'a pas de forme fermée — c'est le point fixe
d'un problème de minimisation de regret. Le réseau n'approxime donc pas une espérance
qu'on saurait écrire, il approxime **la valeur d'un point fixe**.

Les sept opérations effectuées, et ce que chacune impose :

| # | Opération | Conséquence architecturale |
|---|---|---|
| 1 | tirer la carte turn (48 branches) | récursion à 2 niveaux → profondeur du trunk |
| 2 | tirer la carte river (47 branches) | idem |
| 3 | évaluer les mains à 7 cartes | fournir le **rang de force** en feature |
| 4 | comparer `h` contre `h'` | comparateur d'ordre, pas de géométrie de cartes |
| 5 | pondérer par `r₂`, retirer les conflits | card removal exact en feature |
| 6 | moyenner sur l'arbre de mises, pondéré par `σ*` | enveloppe non lisse en SPR |
| 7 | moyenner sur les cartes | invariances à imposer |

**Opérations 3–4 — le showdown est un comparateur d'ordre.** Le résultat ne dépend que du
rang de la main dans l'ordre des 7462 forces. `A♠A♦` et `A♥A♣` sur le même board sont
identiques. C'est une fonction combinatoire arbitraire, parfaitement non lisse : la faire
réapprendre au réseau est du gaspillage. → **Fournir le rang de force normalisé sur le
board courant dans `feat_h`.** Une feature, une opération de moins à apprendre.

**Opération 5 — le card removal est une soustraction de masse**, bilinéaire et exactement
calculable. → **Fournir l'équité brute déjà corrigée du removal, et la masse adverse
bloquée par chaque carte de la main.** Le réseau ne doit pas réapprendre la combinatoire
des blockers ; il doit apprendre ce qu'ils valent stratégiquement.

**Opération 6 — l'enveloppe est non lisse en SPR.** À l'équilibre chaque joueur joue proche
du max sur ses actions : la valeur est une enveloppe supérieure lissée. Le nombre de
tailles de mise disponibles change qualitativement avec la profondeur, donc la fonction a
des **coudes** en SPR, là où une ligne devient possible ou impossible. Un scalaire
`log_spr` dans le trunk capture mal ça. → **Encodage de Fourier du SPR** :

```python
spr_enc = cat([log_spr] + [f(2**k * pi * log_spr) for k in range(6) for f in (sin, cos)])
# 13 dimensions
```

**Opération 7 — la moyenne sur les cartes est une moyenne sur permutations.** → DeepSets
sur le flop (somme d'embeddings, pas de concaténation ordonnée) ; embeddings factorisés
rang ⊗ couleur ; et surtout : l'augmentation par permutation de couleurs n'est pas une
régularisation cosmétique, c'est **la symétrie exacte du calcul**. Idéalement une
contrainte d'architecture plutôt qu'une augmentation — **canonicaliser le board avant
l'encodage** ramène 24 entrées équivalentes à une seule et divise l'espace d'entrée d'autant.

### 8.1 La structure bilinéaire — le point le plus important

La forme globale du calcul est :

```
v₁ ≈ M(board, SPR, r₁) · r₂
```

Au showdown cette structure est **exacte** : `v₁(h) = Σ_h' r₂(h') · résultat(h,h')` est
strictement linéaire en `r₂`. Sur l'arbre de mises elle cesse de l'être, puisque `σ*` dépend
de `r₂` non linéairement — mais la **dépendance dominante reste linéaire**.

Un MLP qui écrase `r₂` dans un vecteur de 384 dims puis le mélange dans six couches non
linéaires doit reconstruire cette linéarité. C'est inutilement dur. → **Donner la structure
explicitement** :

```python
v_base  = equite_realisee_baseline(E, r2)      # exact, non appris
v_bilin = einsum('hk,bk->bh', W(E), R2_full)   # linéaire en r₂ par construction
v_mlp   = tete_par_main(E, c)                  # correction non linéaire
v = v_base + v_bilin + v_mlp
```

Le réseau ne prédit plus la valeur : il prédit **l'écart entre la valeur d'équilibre et
l'équité brute**, c'est-à-dire la réalisation d'équité, la position, l'initiative, la
capacité à bluffer. Quantité bornée, centrée, bien plus facile à apprendre.

Le baseline d'équité a en plus deux vertus pratiques : il donne un modèle non trivial dès
le pas 0, et il valide le pipeline d'extraction de labels avant tout paramètre appris.
C'est exactement ce que fait l'entrée MIT de GTO Wizard en l'absence de réseau — chaque
joueur reçoit la valeur correspondant à son équité.

### 8.2 Représentation des ranges

Envoyer `r1` et `r2` bruts dans un MLP marche mal : le MLP traite l'indice 847 comme une
coordonnée arbitraire et doit apprendre par cœur qu'il correspond à K♠J♠, proche de K♦J♦.
On lui fait réapprendre la structure des cartes 1326 fois.

Solution : exprimer une range **dans l'espace des mains**.

```
e_h = embedding de la main h  (2 cartes + interaction board)
R_i = Σ_h r_i[h] · e_h
```

`R_i` a la même dimension qu'une main. Deux ranges similaires donnent des `R` proches
automatiquement, et une range jamais vue est représentée correctement du premier coup —
la représentation est **construite**, pas apprise. C'est ce qui donne la généralisation aux
nodelocks et aux adversaires exotiques.

Une somme unique écrase trop d'information :

```python
R_i = cat([
    (r_i[:, None] * E).sum(0),                    # moment d'ordre 1
    (r_i[:, None] * E * E).sum(0),                # dispersion
    attention_pool(E, weights=r_i, n_queries=8),  # 8 vues apprises
])
```

### 8.3 Grille rang × couleur et convolutions — portée et limites

**L'idée** : représenter une range comme un tenseur structuré (rang × couleur, ou la grille
13×13 suited/offsuit familière aux joueurs) et y appliquer des opérations spatiales.

**Ce qui est juste.** Il y a un précédent mesuré. Poker-CNN encode chaque carte dans une
matrice binaire 4×13 : un pair devient deux cartes dans la même colonne, une couleur cinq
cartes sur la même ligne — des motifs détectables sans tri de mains ni traitement explicite
des isomorphismes de couleur. L'ablation d'OpenHoldem confirme que la représentation
tensorielle bat la représentation vectorielle plate.

**Ce qui ne transfère pas.** Ces travaux encodent **un ensemble de cartes** (ma main + le
board). Une range est une **distribution sur 1326 paires**. Les objets ne sont pas de même
nature, et trois obstacles apparaissent :

1. **Le rang n'est pas invariant par translation.** Une convolution partage le même noyau
   partout, ce qui suppose que `AK` est à `A` ce que `43` est à `4`. Faux : l'as est
   singulier (nut flush, broadway, roue). Il faudrait des noyaux dépendants de la position,
   c'est-à-dire un MLP avec des étapes en plus.
2. **La couleur n'a aucune structure ordinale.** Il n'existe pas de couleur « adjacente ».
   Convoluer le long d'un axe de couleurs n'a pas de sens ; ce qu'on veut est
   l'**équivariance par permutation** (DeepSets, attention). Poker-CNN s'en tire parce que
   « couleur = même ligne » est un motif détectable par pooling par ligne, pas parce que
   les couleurs seraient ordonnées.
3. **L'interaction main↔board est non locale dans la grille.** Sur `A♠7♦2♣`, les cellules
   pertinentes sont tous les `Ax`, tous les `7x`, tous les `2x` et toutes les mains
   assorties à pique — un motif en étoile dispersé, pas un bloc compact. Le prior de
   localité de la convolution est ici exactement le mauvais.

Et surtout : **les features par main (§8.4) sont déjà le résultat de ce raisonnement
spatial, calculé exactement.** Catégorie de main faite, outs, équité, blockers — ce sont
précisément les quantités qu'une convolution tenterait d'approximer. Les fournir est
strictement supérieur à les faire apprendre.

**Verdict : aucune convolution dans l'architecture.** Décision tranchée, ne pas y revenir
sans élément nouveau.

Raison décisive, au-delà des trois obstacles ci-dessus : **aucun des travaux qui ont fait
ce qu'on fait n'utilise de convolution** (§8.7). Sur plusieurs équipes et des centaines de
millions de mains, la représentation qui gagne est systématiquement embeddings de cartes
factorisés + sommes invariantes par permutation + MLP. Les travaux convolutifs
(Poker-CNN, la variante PokerCNN d'OpenHoldem) encodent un joueur unique face à un board,
pas une distribution sur 1326 paires — et dans la lignée même qui a introduit le tenseur
structuré, l'architecture factorisée dépasse le CNN pur.

Ce qui reste valable de l'intuition, pour mémoire :

- **Préflop, l'idée serait bonne** : pas de board, la grille 13×13 est lisse et les ranges
  d'équilibre y forment des blocs contigus. Si le projet s'étend un jour au préflop, la
  question se rouvre — mais seulement là.
- **En cas de plateau**, la bonne généralisation de « opérations spatiales sur la range »
  est l'attention sur les 1326 mains avec encodage positionnel structuré : elle exprime la
  localité en rang et l'équivariance en couleur, et apprend le motif conditionné au board
  au lieu de le supposer. C'est l'extension à tenter, pas un CNN.

Note honnête sur la lignée : dans OpenHoldem, l'architecture pseudo-siamoise qui traite
séparément cartes et actions **dépasse** la variante PokerCNN. Même dans la famille qui a
introduit le tenseur structuré, la convolution pure a été supplantée par une architecture
plus factorisée.

La généralisation moderne de l'intuition « opérations spatiales sur la range » n'est pas la
convolution mais l'**attention sur les 1326 mains avec encodage positionnel structuré** :
elle exprime la localité en rang et l'équivariance en couleur, et apprend le motif
conditionné au board au lieu de le supposer. C'est l'extension à tenter en cas de plateau,
avant tout CNN.

### 8.4 Schéma complet

```
encodeur cartes
  emb_carte : 52 → 64, factorisé rang(13×32) ⊕ couleur(4×32)
  canonicalisation des couleurs du board en amont
  flop : DeepSets sur 3 cartes (somme, invariant à l'ordre)
  turn : slot séparé
  → b (256)

features par main            E ∈ R^{1326 × 128}
  = emb(c1) ⊕ emb(c2) ⊕ feat_h
  feat_h (≈28 dims, toutes calculées exactement) :
    rang de force normalisé sur le board          ← op. 3-4
    équité brute corrigée du card removal         ← op. 5
    masse adverse bloquée par carte               ← op. 5
    catégorie de main faite, kicker rank
    type de tirage, nb d'outs

encodeur ranges
  R1, R2 = pool(E, r1), pool(E, r2)      → 384 chacun
           (moments d'ordre 1 et 2 + attention_pool)

encodage SPR
  spr_enc : Fourier, 13 dims                      ← op. 6

trunk
  cat(b, R1, R2, spr_enc, scalaires)
  6 blocs résiduels × 1024, pré-LayerNorm, GELU
  → c (512)

sorties (§8.1)
  v = equite_realisee_baseline(E, r₂)      exact, non appris
    + bilineaire(E, R₂_full)               linéaire en r₂
    + tete_par_main(cat(E, c))             MLP partagé 3 × 384 → 2 scalaires
  → dénormalisation par (1 + SPR), couche zero-sum
```

**≈ 18 M paramètres.**

### 8.5 Justifications complémentaires

**Tête par main, jamais de sortie globale.** Le MLP partagé est appliqué en parallèle aux
1326 mains via un batched matmul. Sortir `1326 × n_actions` depuis un vecteur global ne
marche pas.

**Deux scalaires par main plutôt que deux passes.** L'asymétrie entre joueurs est déjà
portée par `c`, qui contient `R1` et `R2` dans cet ordre. Coût de la tête divisé par deux.

**Pas d'attention entre mains au départ.** Pour un policy net elle compte (le mixing dépend
de l'équilibre global de la range). Pour un value net beaucoup moins : la valeur d'une main
dépend de la range adverse, déjà dans `c`. DeepStack s'en sortait avec un MLP de 7 couches
de 500 unités. En cas de plateau → attention structurée (§8.3), pas un CNN.

**Profondeur plutôt que largeur.** La récursion exacte a deux niveaux imbriqués (turn puis
river) et chaque bloc résiduel raffine. Passer de 6 à 8 blocs est préférable à élargir de
1024 à 2048.

### 8.6 Limite de ce raisonnement

Le calcul exact donne les bons **inductive biases**, pas une garantie. Il dit quelles
symétries respecter et quelles sous-fonctions ne pas faire réapprendre. Il ne dit rien sur
la partie difficile — comment `σ*` dépend des ranges, qui est précisément ce qui n'a pas de
forme fermée. C'est pourquoi DeepStack s'en sortait avec un MLP simple : une fois les
symétries et les features en place, le reste est une fonction assez régulière.

**Ordre d'implémentation** : d'abord la branche baseline équité **seule**, et mesurer sa
MAE. Elle donne le plancher à battre et valide tout le pipeline d'extraction de labels
avant d'introduire le moindre paramètre appris.


### 8.7 Prior art — architectures comparables

Décisions déjà tranchées par d'autres équipes. Ne pas rouvrir ces débats sans élément nouveau.

| Système | Sortie | Architecture | Ce qu'on en retient |
|---|---|---|---|
| **DeepStack** (2017) | 1000 buckets | 7 × 500 FC, PReLU, couche zero-sum externe | la découpe flop/turn + la couche zero-sum |
| **Supremus** (2020) | 1000 buckets | identique à DeepStack, entrée aplatie de 2001 | les détails d'implémentation pèsent plus que l'archi |
| **Deep CFR** (2019) | logits d'actions | 7 couches, ~99k paramètres | l'encodage de cartes, repris tel quel |
| **Poker-CNN** (2016) | actions | tenseur 4×13 par carte, CNN | écarté, voir §8.3 |
| **AlphaHoldem** (2022) | actions | pseudo-siamoise, cartes et actions séparées | 4 ms par décision, ordre de grandeur cible |
| **GTO Wizard AI** | 1326, sans abstraction | non publiée | la cible à viser |

**DeepStack** : réseau feedforward standard, sept couches cachées entièrement connectées de
500 nœuds, PReLU, enchâssé dans un réseau externe qui force les valeurs à satisfaire la
propriété de somme nulle. Deux réseaux — l'un après le flop, l'autre après la carte turn —
plus un réseau auxiliaire préflop pour accélérer le resolving des premières actions. C'est
exactement la découpe par street retenue ici (§2).

**Supremus** : même architecture. Entrée = tableau aplati de longueur 2001 (les
distributions des deux joueurs et les cartes communes compressées en 1000 buckets, plus la
taille du pot en fraction du stack de départ) ; sortie = valeurs espérées des 1000 buckets
en fraction du pot. Résultat marquant : une réimplémentation de DeepStack **perd** contre
Slumbot, alors que Supremus le bat très largement et obtient une exploitabilité plus faible
face à un local best response. À architecture identique. C'est l'argument le plus fort en
faveur de soigner les détails listés dans ce document plutôt que de chercher une
architecture exotique.

**Deep CFR** : les cartes sont représentées comme la somme de trois embeddings — rang (1-13),
couleur (1-4), carte (1-52) — sommés pour chaque ensemble de cartes invariant par
permutation (main, flop, turn, river), puis concaténés. C'est **exactement** l'encodage de
§8.4, confirmé indépendamment.

**OpenHoldem** n'est pas une architecture mais un banc d'essai : protocole d'évaluation
standardisé, baselines publiques, plateforme de test en ligne (~20 M de mains accumulées).
Son ablation de représentations d'état est la source du verdict de §8.3.

#### Décision : pas de bucketing

DeepStack et Supremus compressent à 1000 buckets. **On ne le fait pas.** Trois raisons :

1. Le bucketing impose un **plancher d'erreur incompressible** — l'« erreur d'encodage »,
   que Zarick & Tejwani réduisent d'environ 30 % en changeant de cible. Notre sortie pleine
   n'a pas ce plancher.
2. Le bucketing exige de construire et maintenir une abstraction « potential aware » par
   k-means, avec tout le travail d'ingénierie associé.
3. GTO Wizard revendique de solver sans abstractions, et c'est la cible.

**Le contre-argument, honnêtement.** Supremus discute l'alternative naïve : passer la main
détenue en entrée et ne sortir que sa valeur. Ils la qualifient de gaspillage, à juste
titre — ce serait 1326 forwards.

**Notre tête partagée évite les deux écueils** : un seul passage du trunk, puis un MLP
appliqué en parallèle aux 1326 mains via un batched matmul (§8.5). Pas de perte
d'information par bucketing, pas de coût de 1326 forwards. C'est la raison d'être de cette
structure, et c'est ce qui rend l'absence de bucketing viable.

## 9. Perte

```python
L = L_value + 0.1 · L_policy

L_value  = Σ_i Σ_h (ε + w_h^i) · Huber_δ(v̂_i[h], v_i[h])    δ = 0.1, ε = 0.01
L_policy = Σ_h w_h · KL(σ_solver(·|h) ‖ σ̂(·|h))
```

**Huber et pas MSE.** La distribution des CFV est à queues lourdes : nuts et air absolu
produisent des valeurs extrêmes et rares. En MSE une poignée de combos monopolise le
gradient et le réseau sacrifie le corps de la range — ce qui compte le plus. `δ = 0.1` en
fraction de pot.

**Pondération par `w_h`** (masse de range). Dicté par l'usage en aval : dans le CFR flop
qui consommera ces valeurs, la mise à jour de regret d'un combo est multipliée par sa reach
probability. Une erreur sur un combo de reach nulle ne se propage nulle part. On optimise
directement la quantité qui coûtera de l'EV.

**Le plancher `ε`.** Sans lui le réseau est libre de produire n'importe quoi hors range. À
l'inférence, une range inhabituelle active ces combos → valeurs aberrantes. Coût quasi nul,
robustesse achetée.

**Tête policy auxiliaire.** Trois bénéfices pour un coût marginal : elle force les
représentations à encoder l'information stratégique et pas seulement une moyenne ; elle
fournit le warm-start du CFR au runtime (÷2 à ÷5 sur les itérations) ; et les labels sont
déjà là, puisque les fréquences sont la sortie native du solver.

---

## 10. Optimisation

```
AdamW, lr 3e-4, weight_decay 0.01
warmup 2000 pas, puis cosine
batch 128 turn-roots  (= 340k exemples-mains)
bf16 sur le trunk, fp32 sur la tête et la couche zero-sum
EMA des poids, decay 0.999  →  c'est l'EMA qu'on déploie
```

`3e-4` et non `1e-3` : la régression de valeur est bien moins tolérante que la
classification. L'EMA est un gain gratuit ici — les cibles étant déterministes, le bruit
résiduel vient entièrement de l'échantillonnage des batches et le moyennage l'élimine.

**Pas d'échantillonnage de mains.** Le raccourci qui divise le coût par 5 sur un policy net
(tirer 128–256 mains par nœud) est **interdit** ici : la couche zero-sum fait un produit
scalaire sur la range complète et exige les 1326 sorties.

**Augmentation par permutation de couleurs** (jusqu'à ×24, à appliquer dans le dataloader).
Piège : il faut permuter le board **et** réindexer les deux vecteurs de 1326 de façon
cohérente. Précalculer les 24 tables de permutation d'indices une fois pour toutes. Une
incohérence ici produit un réseau qui apprend du bruit sans jamais lever d'erreur.

**Curriculum par SPR** : 20 % premiers pas sur SPR bas. Solves plus rapides, fonction plus
simple, représentations de cartes construites sur le cas facile avant le jeu profond.

### 10.1 Coût et matériel

Forward sur un nœud ≈ 1,5 GFLOP, ≈ 4,5 GFLOPs avec backward. Batch 128 → ~580 GFLOPs/pas.

| Cible | Débit effectif | Temps par pas |
|---|---|---|
| CPU Apple (Accelerate/AMX) | 100–200 GFLOPS | 3–6 s |
| GPU intégrée via MPS | 2–5 TFLOPS | ~0,2 s |

**Entraîner sur MPS.** Lancer avec `PYTORCH_ENABLE_MPS_FALLBACK=1` et traiter tout warning
de fallback comme un bug : un seul fallback CPU dans la boucle chaude coûte un facteur 10.
`torch.compile` est partiel sur MPS — mesurer, ne pas supposer.

**Le vrai goulot reste la génération des données**, pas l'entraînement. C'est elle qui
dimensionne le projet.

---

## 11. Évaluation

Split par `solve_id` **et** par hash de board. Les nœuds d'un même solve sont fortement
corrélés : répartis des deux côtés, la métrique est optimiste d'un facteur insoupçonné.

```
primaire   : MAE pondérée par range, en % du pot        cible < 0,3 %
secondaire : perte d'EV du resolving flop, en bb/100
diagnostic : erreur ventilée par texture de board et par bucket de SPR
```

> **Première mesure de la perte d'EV réelle (23/09/2026, un seul spot).** Méthode : verrouiller la
> stratégie de flop d'un joueur dans `postflop-solver` (`bench/real_ev_loss.py`, binaire `flop_lock`),
> résoudre exactement tout le reste, et comparer la valeur à l'équilibre ; le jeu de référence est celui
> des labels (turn/river 66 % + all-in + relance 2,5x). BTN vs BB, flop 9♠7♠2♠, pot 5,5 bb, bruit du
> solver ~±0,15 % du pot : témoin (équilibre verrouillé) 0,00 % ; résolveur avec `net_turn` TINY 0,92 %,
> SMALL 0,64 %, DOC 0,74 % (soit 3,5 à 5,1 bb par 100 flops de ce spot) ; feuilles à l'équité seule 3,11 % ;
> stratégie uniforme 20,9 %. Les réseaux appris divisent la perte par ~4-5 par rapport à l'équité seule ;
> l'écart entre tailles de réseau n'est pas significatif à cette précision. Ordre de grandeur : 2 à 4× la
> valeur annoncée par GTO Wizard (0,1-0,3 %), sur un spot, un run, un arbre de flop sans relance.

**Ne jamais utiliser l'accuracy de match d'action.** Un réseau peut atteindre 85 % et être
catastrophique s'il se trompe sur les nœuds à forte masse de range.

**Ne jamais utiliser le résidu zero-sum comme métrique.** La couche l'impose par
construction : un réseau totalement faux de façon corrélée le satisfait parfaitement.
C'est une contrainte, pas un signal.

**Test de sensibilité aux ranges — à automatiser.** Prendre un turn-root, remplacer la range
adverse par une range très différente (polarisée au lieu de linéaire), vérifier que la
sortie bouge substantiellement. C'est le mode d'échec le plus courant et le plus silencieux :
un réseau qui a appris une moyenne conditionnée au board et ignore largement les ranges. Il
aura une perte correcte à l'entraînement et sera inutile au runtime, où la range est
précisément la variable qui bouge entre deux solves.

---

## 12. Runtime — resolving flop

```
état courant (flop, pot, r1, r2)
  └─ arbre de mises du flop UNIQUEMENT
      └─ CFR discounted (α=1.5, β=0, γ=2), 100–300 itérations
          ├─ feuilles terminales (fold/showdown) → exact
          └─ feuilles non terminales → turn-root × 49 cartes → net_turn, moyenné
      └─ stratégie moyenne à la racine → jouer
          └─ propager les ranges → décision suivante
```

**Points de performance**

1. **CFR entièrement vectorisé sur les mains.** Tenseurs `(n_infosets, 1326, n_actions)`.
   Toute boucle Python sur les combos rend l'approche inutilisable.

2. **Batcher les appels réseau.** `n_feuilles × 49` appels par itération. Un seul forward :

```python
leaves      = build_leaf_tensor(tree, strategies)   # (n_feuilles * 49, D)
values      = net_turn(leaves)
leaf_values = values.view(n_feuilles, 49, 2, 1326).mean(dim=1)
```

3. **Cache partiel.** Les valeurs de feuilles dépendent des ranges, qui changent à chaque
   itération — pas de cache naïf. Les rafraîchir toutes les `k` itérations (k = 5–10) :
   convergence peu affectée, coût divisé par k.

   > **Mesuré (23/09/2026, contredit l'hypothèse ci-dessus).** Test sans réseau, feuilles
   > valorisées exactement par le solver (`bench/validate_resolver.py`, un seul état, résolution
   > d'un *turn* à feuilles-river, 100 itérations) : rafraîchir à **chaque** itération donne une
   > exploitabilité de 0,19 % et des EV à 0,7 % du pot du solve exact ; avec `k = 5`, 2,3 % et
   > 5–7 % ; les calendriers géométriques ou « denses au début » ne font pas mieux (2 à 10 %).
   > Il faut donc *un appel réseau batché par itération*, ce qui change le budget de §12
   > (voir ci-dessous). À reconfirmer au niveau flop avec `net_turn`.
   >
   > **Plancher ε** : le CFR a besoin de la valeur de chaque main à chaque feuille, même à reach
   > nulle. Le réseau est entraîné sur les seules mains de la range, donc on lui donne
   > `(1-m)·range + m·range_initiale` avec `m = 0,003` (`m = 1` si la feuille est inatteignable).

4. **Warm-start par la tête policy** : ÷2 à ÷5 sur le nombre d'itérations.

5. **Re-solving continu** : propager sa range via sa stratégie, celle de l'adversaire via
   son action observée. Le *re-solving gadget* (mémoriser les CFV adverses du solve
   précédent comme contraintes) donne la garantie de non-exploitabilité ; l'*unsafe
   resolving* suffit pour un outil de coaching et coûte deux fois moins en complexité.
   Décision : commencer unsafe, ajouter le gadget seulement si l'exploitabilité mesurée
   le justifie.

Budget cible : **< 400 ms par décision flop**.

> **Mesuré (voir `bench/bench_resolve.py`, machine chargée donc pessimiste)** : la boucle CFR
> elle-même est négligeable (~1 ms par itération avec feuilles en cache) ; ce qui domine, ce sont
> (1) les **tables de board** pour les 49 cartes de turn (~0,9 s par board, ~45 s par nouveau
> flop : évaluateur 7 cartes 0,22 s + table de victoire 0,48 s), à faire une fois par flop et
> masquable derrière le temps de réflexion de l'utilisateur ; (2) l'**appel réseau par itération**
> (~0,85 s par rafraîchissement de 245 feuilles avec SMALL, à remesurer machine libre). Avec un
> appel par itération, 100 itérations sont hors budget d'un facteur ~100 : le coût par état du
> réseau (largeur de la tête par main) fixe ce qui est atteignable.
>
> **Mesuré ensuite, machine libre** (`net_turn` TINY 0,12 M, flop SRP, 100 itérations, un appel
> par itération, paquets de 24) : tables de board 17,6 s (une fois par flop) ; puis 18,9 s pour
> les 49 cartes de turn (features 4,1 + réseau 14,1 + post 0,6 + CFR 0,09), et **7,4 s avec
> échantillonnage de 12 cartes par itération** (features 1,8 + réseau 4,7 + post 0,7 + CFR 0,09).
> Latence d'inférence de 245 états (une itération, 49 cartes), meilleur découpage, MPS : TINY 43 ms,
> S2 ~140 ms, SMALL ~270 ms, DOC ~1,2 s. Le coût ne suit **pas** la largeur de la tête : c'est la taille
> des tenseurs `(paquet × 1326 × largeur)` qui compte (chute de débit dès ~100 états par paquet,
> falaise à 245 avec SMALL : 21 s). L'échantillonnage de cartes de turn (estimateur non biaisé,
> repondéré par 49/n) testé avec feuilles exactes : 12 cartes/itération = 4× moins d'évaluations pour
> 0,37 % d'exploitabilité contre 0,19 % ; 6 cartes 0,80 %. Le budget de 400 ms n'est pas atteint
> (~18× trop lent avec TINY + 12 cartes) : il faudra réduire aussi le nombre d'itérations, limiter le
> calcul aux mains de la range et canonicaliser une fois par paquet plutôt qu'à chaque sous-paquet.
>
> **Fait (23/09/2026)**, mesuré machine libre (TINY, 12 cartes, 100 itérations) : (1) canonicalisation par
> renommage des cartes du board et des mains, sans permuter aucun vecteur (le réseau est une fonction d'ensemble
> des mains) ; (2) seules les mains de la range initiale sont évaluées (617 sur 1326 en SRP BTN-BB) ; (3) tête policy
> sautée ; (4) `tools/boardlib` (Rust) : évaluateur 7 cartes 115× plus rapide (1,2 ms pour 63 600 mains) et tables d'un
> flop en 1,1 s au lieu de ~27 s. Résultat : 7,4 s → 4,5 s par décision (réseau 4,7 → 2,0 s), tables par flop
> 17–21 s → 0,8 s, **sorties strictement identiques** (tests d'équivalence à 1e-5). Reste : features 1,7 s,
> réseau 2,0 s, post 0,7 s pour 100 itérations, soit ~11× le budget de 400 ms.
>
> **Fait ensuite (23/09/2026)**, mesuré machine libre, même session avant/après (TINY, 12 cartes, 100 itérations, un
> rafraîchissement par itération ; 5 feuilles, donc 60 états par appel) : les équités des feuilles
> (`NetLeaves._equity_all`) passent de 48 petits produits matriciels par itération à 12 (un par carte, sur la table
> de victoire en place) plus un seul pour tous les dénominateurs. Features 1,48 → 0,72 s ; une décision sur un flop
> déjà préparé 3,8 → 2,95 s, sur un flop neuf (tables comprises) 4,9 → 3,7 s. Sorties identiques à 1,3e-5 près
> (bruit float32). Piège mesuré : regrouper les 12 tables en un seul `bmm` est PLUS lent (`wd[ks]` copie 84 Mo,
> 18 ms sur MPS), l'indexation `torch.index_select` ne coûte que 1,8 ms. Reste, pour 100 itérations : réseau 1,6 s,
> features 0,7 s, post 0,6 s, CFR 0,1 s.
>
> **Mesuré sur les flops jamais vus à l'entraînement** (`bench/eval_heldout.py time` : les 28 flops val et 22 test des
> matchups SRP natifs, TINY, arbre de flop 50 % sans relance donc 3 feuilles, 12 cartes, 100 itérations, un
> rafraîchissement par itération, machine libre) : décision sur flop neuf **3,36 s** en moyenne (médiane 3,27, p90 3,58,
> max 5,15), répétée sur le même flop 2,86 s ; exploitabilité *dans le jeu du réseau* 0,38 % du pot (val 0,38, test 0,38 ;
> p90 0,54 ; max 0,70). Cette exploitabilité mesure la convergence du CFR sur le modèle, pas l'erreur du réseau.
>
> **Élagage des mains de faible poids** (poids < tau x le plus grand poids de la range mis à 0) : les ranges sont presque
> toutes à poids 1 (préflop pur), donc tau = 0,5 ne retire que 14 % des mains (434 -> 374) et tau = 0,75 24 % (-> 330). Temps
> 3,36 -> 3,25 -> 3,19 s (-3 %, -5 %), exploitabilité du jeu du réseau 0,38 -> 0,43 -> 0,46 %, et la fréquence de mise OOP à
> la racine bouge de 0,10 / 0,21 en moyenne (pondérée par la range). Le coût ne dépend presque pas du nombre de mains : ce
> n'est pas un levier.
>
> **Coût par appel du réseau** (TINY, 36 états, 611 mains) : 768 opérations ATen, ~4,5 ms d'émission côté CPU pour ~4,4 ms de
> calcul (bloqué par l'émission, pas par le GPU) ; le calcul pur est de l'ordre de la milliseconde. `torch.compile` (inductor) fonctionne
> sur MPS (compilation 7 s par forme) mais ne gagne que 6 %. Sur CPU (4 threads) la même décision prend 3,0 s contre 2,7 s
> sur MPS : la charge est dominée par la surcharge par opération. Dans la boucle, un appel coûte ~11 ms au lieu de 4,5 ms mesurés
> en rafale : avec des trous d'au moins 5 ms de travail CPU entre deux appels, le GPU redevient ~2x plus lent. Un thread de
> maintien d'activité GPU bloque MPS (à ne pas refaire).
>
> **Fait (23/09/2026)** : forward du réseau à opérations partagées (masses de cartes calculées une fois pour les deux
> côtés, projections clé/valeur de l'attention une fois pour les deux ranges, `arange` constant hors de l'appel) : 768 -> 605
> opérations, 4,5 -> 3,1 ms par appel, sorties identiques à 7e-8 ; et paquets équilibrés d'au plus 36 états au lieu de 24
> (un saut de 3x apparaît dès 48 états par appel). Sur les mêmes 50 flops : **2,73 s** sur flop neuf (médiane 2,68, p90 3,10,
> max 3,98), **2,40 s** en répétition, exploitabilité du jeu du réseau inchangée (0,38 %).
>
> **Core ML** (coremltools 9.0, torch 2.14 non testé par l'outil) : la conversion du forward TINY (36 états, 611 mains fixes,
> 285 opérations) réussit du premier coup en fp32, écart 1,8e-7 avec PyTorch. Un appel coûte 4,8 ms sur GPU (entrées et sorties
> numpy comprises) et 14 ms sur CPU, contre ~3,1 ms en rafale et ~9 ms dans la boucle pour PyTorch-MPS : pas de gain décisif,
> et il faudrait figer B et K (remplissage des mains) et convertir les tenseurs à chaque appel. Écarté. Le plancher d'un appel
> réseau est de l'ordre de 3 à 5 ms quel que soit le moteur ; sur 100 itérations, réseau ~0,9 s, features 0,9 s, post 0,5 s : le
> prochain poste à réduire est donc features + post (beaucoup de petits tenseurs), puis le nombre d'itérations.
>
> **Nombre d'itérations** (12 flops val/test, réf. = 400 itérations sur les 49 cartes ; temps proportionnel aux itérations) :
> 36 it 1,13 s, exploitabilité du jeu du réseau 0,81 % ; 60 it 1,4-1,7 s, 0,55 % ; 72 it 2,0 s, 0,50 % ; 100 it 2,4-2,8 s, 0,40 %
> (référence 0,14 %). La distance de la stratégie à la racine à la référence (fréquence de mise OOP, pondérée par la range) reste
> grande : 0,26 / 0,21 / 0,195 / 0,173 : à 100 itérations la stratégie est loin d'être convergée même si elle est peu exploitable.
> Rien ne relie le nombre d'itérations à la taille des paquets (un appel par itération, quel que soit le multiple).
>
> **Post-traitement** (fait) : la moyenne sur les cartes ne porte plus que sur les cartes échantillonnées (12 et non 49),
> R0/R1 ne sont transférés qu'une fois, les deux côtés du plancher epsilon sont mélangés en un appel : post 0,68 -> 0,33 s,
> décision 3,0 -> 2,45 s (A/B alterné, mêmes sessions), stratégie identique à 2e-4 près.
>
> **Démarrage à chaud** (`resolve(..., warm=W)`, désactivé par défaut) : les W premières itérations valorisent les feuilles par la
> ligne de base d'équité (sans réseau, ~4x moins cher), puis le CFR continue avec le réseau. Sans remise à zéro de la stratégie
> moyenne c'est pire que le démarrage à froid (100 it, 30 chaudes : 0,84 % contre 0,40 %). Avec remise à zéro de la moyenne
> (les regrets restent) : 20 chaudes sur 60 itérations = 1,08 s, 0,60 % contre ~0,85 % pour un démarrage à froid de même durée,
> mais aucun gain à 100 itérations. Les 8 flops de ce test sont peu nombreux ; gain modeste.

---

## 13. Pipeline de données

### 13.1 Réduction de scope

- Commencer par une famille de spots (ex. BTN vs BB, SRP, 100 bb).
- 2 à 3 tailles de mise par nœud.
- Paralléliser N instances du solver, lancer en batch la nuit.
- **Solver retenu : `postflop-solver`** (Rust, Discounted CFR, `tools/postflop-solver`) — ranges
  par combo, EV par main, démarrage au turn, arrêt sur exploitabilité cible, ~5× plus rapide
  que TexasSolver sur un flop. TexasSolver reste utilisé par le trainer uniquement.
  Wrapper JSONL : `tools/turn-labels` (`turn-labels` pour les labels, `flop_lines` pour les
  ranges qui arrivent au turn).

### 13.2 Schéma de stockage

Parquet + zstd, colonnaire, un enregistrement par turn-root :

```
board_flop    uint8[3]
board_turn    uint8
spr           float32
r1, r2        float32[1326]     corrigées du card removal
cfv1, cfv2    float32[1326]     fraction du pot, centrées, /(1+SPR)
strategy      float32[1326, n_actions]   pour la tête policy
legal_mask    uint8[n_actions]
solve_id      uint64
source        uint8             simulateur / bibliothèque / heuristique
```

Enregistrer en plus les river-roots rencontrés (§6.3) : coût marginal nul.

Le split train/test se fait par `solve_id` et hash de board, jamais par nœud.

### 13.3 Un solve = des milliers d'exemples

On ne récolte pas seulement la racine : tout l'arbre résolu est labellisé — mais attention,
seuls les **street-roots** sont dans le domaine du réseau. Les nœuds intermédiaires du round
de mises servent à la tête policy, pas à la tête value.

---

## 14. Couche d'explicabilité

Un LLM fine-tuné n'explique pas le raisonnement du réseau — il produit une rationalisation
post-hoc plausible. C'est un **verbaliseur**, pas un interpréteur.

La vraie explicabilité est déterministe et calculable :

- décomposition de la perte d'EV : coût exact de l'action jouée vs l'optimale, en bb et % du pot
- intention globale de la range : % value / bluff / air, position de la main dans l'ordre
- effet de blocker quantifié
- contrefactuels : même main sur une autre carte, ou à un autre SPR

```
solver / net  →  faits calculés (JSON)  →  LLM  →  explication en français
```

**Règle dure** : le modèle ne peut mentionner que des nombres présents dans le JSON. Un
vérificateur automatique extrait tous les nombres de la sortie et rejette toute réponse
contenant une valeur absente de l'entrée.

Données de fine-tuning : distillation d'explications générées par un grand modèle à partir
du JSON, filtrées par le vérificateur. Quelques milliers d'exemples. Éviter les corpus de
forums et transcriptions de coaching — non alignés et riches en raisonnement faux.

Faisabilité : 3–4B en QLoRA 4-bit sur Colab, 2–4 h sur T4. Déploiement local en MLX ou
llama.cpp quantifié, 30–60 tok/s.

**Ordre** : ne pas fine-tuner d'emblée. Prompter un modèle existant avec le JSON et mesurer.
Le fine-tuning est une optimisation de coût, latence et style. **Le JSON de faits est la
fondation** — le construire en premier.

---

## 15. Ordre de travail

| Étape | Contenu | Critère de sortie |
|---|---|---|
| 0 | Vérifier que l'export solver donne les fréquences de tout l'arbre | bloquant si non |
| 1 | Extraction des CFV depuis les fréquences (§5) + showdown O(n log n) | `\|r1·v1 + r2·v2\| < 1e-6` |
| 2 | Générateur de turn-roots réalistes via le simulateur (§7) | distribution documentée |
| 3 | Solves turn+river en batch + schéma Parquet | premiers 5k solves |
| 4 | Harnais d'évaluation (MAE pondérée, sensibilité aux ranges) | métriques reproductibles |
| 4b | Baseline équité réalisée SEULE, sans paramètre appris (§8.6) | plancher de MAE établi |
| 5 | `net_turn` : entraînement | MAE < 0,3 % du pot sur holdout |
| 6 | CFR flop vectorisé + batching des feuilles | < 400 ms par décision |
| 7 | Mesure de la perte d'EV du resolving complet | en bb/100 |
| 8 | JSON de faits + couche d'explication | explications vérifiées |
| 9 | Bascule bootstrap si limité par la quantité (§6.3) | — |

**Avancement.** 0 : sans objet (solver renvoie les EV, §5). 1 : remplacée par le wrapper
`tools/turn-labels` ; le test `|r1·v1 + r2·v2|` est dans `model/tests`. 2 : ranges turn tirées
des solves de flop (source 2 du §7), 70 % on-policy / 30 % perturbées. 3 : en cours
(`model/scripts/gen_flops.py`, `gen_labels.py`). 4, 4b, 5 : `model/gtonet` (évaluation,
baseline équité, `net_turn`).

---

## 16. Références

- Moravčík et al. (2017), *DeepStack* — counterfactual value networks, couche zero-sum,
  continual re-solving, MLP 7×500.
- Brown et al. (2019), *Deep CFR* — approximation neuronale des regrets.
- Steinberger (2019), *Single Deep CFR* — variante simplifiée.
- Brown et al. (2020), *ReBeL* — value + policy net sur public belief states, warm-start
  de la recherche ; démonstration que les PBS aléatoires échouent.
- `gtowizard-ai/mitpoker-2024` — code public de l'équipe GTO Wizard (MIT Auction Hold'em).
  Value network remplacé par une heuristique d'équité, mais la structure du subgame
  depth-limited est là. Le document le plus concret venant d'eux.
- Yakovenko et al. (2016), *Poker-CNN* — tenseur 4×13 par carte ; paires = même colonne,
  couleurs = même ligne. Origine de l'idée de représentation spatiale des cartes.
- Zhao et al. (2022), *AlphaHoldem* — architecture pseudo-siamoise, cartes et actions
  traitées séparément ; 4 ms par décision.
- *OpenHoldem* (2020) — ablation utile : le tenseur structuré bat le vecteur plat, mais
  l'architecture factorisée bat le CNN pur. Voir §8.3.
- Zarick & Tejwani et al. (AAAI 2023), *Don't Predict Counterfactual Values, Predict
  Expected Values Instead* — justifie la cible EV conditionnelle de §4.2.
- Zarick et al. (2020), *Unlocking the Potential of Deep Counterfactual Value Networks*
  (Supremus) — mêmes poids que DeepStack, résultats très supérieurs par les détails.
- `postflop-solver` (b-inary) — solver Discounted CFR open source en Rust, **générateur retenu**
  (développement suspendu depuis oct. 2023 ; licence AGPL-3.0). `wasm-postflop` en est le
  front web.
- Blog GTO Wizard : *GTO Wizard AI Explained*, *AI Benchmarks*, *3-way Benchmarks* —
  ordres de grandeur de Nash distance et de coût de solve.
