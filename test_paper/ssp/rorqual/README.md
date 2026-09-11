# Jeu de test SSP — test_paper/ssp/rorqual

Instances générées pour évaluer sur Trillium : temps de résolution GNN vs
référence (B&B), nombre de nœuds, gap relatif, comparaison forward pass GNN
vs sous-gradient pur, et SLD vs DLD.

Générées sur Rorqual (login/job SLURM), pas sur Trillium — poussées ensuite
via git pour être récupérées côté Trillium.

## Grille

E ∈ {4, 6, 8} × T ∈ {8, 10, 12} × N ∈ {50, 100, 500} = 27 combos × 10
répétitions = 270 instances.

Structure : `E{E}_T{T}_N{N}/rep{0..9}/{instance,automaton,cost,demand}.txt`
(format 4-fichiers standard, cf. `solver_generic/ssp/ssp_loader.hpp`).

## Calibration (validée le 2026-08-22)

- **epsilon = 0.1** pour toutes les instances.
- **Fenêtre de difficulté cible** (nombre de nœuds B&B sous le probe Gecode
  base, calibrée par bissection) :
  - **E = 4, T = 10 ou 12** : [1e6, 1e8] — la fenêtre par défaut [1e4,1e6] est
    documentée (docstring de `gnn_ssp/gen_ssp_instance_custom.py`) comme
    donnant des instances triviales (<0.2s) pour E=4 à grand T/N.
  - **E = 4, T = 8** : [1e4, 1e6] — **corrigé après le smoke test** (job
    19418037, 2026-08-22) : la fenêtre relevée [1e6,1e8] est INATTEIGNABLE
    pour ce combo précis (10/10 échecs — bissection oscille entre "too_easy"
    ≤~24k nœuds et INFEASIBLE, sans palier intermédiaire), conformément à
    l'étude de calibration de `run_sweep_ssp_bound_efficacy_rorqual.sh`
    (E∈{2,3,4} avec T≤8 bascule directement trivial→INFEASIBLE). La fenêtre
    par défaut, elle, est documentée fonctionnelle pour ce combo précis.
  - **E = 6, 8** (tout T) : [1e4, 1e6] — fenêtre par défaut du repo.
- **master_seed = 42**, répétitions indexées par `ep_id` = 0..9.
- Générateur : `gnn_ssp/gen_ssp_instance_custom.py` (bissection sur la
  difficulté via `ssp_stochastic.generator_adaptive.generate_episode_fixed_E`).
- Probe : `solver_bab/build_collect_cpu/ssp_probe`.

## Provenance des répétitions

- **E = 6, 8** (18 combos) : `rep0` est **réutilisé** depuis l'instance
  `gnn_ssp/test_data/ssp_native_test_E{E}T{T}N{N}_sweep/` déjà présente dans
  le repo — **calibration d'origine non documentée** (paramètres exacts de
  génération inconnus, probablement fenêtre par défaut [1e4,1e6] et
  epsilon=0.1 mais non vérifié). `rep1`..`rep9` sont générés from scratch
  avec les paramètres ci-dessus.
- **E = 4** (9 combos, y compris T=8 qui n'avait aucun `_sweep` existant) :
  toutes les répétitions `rep0`..`rep9` sont générées from scratch avec la
  fenêtre relevée [1e6,1e8], pour homogénéité (éviter de mélanger une
  instance triviale avec 9 instances dures).

Chaque dossier `E{E}_T{T}_N{N}/PROVENANCE.txt` enregistre, par combo, les
paramètres exacts utilisés et le statut de chaque répétition
(`generated` / `reused_from_sweep` / `FAILED`).

## Risques connus

- Pour certains petits combos (notamment **E=4, T=8**), une étude de
  calibration antérieure du repo (`run_sweep_ssp_bound_efficacy_rorqual.sh`)
  a montré que E∈{2,3,4} avec T≤8 peut basculer directement de trivial à
  INFEASIBLE sans palier de difficulté intermédiaire — sous la fenêtre
  [1e4,1e6]. Sous la fenêtre relevée [1e6,1e8] utilisée ici pour tout E=4,
  ce risque est probablement plus élevé encore : la bissection peut échouer
  à atteindre la fenêtre pour certaines répétitions. Le script de génération
  (`run_gen_ssp_test_paper_trillium_rorqual.sh`) ne s'arrête pas sur un
  échec : il logue `FAILED` dans `PROVENANCE.txt`, supprime le dossier
  partiel, et continue. **Vérifier après génération que chaque combo a bien
  10 répétitions** — sinon ajuster la fenêtre (ou accepter <10 reps) pour
  les combos concernés.
