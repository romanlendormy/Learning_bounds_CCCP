#pragma once
/*
 * gecode_bab_ccpo_gnn_batched.hpp — Driver B&B CCPO avec borne lagrangienne
 * à λ PRÉDIT par un GNN, évalué PAR BATCH de nœuds. Miroir de solver_bab/mkp/gecode_bab_gnn_batched.*
 * (cf. rapport d'exploration de l'architecture MKP) :
 *
 *   - DFS manuel (pile explicite `open`), comme solve_gecode_bab_ccpo ;
 *   - un nœud enfant, une fois cloné + committé + propagé en CP pur
 *     (status()), n'est PAS borné tout de suite : son sous-problème résiduel
 *     est mis en attente dans pending[n_free] ;
 *   - quand pending[n_free] atteint K nœuds (ou quand `open` se vide), le
 *     bucket est "flushé" : UN seul appel backend->query_batch() prédit les
 *     λ de tous les membres (tous de même n_free, donc même forme de graphe),
 *     puis la borne L(λ) est évaluée par membre (evaluate_ccpo_lagrangian_omp,
 *     boucle scénario parallélisée OpenMP) et postée (obj <= floor(fp+L)).
 *
 * Validité : L(λ) est une borne supérieure valide pour N'IMPORTE QUEL λ
 * (dualité faible), donc quelle que soit la qualité du backend l'optimum
 * trouvé doit être IDENTIQUE à celui de REF -- seul le nombre de nœuds/temps
 * varie. C'est le critère de validation principal (cf. test avec le backend
 * factice ZeroBackend, λ=0).
 *
 * Bucketing par n_free SEUL (pas (n_free, n_eligible)) : n_eligible est
 * constant pour tout un run (une instance, présolve fait une fois), cf.
 * gnn_ccpo/ccpo_gnn_scriptable_batched.py.
 */

#include "ccpo_gnn_backend_batch.hpp"
#include "gecode_bab_ccpo.hpp"

#include <limits>
#include <memory>
#include <vector>

struct CCPOGNNBatchedResult : public CCPOResult {
    int64_t n_bucket_flushes = 0;
    double  mean_bucket_fill = 0.0;
    int64_t max_pending_live = 0;
    int64_t nodes_bypassed_leaf = 0;   // sans variable libre sur q (branchement z) -- pas de borne
    int64_t infeas_inherited = 0;      // scénarios fixés à z_j=0 suite à une preuve d'infaisabilité de zeta_solve (héritage)
    double  t_backend_s = 0.0;         // temps cumulé dans backend->query_batch
    double  t_eval_s = 0.0;            // temps cumulé dans evaluate_ccpo_lagrangian_omp
    // Borne du PREMIER NIVEAU (comme CCMKP) : max sur les enfants directs de la racine de (cost_fixed + L(lambda)) = majorant valide de l'optimum global ;
    // NaN si aucun enfant évalué (ex. racine résolue par propagation). Le gap racine (%) se calcule HORS LIGNE par rapport à l'optimum prouvé.
    double  first_level_bound = std::numeric_limits<double>::quiet_NaN();
    int64_t first_level_children = 0;  // nombre d'enfants de la racine effectivement évalués (en général 2)
};

CCPOGNNBatchedResult solve_gecode_bab_ccpo_gnn_batched(
    int n, int N, int k, int Q, int L, int U, long long B, int min_feas,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    double time_lim,
    int K,                          // taille de bucket déclenchant un flush
    CCPOGNNBackendBatch& backend,
    int omp_threads);               // threads OpenMP pour la boucle scénario (réaffirmé avant CHAQUE flush)
