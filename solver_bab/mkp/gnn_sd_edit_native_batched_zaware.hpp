#pragma once
/*
 * gnn_sd_edit_native_batched_zaware.hpp — Variante Z-AWARE de
 * GNNSdEditNativeBatched (gnn_sd_edit_native_batched.hpp) : exploite les
 * z_i déjà résolus par la propagation CP PENDANT le x-branching, au lieu de
 * les ignorer (N_z = N, n_fixed_z1 = 0 toujours codé en dur dans la variante
 * d'origine) — cf. discussion "extract_pending ne renvoie que x_mask/fp,
 * jamais l'état de z" et mesure de fréquence (33.4% des nœuds de
 * x-branching ont ≥1 z déjà fixé, seed=43 n=20/N=500, cf. mémoire de session).
 *
 * Fichier séparé de gnn_sd_edit_native_batched.hpp/.cpp — AUCUNE modification
 * du backend V1LD de production (même convention que le reste du repo).
 * Retour en arrière trivial : ne pas utiliser ce backend/binaire, l'original
 * reste intact et continue de tourner exactement comme avant.
 *
 * Différence mathématique avec la version d'origine (portée fidèlement
 * depuis train_v1LD_sd_edit.py::build_graph, PAS une nouvelle dérivation) :
 *   - free_z = indices des scénarios encore réellement libres (z non assigné)
 *   - N_z = |free_z| (au lieu de N toujours)
 *   - n_fixed_z1 = nombre de z déjà fixés à 1 (toujours 0 en pratique tant que
 *     le x-branching n'est pas terminé — RM_IMP ne peut déduire z=1 que par
 *     branchement explicite, jamais par propagation seule, cf. discussion)
 *   - A_g/b_res_g/c_g/A_mean_scen/A_std_scen/dantzig_rank recalculés
 *     UNIQUEMENT sur les scénarios de free_z (pas sur les N complets) —
 *     validité : un scénario à z fixé (0 ou 1) ne contribue RIEN à la
 *     formule lagrangienne (son objectif ne dépend que de x ; sa contrainte
 *     de capacité est déjà gérée par la propagation CP native de Gecode,
 *     dure si z=1, désactivée si z=0) — l'EXCLURE de la somme est la
 *     transcription exacte de lagrangian_v1ld/build_graph, pas une
 *     approximation.
 *
 * RÉVISION (2026-08-19) : bucketing par (n_g, N_z) abandonné — fragmentait
 * trop les buckets (remplissage moyen 56→23 sur K=64, mesuré sur seed=43
 * n=20/N=500), le coût dépassait le gain. Remplacé par un batch RAGGED : les
 * B membres d'un appel query_batch() peuvent avoir des N_z DIFFÉRENTS (pas de
 * padding, pas de troncature) — le pooling utilise
 * GNNSdEditScriptableBatchedRagged (gnn_sd_edit_scriptable_batched_ragged.py,
 * scatter/index_add_ paramétré par des index par nœud, validé max|diff|~3e-8
 * contre le modèle dense sur cas uniforme ET hétérogène) au lieu du reshape
 * dense `.view(B, n_z, ...)` de GNNSdEditNativeBatched. MÊME checkpoint, MÊME
 * poids, aucun réentraînement — seul un second artefact exporté
 * (.ragged_scripted_batched.pt) diffère. Le driver bucket à nouveau par
 * n_g SEUL (gecode_bab_gnn_batched_zaware.hpp/.cpp) — plus de fragmentation.
 */

#include "gnn_bound_backend_batch.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSdEditNativeBatchedZAware {
public:
    GNNSdEditNativeBatchedZAware(const std::string& checkpoint_path, const std::string& device,
                                  int n, int m, int N, double epsilon,
                                  const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                                  bool use_surrogate_zeta = false);

    ~GNNSdEditNativeBatchedZAware();

    GNNSdEditNativeBatchedZAware(const GNNSdEditNativeBatchedZAware&)            = delete;
    GNNSdEditNativeBatchedZAware& operator=(const GNNSdEditNativeBatchedZAware&) = delete;

    // z_masks[b] : (N,) int8, mêmes conventions que x_mask (0=fixé_0, 1=fixé_1,
    // 2=libre) — TOUS les membres du batch doivent partager le même N_z
    // (= nombre de z_masks[b][i] != 0/1, i.e. encore libres OU déjà comptés
    // fixés — cf. doc ci-dessus : n_fixed_z1 est compté séparément, seuls les
    // z encore réellement libres (valeur 2) entrent dans N_z/A_g/b_res_g/c_g).
    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<std::vector<int8_t>>& z_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<Vec2d>& b_res_list, int depth,
                      std::vector<double>& ub_relative_out,
                      GNNBatchCallTiming& timing);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
