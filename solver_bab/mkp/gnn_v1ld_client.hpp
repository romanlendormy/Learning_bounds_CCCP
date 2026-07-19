#pragma once
/*
 * gnn_v1ld_client.hpp — Pont vers un serveur d'inférence GNN one-shot
 * (gnn/gnn_inference_server.py), pour remplacer la descente de sous-gradient
 * V1LD par une borne calculée en un seul passage avant.
 *
 * Le serveur Python est un process persistant, lancé une fois (pipe()+fork()+
 * execvp()) pour toute la durée de la résolution B&B, et interrogé à chaque
 * nœud via un protocole binaire à taille fixe sur stdin/stdout (cf. docstring
 * de gnn_inference_server.py pour le détail des messages). Pas de LibTorch ni
 * d'ONNX côté C++ : on paie une latence IPC par appel, mesurée et isolée du
 * temps de calcul réel (cf. GNNCallTiming) pour ne pas fausser la comparaison
 * avec la borne sous-gradient qu'on cherche à battre.
 *
 * Restriction : lagrangian_v1ld() (côté Python) suppose p_i = 1/N (scénarios
 * équiprobables) — le constructeur vérifie cette hypothèse et lève une
 * exception si elle n'est pas respectée, plutôt que de laisser le serveur
 * renvoyer silencieusement une borne invalide.
 */

#include "gnn_bound_backend.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

class GNNV1LDClient : public GNNBoundBackend {
public:
    // Spawn le process Python et lui envoie l'instance complète (une seule
    // fois). Lève std::invalid_argument si p n'est pas uniforme, et
    // std::runtime_error si le spawn/l'INIT échoue (y compris mismatch de m
    // avec le checkpoint, remonté par le serveur dans son ACK).
    // variant : "homogene" (défaut, arêtes scénario = clique) ou "sd_edit"
    // (arêtes scénario = étoile ancrée sur le scénario 0, cf.
    // gnn_sd_edit/train_v1LD_sd_edit.py::build_graph) — transmis tel quel au
    // serveur Python (--variant), qui choisit le module build_graph/GNN à
    // importer en conséquence (poids compatibles dans les deux cas, seule la
    // topologie du graphe construit change).
    GNNV1LDClient(const std::string& python_exe,
                  const std::string& server_script,
                  const std::string& checkpoint_path,
                  const std::string& device,
                  int n, int m, int N, double epsilon,
                  const Vec1d& c, const Vec1d& p,
                  const Vec3d& A, const Vec2d& b,
                  const std::string& variant = "homogene");

    ~GNNV1LDClient();

    GNNV1LDClient(const GNNV1LDClient&)            = delete;
    GNNV1LDClient& operator=(const GNNV1LDClient&) = delete;

    // x_mask[k] ∈ {0,1,2} (0=fixé_0, 1=fixé_1, 2=libre), curr_cost = profit
    // déjà fixé (fp), depth = nombre de variables fixées. Retourne
    // ub_relative = L1(λ) sur le sous-problème résiduel (même convention que
    // v1LD_star côté C++ : l'appelant calcule ub = fp + ub_relative). Remplit
    // timing avec la décomposition temporelle renvoyée par le serveur.
    //
    // b_res (signature imposée par GNNBoundBackend, cf. revue niveau 1 point
    // B) n'est PAS utilisé ici : le protocole binaire vers
    // gnn_inference_server.py ne transmet que x_mask, et le serveur Python
    // recalcule b_res depuis x_mask de son côté (même limitation qu'avant
    // cette revue). Ce backend IPC n'est plus le chemin de production
    // (gnn_backend=native/native_aoti), donc pas retenu dans le périmètre de
    // ce correctif — corriger le protocole (transmettre b_res, adapter
    // gnn_inference_server.py) resterait à faire séparément si l'IPC redevient
    // pertinent.
    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    int n_;
    pid_t child_pid_   = -1;
    int   fd_to_child_ = -1;
    int   fd_from_child_ = -1;

    void write_all(const void* buf, size_t len);
    void read_all(void* buf, size_t len);
};
