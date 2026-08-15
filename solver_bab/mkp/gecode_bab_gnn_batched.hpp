#pragma once
/*
 * gecode_bab_gnn_batched.hpp — Étape 2e du plan de batching : intègre
 * GNNSdEditNativeBatched dans le vrai moteur B&B, en cassant la fusion
 * commit()→more()→status() de MKPSpaceV1LD (gecode_bab.hpp) — ici, commit()
 * fixe juste la variable, AUCUN calcul de borne. Le calcul est différé,
 * regroupé par bucket de n_g (même mécanisme que gecode_bab_bucketed.cpp,
 * Étape 1 — risque algorithmique déjà mesuré comme faible), et effectué en
 * UN SEUL appel GNNSdEditNativeBatched::query_batch() par bucket flush.
 *
 * Nouvelle classe MKPSpaceGNNBatched (PAS une réutilisation de MKPSpaceV1LD).
 * b_res EST hérité incrémentalement depuis le 2026-07-18 (cf.
 * MKPSpaceGNNBatched::b_res/apply_item_fixed) — avant cette date, le mode GNN
 * n'en avait pas besoin car GNNSdEditNativeBatched::query_batch()
 * reconstruisait b_res intégralement depuis le masque à chaque appel (coût
 * croissant avec la profondeur) ; ce n'est plus le cas, le backend GNN reçoit
 * maintenant b_res déjà à jour (cf. GNNBoundBackendBatch::query_batch).
 * fixed_mask/fp_running restent, eux, non maintenus (fp reste extrait via un
 * scan dans extract_pending, coût O(n) jugé négligeable face à b_res). Réutilise
 * MKPSharedData (gecode_bab.hpp) TELLE QUELLE pour les données du problème
 * (c/A/b/c_int/A_int/b_int/branch_order/obj_max/initial_lb_int/SCALE).
 *
 * Fichier séparé de gecode_bab.hpp/.cpp — AUCUNE modification du code de
 * production (même convention que l'Étape 1).
 */

#include "gecode_bab.hpp"
#include "gnn_sd_edit_native_batched.hpp"

#include <limits>

// ═════════════════════════════════════════════════════════════════════════════
// MKPSpaceGNNBatched — espace de recherche Gecode, borne calculée EXTERNEMENT
// ═════════════════════════════════════════════════════════════════════════════
class MKPSpaceGNNBatched : public IntMaximizeSpace {
public:
    BoolVarArray x, z;
    IntVar       obj;
    const MKPSharedData* data;   // read-only, non cloné (même patron que MKPSpaceV1LD)

    // Capacités résiduelles, HÉRITÉES du parent au clonage (copie simple,
    // O(N*m)) puis mises à jour incrémentalement par apply_item_fixed()
    // (O(N*m), UNIQUEMENT quand l'item vient d'être fixé à 1 — rien à faire
    // sinon, b_res hérité reste correct) — remplace la reconstruction
    // depuis le masque complet (O(profondeur*N*m), coût croissant avec la
    // profondeur) qui se faisait auparavant DANS le backend GNN à chaque
    // appel query_batch (cf. échange du 2026-07-18, même principe déjà
    // utilisé par MKPSharedData::b_res dans le moteur séquentiel,
    // gecode_bab.hpp).
    Vec2d b_res;

    // ── SimpleBrancher — identique à BoundBrancher (gecode_bab.hpp) sauf que
    // commit() NE CALCULE AUCUNE BORNE : fixe juste la variable. Le calcul
    // est fait par le driver (solve_gecode_bab_mkp_gnn_batched), en dehors du
    // cycle commit()/status() de Gecode. ─────────────────────────────────────
    class SimpleBrancher : public Brancher {
    public:
        // Public (pas protected) : le driver (gecode_bab_gnn_batched.cpp) a
        // besoin de lire pos/val après commit() pour mettre à jour
        // MKPSpaceGNNBatched::b_res de façon incrémentale (SEUL l'item
        // pos vient de changer), sans re-scanner tout le masque — cf. doc
        // sur b_res plus bas (héritage incrémental, échange du 2026-07-18).
        struct PosVal : public Choice {
            int pos, val;
            PosVal(const SimpleBrancher& b, int p, int v) : Choice(b, 2), pos(p), val(v) {}
            void archive(Archive& e) const override { Choice::archive(e); e << pos << val; }
        };

    protected:
        ViewArray<Int::BoolView> vars;

    public:
        SimpleBrancher(Home home, ViewArray<Int::BoolView>& v) : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::BoolView>& v) { new (home) SimpleBrancher(home, v); }

        size_t dispose(Space& home) override { Brancher::dispose(home); return sizeof(*this); }

        SimpleBrancher(Space& home, SimpleBrancher& b) : Brancher(home, b) { vars.update(home, b.vars); }

        Brancher* copy(Space& home) override { return new (home) SimpleBrancher(home, *this); }

        bool status(const Space&) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        Choice* choice(Space& home) override {
            const auto& order = static_cast<MKPSpaceGNNBatched&>(home).data->branch_order;
            for (int k : order)
                if (!vars[k].assigned())
                    return new PosVal(*this, k, 1);
            GECODE_NEVER;
            return nullptr;
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos, val; e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        // commit() : fixe vars[pos]=val, RIEN D'AUTRE — pas de more(), pas de
        // borne. C'est la différence structurelle avec BoundBrancher.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos;
            int val = (a == 0) ? 1 : 0;
            return me_failed(vars[pos].eq(home, val)) ? ES_FAILED : ES_OK;
        }

        void print(const Space&, const Choice& c, unsigned int a, std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            os << "x[" << pv.pos << "] = " << (a == 0 ? 1 : 0);
        }
    };

    // ── Constructeur racine — IDENTIQUE au modèle CP de MKPSpaceV1LD (channel,
    // réification z_i=1→A_i x≤b_i, chance constraint, objectif, incumbent
    // initial) — cf. gecode_bab.hpp:515-577, juste SimpleBrancher au lieu de
    // BoundBrancher. ─────────────────────────────────────────────────────────
    explicit MKPSpaceGNNBatched(const MKPSharedData* d)
        : x(*this, d->n, 0, 1), z(*this, d->N, 0, 1), obj(*this, 0, d->obj_max), data(d), b_res(d->b)
    {
        int n = d->n, m = d->m, N = d->N;
        int min_feas = static_cast<int>(std::ceil((1.0 - d->epsilon) * N));

        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j) {
                IntArgs coefs(n);
                for (int k = 0; k < n; ++k) coefs[k] = d->A_int[i][j][k];
                linear(*this, coefs, x_int, IRT_LQ, d->b_int[i][j], Reify(z[i], RM_IMP));
            }

        linear(*this, z, IRT_GQ, min_feas);

        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = d->c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        if (d->initial_lb_int > 0)
            rel(*this, obj >= d->initial_lb_int);

        BoolVarArgs x_args(n);
        for (int k = 0; k < n; ++k) x_args[k] = x[k];
        ViewArray<Int::BoolView> xv(*this, x_args);
        SimpleBrancher::post(*this, xv);

        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    MKPSpaceGNNBatched(MKPSpaceGNNBatched& s) : IntMaximizeSpace(s), data(s.data), b_res(s.b_res)
    {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }

    Space* copy() override { return new MKPSpaceGNNBatched(*this); }

    IntVar cost() const override { return obj; }

    // Met à jour b_res pour tenir compte du SEUL item k venant d'être fixé
    // à 1 (O(N*m)) — à appeler par le driver juste après commit() quand
    // pv.val==1 (cf. SimpleBrancher::PosVal, rendu public pour ça). Rien à
    // faire si l'item est fixé à 0 : b_res hérité du parent reste correct.
    void apply_item_fixed(int k) {
        for (int i = 0; i < data->N; ++i)
            for (int j = 0; j < data->m; ++j)
                b_res[i][j] -= data->A[i][j][k];
    }

    // true tant qu'il reste une variable x non assignée. Sert au driver à
    // savoir SI le prochain choice()/commit() porte sur x (SimpleBrancher,
    // PosVal exploitable pour b_res) ou sur z (brancher natif Gecode une
    // fois tous les x fixés, cf. zone "z-branching" — caster son Choice en
    // PosVal serait un comportement indéfini). Un nœud avec tous les x fixés
    // peut tout à fait rester SS_BRANCH (z pas encore résolu) et repasser
    // par ce point plus tard, cf. le bypass ng_depth==0 qui pousse de tels
    // nœuds dans `open`.
    bool has_free_x() const {
        for (int k = 0; k < data->n; ++k)
            if (!x[k].assigned()) return true;
        return false;
    }

    // Extrait (x_mask, fp) pour l'appel GNN — même logique que l'Étape 1 de
    // more() (gecode_bab.hpp:696-715). b_res est maintenu à part (cf.
    // apply_item_fixed) — plus reconstruit depuis le masque complet par le
    // backend GNN à chaque appel (cf. échange du 2026-07-18).
    void extract_pending(std::vector<int8_t>& x_mask, double& fp) const {
        int n = data->n;
        x_mask.assign(n, 2);
        fp = 0.0;
        for (int k = 0; k < n; ++k) {
            if (!x[k].assigned()) continue;
            int val = x[k].val();
            x_mask[k] = static_cast<int8_t>(val);
            if (val == 1) fp += data->c[k];
        }
    }

    // Poste la borne calculée EXTERNEMENT (par GNNSdEditNativeBatched) — la
    // queue de l'actuel more() (gecode_bab.hpp:774-782), sans le calcul.
    void post_gnn_bound(int ub_int) {
        if (ub_int < obj.max())
            rel(*this, obj <= ub_int);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Résultat + point d'entrée
// ═════════════════════════════════════════════════════════════════════════════
struct GNNBatchedBabResult : public GecodeV1LDResult {
    int     K                         = 1;
    int64_t n_bucket_flushes          = 0;
    double  mean_bucket_fill_at_flush = 0.0;
    int64_t max_pending_live          = 0;
    // Nombre d'enfants (ng_depth > 0, donc pas la zone z-branching) traités
    // IMMÉDIATEMENT sans passer par un bucket/appel GNN, parce que
    // depth = n - ng_depth dépassait gnn_max_depth_frac * n (cf. doc sur ce
    // paramètre plus bas). 0 si gnn_max_depth_frac = 1.0 (comportement
    // historique, jamais de bypass).
    int64_t nodes_bypassed_depth_cutoff = 0;
    // Symétrique, pour gnn_node_stride (cf. doc plus bas) : nombre d'enfants
    // candidats à un appel GNN (ng_depth>0, depth_here<=cutoff) sautés parce
    // que ce n'était pas leur tour dans le cycle de stride. 0 si
    // gnn_node_stride <= 1 (comportement historique, jamais de bypass).
    int64_t nodes_bypassed_stride = 0;
    // Borne GNN au premier niveau réellement calculé (enfants directs de la
    // racine, gnn_depth==1) — la racine elle-même n'a jamais de borne GNN
    // dans ce moteur (cf. commentaire "Racine" dans gecode_bab_gnn_batched.cpp).
    // Sert de proxy pour le "gap à la racine" rapporté par les modes gnn/
    // gnn_root (bound_mode != gnn_batched) : max(ub) sur les membres de ce
    // premier bucket flushé à gnn_depth==1 (au plus 2 : les deux enfants du
    // branchement racine), car UB(racine) = max_i UB(enfant_i) pour une
    // relaxation admissible. NaN si jamais capturé (n<=1 ou bucket vidé sans
    // flush avant timeout).
    double  first_level_bound_value      = std::numeric_limits<double>::quiet_NaN();
    int64_t first_level_bound_n_members  = 0;
};

// gnn_max_depth_frac : symétrique de MKPSharedData::gnn_max_depth_frac
// (gecode_bab.hpp) — mais ici la sémantique de depth diffère légèrement de la
// version séquentielle : chaque bucket pending[ng_depth] contient déjà des
// nœuds à profondeur EXACTEMENT depth = n - ng_depth (le bucketing par n_g EST
// un bucketing par profondeur), donc le seuil s'applique bucket par bucket,
// pas nœud par nœud. Au-delà du seuil, l'enfant est poussé directement dans
// `open` avec son obj.max() courant (post-CP, sans borne GNN) — même
// comportement que la zone ng_depth==0 (z-branching), pas un appel GNN
// "sauté silencieusement" comme dans more() (ici on ne bufferise même pas
// l'entrée pending, on évite tout travail, y compris extract_pending côté
// x_mask/fp qui a déjà eu lieu mais dont le résultat est simplement jeté).
// Défaut 1.0 = jamais de bypass (comportement historique).
// use_surrogate_zeta : cf. doc sur GNNSdEditNativeBatched (gnn_sd_edit_native_batched.hpp).
// Défaut false = comportement historique inchangé (MkpBnB).
//
// gnn_node_stride : throttle ORTHOGONAL à gnn_max_depth_frac — au lieu d'un
// seuil de profondeur fixe, sous-échantillonne les candidats à un appel GNN
// par un compteur global (incrémenté UNIQUEMENT sur les candidats qui
// passeraient le filtre depth_cutoff/z-branching) : 1 candidat sur
// gnn_node_stride obtient réellement sa borne GNN, les autres sont bypassés
// EXACTEMENT comme le cutoff de profondeur (poussés dans `open` avec leur
// obj.max() post-CP, aucune borne postée). Motivation : diagnostic terrain
// (beaucoup de ub_tightened, peu de fails_from_bound — la borne GNN élague
// rarement alors qu'elle coûte cher à chaque nœud) — teste si résoudre CE
// problème par sous-échantillonnage pur (indépendant de la profondeur) fait
// mieux que le gating par profondeur déjà mesuré comme modeste (cf. mémoire
// gnn_depth_gated_threshold_result / gnn_max_depth_frac_batched_result).
// Défaut 1 = jamais de bypass (comportement historique, un appel par candidat).
//
// gnn_backend : "native" (défaut, TorchScript, GNNSdEditNativeBatched) ou
// "native_aoti" (AOTInductor, GNNSdEditNativeAOTIBatched — cf.
// gnn_sd_edit_native_aoti_batched.hpp) : un artefact .pt2 par n_g, B
// dynamique à l'intérieur (feasibility validée le 2026-07-17, cf.
// gnn_sd_edit/probe_aoti_batched_dynamic_B.py) — MAIS bug compilateur
// CUDA/Inductor non résolu sur cette voie (device-side assert, cf. mémoire
// "aoti-batched-gru-decomposed-abandoned"), NE PAS UTILISER EN PRODUCTION.
// "native_aoti_static" (AOTInductor, GNNSdEditNativeAOTIStaticBatched — cf.
// gnn_sd_edit_native_aoti_batched_static.hpp) : même principe mais B FIGÉ =
// K à l'export (un artefact par n_g, compilé pour EXACTEMENT K graphes),
// contourne le bug ci-dessus ; tout bucket partiel est complété par du
// padding avant l'appel (coût : calcul gaspillé proportionnel à
// (K-B_réel)/K, négligeable si les buckets sont presque toujours pleins).
// Même contrat query_batch (GNNBoundBackendBatch), résultats numériquement
// équivalents (~1e-5, pas bit-identiques comme bnb/surrogate — Inductor
// recompile les kernels).
GNNBatchedBabResult solve_gecode_bab_mkp_gnn_batched(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    double time_lim, int K,
    std::string gnn_checkpoint_path, std::string gnn_device = "cuda",
    double gnn_max_depth_frac = 1.0,
    bool use_surrogate_zeta = false,
    int gnn_node_stride = 1,
    std::string gnn_backend = "native");
