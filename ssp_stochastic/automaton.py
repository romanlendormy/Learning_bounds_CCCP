"""
automaton.py — Module A : automate paramétrique des quarts légaux.

Architecture
------------
On construit un DFA à état-tuple (sp, cw, pc, ms, wbm, last_p), puis on
minimise avec l'algorithme de Hopcroft.  Cela revient à l'intersection de
mini-DFAs (amplitude, work-max, pause, repas) mais sans explosion d'états
intermédiaires grâce à l'encodage direct des invariants.

Alphabet fixe : Σ = {O, P, R, W}  (ordre lexicographique, immuable).

Composantes de l'état-tuple
---------------------------
sp   : position dans le service (0 = pré-service, 1..L_max = en service,
       L_max+1 = post-service)
cw   : nombre de W consécutifs courants (0..m)
pc   : nombre de P vus dans le service (0..n_P)
ms   : état du repas  (0 = pas encore, 1..meal_len = en cours, meal_len+1 = terminé)
wbm  : travail avant repas, plafonné à a_min (0..a_min)
last_p : True si le dernier symbole de service était P (pour interdire P en
         dernière période)

Contraintes encodées
--------------------
1. Amplitude     : mot = O* [service de longueur L_min..L_max] O*
2. Work-max      : jamais plus de m symboles W consécutifs
3. Pauses        : exactement n_P symboles P dans le service, interdits en
                   première et dernière période ; fenêtre [win_start, win_end]
4. Repas         : un seul bloc R de longueur meal_len, commençant après ≥ a_min
                   symboles W, et démarrant à la position de service ≤ b_max

Déterminisme
------------
Toutes les itérations sur ensembles/dicts utilisent des listes triées pour
que la numérotation des états et la minimisation soient reproducibles.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# ── constantes globales ────────────────────────────────────────────────────────

ALPHABET: tuple[str, ...] = ('O', 'P', 'R', 'W')  # ordre fixe, trié
WORK_IND: dict[str, int] = {'W': 1, 'P': 0, 'R': 0, 'O': 0}

_DEAD = object()   # sentinelle interne (n'est jamais dans state_map)


# ── dataclass DFA ──────────────────────────────────────────────────────────────

@dataclass
class DFA:
    """
    Automate fini déterministe sur ALPHABET.

    Attributs
    ---------
    num_states : nombre d'états (entiers 0..num_states-1)
    delta      : dict {(état, sym): état_suivant}  — partiel (absent = bloquant)
    initial    : état initial
    accepting  : frozenset d'états acceptants
    """
    num_states: int
    delta: dict  # (int, str) -> int
    initial: int
    accepting: frozenset

    # ── exécution ──────────────────────────────────────────────────────────────

    def run(self, word: str) -> int | None:
        """Exécute le mot ; retourne l'état final ou None si bloqué."""
        state = self.initial
        for sym in word:
            nxt = self.delta.get((state, sym))
            if nxt is None:
                return None
            state = nxt
        return state

    def accepts(self, word: str) -> bool:
        """True si le DFA accepte le mot."""
        s = self.run(word)
        return s is not None and s in self.accepting

    # ── sérialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Représentation JSON-compatible (pour writers.py)."""
        return {
            "num_states": self.num_states,
            "alphabet": list(ALPHABET),
            "initial": self.initial,
            "accepting": sorted(self.accepting),
            "transitions": sorted(
                (src, sym, dst)
                for (src, sym), dst in self.delta.items()
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DFA":
        """Reconstruit un DFA depuis to_dict()."""
        delta = {(src, sym): dst for src, sym, dst in d["transitions"]}
        return cls(
            num_states=d["num_states"],
            delta=delta,
            initial=d["initial"],
            accepting=frozenset(d["accepting"]),
        )


# ── paramètres de l'automate ──────────────────────────────────────────────────

@dataclass(frozen=True)
class AutomatonParams:
    """
    Paramètres tirés par épisode pour définir les contraintes du quart.

    L_min, L_max : longueur du service (symboles non-O)
    m            : max W consécutifs
    n_P          : nombre exact de pauses P dans le service
    meal_len     : longueur du bloc repas (symboles R)
    a_min        : min symboles W avant le repas
    b_max        : position max dans le service pour démarrer le repas (1-indexé)
    win_start    : 1er position autorisée pour P dans le service (1-indexé, ≥ 2)
    win_end      : dernière position autorisée pour P (1-indexé, ≤ L_max-1)
    """
    L_min: int
    L_max: int
    m: int
    n_P: int
    meal_len: int
    a_min: int
    b_max: int
    win_start: int
    win_end: int


def sample_automaton_params(
    rng: np.random.Generator,  # type: ignore[name-defined]
    L_min_range: tuple[int, int] = (4, 6),
    L_max_range: tuple[int, int] = (6, 8),
    m_choices: tuple[int, ...] = (2, 3, 4),
    n_P_choices: tuple[int, ...] = (1, 2),
    meal_len_choices: tuple[int, ...] = (1, 2),
    a_min_range: tuple[int, int] = (1, 3),
) -> AutomatonParams:
    """
    Tire aléatoirement les paramètres d'un automate de quart.

    Les plages sont configurables ; les valeurs retournées respectent les
    contraintes de cohérence (L_min ≤ L_max, fenêtres valides, etc.).
    """
    import numpy as np

    L_min = int(rng.integers(L_min_range[0], L_min_range[1] + 1))
    L_max = int(rng.integers(max(L_min, L_max_range[0]), L_max_range[1] + 1))
    m     = int(rng.choice(sorted(m_choices)))
    n_P   = int(rng.choice(sorted(n_P_choices)))
    meal_len = int(rng.choice(sorted(meal_len_choices)))

    # a_min : nb de W requis avant le repas (≥ 1, ≤ L_max - meal_len - 1)
    a_min_hi = max(1, L_max - meal_len - 1)
    a_min = int(np.clip(rng.integers(a_min_range[0], a_min_range[1] + 1), 1, a_min_hi))

    # b_max : position max de début du repas (laisse room pour au moins 1 W après)
    # b_max ∈ [a_min + 1, L_max - meal_len]
    b_max_lo = a_min + 1
    b_max_hi = L_max - meal_len
    if b_max_lo > b_max_hi:
        b_max = b_max_hi  # contrainte serrée : on prend le max possible
    else:
        b_max = int(rng.integers(b_max_lo, b_max_hi + 1))

    # Fenêtre P : [win_start, win_end] ⊂ [2, L_max-1]
    win_start = 2
    win_end   = max(win_start, L_max - 1)

    return AutomatonParams(
        L_min=L_min, L_max=L_max, m=m, n_P=n_P,
        meal_len=meal_len, a_min=a_min, b_max=b_max,
        win_start=win_start, win_end=win_end,
    )


# ── fonction de transition (cœur du DFA) ──────────────────────────────────────

def _step(state: Any, sym: str, p: AutomatonParams) -> Any:
    """
    Calcule l'état suivant depuis `state` en lisant `sym`.
    Retourne _DEAD si la transition est invalide.
    """
    sp, cw, pc, ms, wbm, last_p = state

    # ── pré-service ────────────────────────────────────────────────────────────
    if sp == 0:
        if sym == 'O':
            return (0, 0, pc, ms, wbm, False)
        if sym == 'P':
            return _DEAD                          # P interdit en 1re période
        # sym ∈ {W, R} → entrée en service
        new_sp = 1
        if sym == 'W':
            new_cw = 1
            if new_cw > p.m:
                return _DEAD
            new_ms, new_wbm = _meal_step_W(ms, wbm, p, new_sp)
            if new_ms is _DEAD:
                return _DEAD
        else:  # R
            new_cw = 0
            new_ms, new_wbm = _meal_step_R(ms, wbm, p, new_sp)
            if new_ms is _DEAD:
                return _DEAD
        return (new_sp, new_cw, pc, new_ms, new_wbm, False)

    # ── en service ─────────────────────────────────────────────────────────────
    if 1 <= sp <= p.L_max:
        if sym == 'O':
            # Fin du service
            if sp < p.L_min:
                return _DEAD                      # service trop court
            if last_p:
                return _DEAD                      # P en dernière période
            if ms != p.meal_len + 1:
                return _DEAD                      # repas incomplet
            return (p.L_max + 1, 0, pc, ms, wbm, False)

        # Service continue — trop long ?
        if sp >= p.L_max:
            return _DEAD
        new_sp = sp + 1

        if sym == 'W':
            new_cw = cw + 1
            if new_cw > p.m:
                return _DEAD
            new_pc = pc
            new_ms, new_wbm = _meal_step_W(ms, wbm, p, new_sp)
            if new_ms is _DEAD:
                return _DEAD
            return (new_sp, new_cw, new_pc, new_ms, new_wbm, False)

        if sym == 'P':
            pos = new_sp  # position 1-indexée dans le service
            if not (p.win_start <= pos <= p.win_end):
                return _DEAD                      # hors fenêtre
            new_pc = pc + 1
            if new_pc > p.n_P:
                return _DEAD
            new_cw = 0
            new_ms, new_wbm = _meal_step_P(ms, wbm, p)
            if new_ms is _DEAD:
                return _DEAD
            return (new_sp, new_cw, new_pc, new_ms, new_wbm, True)

        # sym == 'R'
        new_cw = 0
        new_ms, new_wbm = _meal_step_R(ms, wbm, p, new_sp)
        if new_ms is _DEAD:
            return _DEAD
        return (new_sp, new_cw, pc, new_ms, new_wbm, False)

    # ── post-service ───────────────────────────────────────────────────────────
    if sym == 'O':
        return (p.L_max + 1, 0, pc, ms, wbm, False)
    return _DEAD                                  # 2e bloc de service interdit


# ── helpers pour l'état du repas ──────────────────────────────────────────────

def _meal_step_W(ms: int, wbm: int, p: AutomatonParams, new_sp: int):
    """Transition repas lors d'un W. Retourne (new_ms, new_wbm) ou (_DEAD, _DEAD)."""
    if 1 <= ms < p.meal_len:
        return _DEAD, _DEAD                       # W au milieu du bloc repas
    if ms == p.meal_len:
        return p.meal_len + 1, wbm               # fermeture du bloc repas
    # ms == 0 ou ms == meal_len+1
    new_wbm = min(wbm + 1, p.a_min) if ms == 0 else wbm
    return ms, new_wbm


def _meal_step_R(ms: int, wbm: int, p: AutomatonParams, new_sp: int):
    """Transition repas lors d'un R. Retourne (new_ms, new_wbm) ou (_DEAD, _DEAD)."""
    if ms == 0:
        if wbm < p.a_min:
            return _DEAD, _DEAD                   # pas assez de W avant repas
        if new_sp > p.b_max:
            return _DEAD, _DEAD                   # démarrage trop tardif
        return 1, wbm
    if 1 <= ms < p.meal_len:
        return ms + 1, wbm                        # continuation du bloc repas
    # ms == meal_len → R supplémentaire interdit ; ms == meal_len+1 → 2nd repas
    return _DEAD, _DEAD


def _meal_step_P(ms: int, wbm: int, p: AutomatonParams):
    """Transition repas lors d'un P."""
    if 1 <= ms < p.meal_len:
        return _DEAD, _DEAD                       # P dans le bloc repas
    if ms == p.meal_len:
        return p.meal_len + 1, wbm               # ferme le repas
    return ms, wbm                                # rien ne change


# ── condition d'acceptation ───────────────────────────────────────────────────

def _is_accepting(state: Any, p: AutomatonParams) -> bool:
    """True si l'état est acceptant après lecture de T symboles."""
    sp, cw, pc, ms, wbm, last_p = state
    if sp == 0:
        return False                              # aucun service
    if 1 <= sp <= p.L_max:
        if sp < p.L_min:
            return False                          # service trop court
        if last_p:
            return False                          # P en dernière période
        if ms != p.meal_len + 1:
            return False                          # repas incomplet
    # post-service : contraintes vérifiées à la transition (sauf pc)
    return pc == p.n_P


# ── construction du DFA ───────────────────────────────────────────────────────

def build_shift_dfa(p: AutomatonParams) -> DFA:
    """
    Construit l'automate de quart par BFS sur les états-tuples, puis minimise.

    Le résultat est un DFA minimal sur ALPHABET dont les mots acceptés de
    longueur T représentent exactement les quarts légaux selon les paramètres p.
    """
    initial_tuple = (0, 0, 0, 0, 0, False)

    # ── BFS sur les états-tuples ───────────────────────────────────────────────
    state_map: dict[Any, int] = {}
    id_to_tuple: list[Any] = []
    queue: deque = deque()

    def _get_id(t: Any) -> int:
        if t not in state_map:
            idx = len(state_map)
            state_map[t] = idx
            id_to_tuple.append(t)
            queue.append(t)
        return state_map[t]

    initial_id = _get_id(initial_tuple)

    raw_delta: dict[tuple[int, str], int] = {}

    while queue:
        tup = queue.popleft()
        sid = state_map[tup]
        for sym in ALPHABET:                      # ordre fixe → déterminisme
            nxt = _step(tup, sym, p)
            if nxt is not _DEAD:
                nxt_id = _get_id(nxt)
                raw_delta[(sid, sym)] = nxt_id

    n = len(state_map)
    accepting = frozenset(
        i for i, t in enumerate(id_to_tuple) if _is_accepting(t, p)
    )

    dfa = DFA(num_states=n, delta=raw_delta, initial=initial_id, accepting=accepting)
    return _hopcroft_minimize(dfa)


# ── minimisation de Hopcroft ──────────────────────────────────────────────────

def _hopcroft_minimize(dfa: DFA) -> DFA:
    """
    Algorithme de Hopcroft pour la minimisation DFA.

    Le DFA en entrée peut être partiel (transitions absentes = blocage).
    On ajoute un état mort explicite (trap) pour le rendre complet, puis on
    minimise et on retire l'état mort s'il est non-atteignable.

    Toutes les structures de données sont triées pour garantir le déterminisme.
    """
    Q_orig = list(range(dfa.num_states))
    DEAD_STATE = dfa.num_states          # état mort ajouté

    # DFA complet : transitions manquantes → état mort
    def delta_complete(q: int, sym: str) -> int:
        return dfa.delta.get((q, sym), DEAD_STATE)

    # États atteignables depuis initial (incluant éventuellement DEAD)
    reachable: set[int] = set()
    stack = [dfa.initial]
    while stack:
        q = stack.pop()
        if q in reachable:
            continue
        reachable.add(q)
        for sym in ALPHABET:
            nxt = delta_complete(q, sym)
            if nxt not in reachable:
                stack.append(nxt)

    # Restreindre aux états atteignables
    states = sorted(reachable)
    acc_set = frozenset(q for q in states if q in dfa.accepting)
    non_acc = frozenset(q for q in states if q not in dfa.accepting)

    if not acc_set:
        # Aucun état acceptant atteignable : DFA vide (ne devrait pas arriver)
        return DFA(num_states=1, delta={}, initial=0, accepting=frozenset())

    # Partition initiale : {F, Q\F}
    partition: list[frozenset[int]] = []
    if acc_set:
        partition.append(acc_set)
    if non_acc:
        partition.append(non_acc)

    worklist: list[frozenset[int]] = list(partition)

    part_of: dict[int, int] = {}   # état → indice de son bloc
    def _rebuild_part_of():
        for idx, blk in enumerate(partition):
            for q in blk:
                part_of[q] = idx
    _rebuild_part_of()

    while worklist:
        A = worklist.pop()
        A_set = set(A)
        for sym in ALPHABET:                      # ordre fixe
            # X = préimage de A sur sym
            X = frozenset(q for q in states if delta_complete(q, sym) in A_set)
            if not X:
                continue
            new_partition: list[frozenset[int]] = []
            new_worklist_adds: list[frozenset[int]] = []
            changed = False
            for Y in partition:
                Y1 = Y & X
                Y2 = Y - X
                if Y1 and Y2:
                    changed = True
                    new_partition.extend([Y1, Y2])
                    if Y in worklist or Y == A:
                        new_worklist_adds.extend([Y1, Y2])
                    else:
                        # Ajouter le plus petit
                        new_worklist_adds.append(Y1 if len(Y1) <= len(Y2) else Y2)
                else:
                    new_partition.append(Y)
            if changed:
                partition = new_partition
                worklist.extend(new_worklist_adds)
                _rebuild_part_of()

    # Construire le DFA quotient
    # Représentant de chaque bloc = plus petit état (déterminisme)
    repr_of: dict[int, int] = {}     # ancien état → repr du bloc
    blocks: dict[int, list[int]] = {}
    for blk in partition:
        blk_list = sorted(blk)
        rep = blk_list[0]
        for q in blk_list:
            repr_of[q] = rep
        blocks[rep] = blk_list

    # Numérotation contiguë des blocs
    all_reps = sorted(blocks.keys())
    rep_to_new: dict[int, int] = {r: i for i, r in enumerate(all_reps)}

    new_initial = rep_to_new[repr_of[dfa.initial]]
    new_accepting = frozenset(
        rep_to_new[repr_of[q]] for q in states if q in dfa.accepting
    )
    new_delta: dict[tuple[int, str], int] = {}
    for rep in all_reps:
        new_src = rep_to_new[rep]
        for sym in ALPHABET:
            nxt_orig = delta_complete(rep, sym)
            if nxt_orig in repr_of:
                nxt_rep = repr_of[nxt_orig]
                new_dst = rep_to_new[nxt_rep]
                # Omettre les transitions vers l'état mort (si c'est un bloc non-acceptant sans sortie utile)
                # On garde uniquement si DEAD_STATE n'est pas dans un bloc de repr minimal
                if nxt_orig != DEAD_STATE or DEAD_STATE in repr_of:
                    new_delta[(new_src, sym)] = new_dst
            # si nxt_orig n'est pas dans repr_of → transition morte → on n'ajoute pas

    # Retirer les self-boucles vers l'état mort (état non-acceptant absorbant)
    # On identifie l'éventuel état mort dans le quotient
    if DEAD_STATE in repr_of:
        dead_new = rep_to_new[repr_of[DEAD_STATE]]
        # Supprimer les transitions vers dead_new
        new_delta = {k: v for k, v in new_delta.items() if v != dead_new}

    n_new = len(all_reps)
    return DFA(
        num_states=n_new,
        delta=new_delta,
        initial=new_initial,
        accepting=new_accepting,
    )


# ── non-vacuité à longueur T ──────────────────────────────────────────────────

def is_nonempty_at_length(dfa: DFA, T: int) -> bool:
    """
    BFS : existe-t-il un chemin de longueur T depuis initial vers un état acceptant ?
    Complexité : O(T * |Q| * |Σ|).
    """
    if T == 0:
        return dfa.initial in dfa.accepting

    # Ensemble d'états atteignables à chaque profondeur
    current: set[int] = {dfa.initial}
    for _ in range(T):
        nxt: set[int] = set()
        for q in current:
            for sym in ALPHABET:
                q2 = dfa.delta.get((q, sym))
                if q2 is not None:
                    nxt.add(q2)
        current = nxt
        if not current:
            return False
    return bool(current & dfa.accepting)


# ── échantillonnage de mots acceptés ─────────────────────────────────────────

def sample_words(
    dfa: DFA,
    T: int,
    n: int,
    rng,              # np.random.Generator
    max_attempts: int = 10_000,
) -> list[str]:
    """
    Retourne au plus n mots de longueur T acceptés par le DFA, par marche aléatoire.
    Utilisé pour les tests de spot-check sémantique.
    """
    import numpy as np
    words = []
    attempts = 0
    while len(words) < n and attempts < max_attempts:
        attempts += 1
        state = dfa.initial
        word_chars = []
        blocked = False
        for _ in range(T):
            # Transitions disponibles depuis l'état courant
            avail = [(sym, dfa.delta[(state, sym)])
                     for sym in ALPHABET if (state, sym) in dfa.delta]
            if not avail:
                blocked = True
                break
            syms, nexts = zip(*avail)
            idx = int(rng.integers(0, len(avail)))
            word_chars.append(syms[idx])
            state = nexts[idx]
        if not blocked and state in dfa.accepting:
            words.append("".join(word_chars))
    return words


# ── validation sémantique d'un mot ────────────────────────────────────────────

def check_word_semantics(word: str, p: AutomatonParams) -> list[str]:
    """
    Vérifie manuellement les contraintes sur un mot.
    Retourne une liste de violations (vide = valide).
    """
    T = len(word)
    violations = []

    # Trouver le bloc de service
    start, end = None, None
    for i, c in enumerate(word):
        if c != 'O':
            if start is None:
                start = i
            end = i

    if start is None:
        violations.append("Pas de service (tout O)")
        return violations

    service = word[start:end + 1]
    service_len = end - start + 1

    # Contrainte amplitude
    if not all(c == 'O' for c in word[:start]):
        violations.append("Symbole non-O avant le service")
    if not all(c == 'O' for c in word[end + 1:]):
        violations.append("Symbole non-O après le service")
    if 'O' in service:
        violations.append("O dans le service (service non contigu)")
    if not (p.L_min <= service_len <= p.L_max):
        violations.append(f"Longueur service {service_len} hors [{p.L_min},{p.L_max}]")

    if service:
        # Première/dernière période
        if service[0] == 'P':
            violations.append("P en première période de service")
        if service[-1] == 'P':
            violations.append("P en dernière période de service")

    # Work-max
    cw = 0
    for c in service:
        cw = (cw + 1) if c == 'W' else 0
        if cw > p.m:
            violations.append(f"Plus de {p.m} W consécutifs")
            break

    # Fenêtre P (1-indexé)
    for i, c in enumerate(service):
        pos = i + 1
        if c == 'P' and not (p.win_start <= pos <= p.win_end):
            violations.append(f"P à position {pos} hors fenêtre [{p.win_start},{p.win_end}]")

    # Nombre exact de P
    n_p = service.count('P')
    if n_p != p.n_P:
        violations.append(f"{n_p} P trouvés, attendu {p.n_P}")

    # Repas : un seul bloc R de longueur meal_len
    # Trouver tous les blocs R dans le service
    r_blocks = []
    in_r = False
    r_start = 0
    for i, c in enumerate(service):
        if c == 'R' and not in_r:
            in_r = True
            r_start = i
        elif c != 'R' and in_r:
            in_r = False
            r_blocks.append((r_start, i - 1))
    if in_r:
        r_blocks.append((r_start, len(service) - 1))

    if len(r_blocks) != 1:
        violations.append(f"{len(r_blocks)} blocs R, attendu 1")
    else:
        rb_start, rb_end = r_blocks[0]
        rb_len = rb_end - rb_start + 1
        if rb_len != p.meal_len:
            violations.append(f"Bloc R longueur {rb_len}, attendu {p.meal_len}")
        # Travail avant repas
        work_before = sum(1 for c in service[:rb_start] if c == 'W')
        if work_before < p.a_min:
            violations.append(f"{work_before} W avant repas, minimum {p.a_min}")
        # Position de début du repas
        pos_r = rb_start + 1  # 1-indexé dans le service
        if pos_r > p.b_max:
            violations.append(f"Repas commence à position {pos_r} > b_max={p.b_max}")

    return violations


# ── construction depuis un épisode ────────────────────────────────────────────

def make_shift_automaton(
    p: AutomatonParams,
    T: int,
    max_retries: int = 10,
) -> DFA:
    """
    Construit l'automate et vérifie la non-vacuité à longueur T.
    Lève RuntimeError si le langage est vide après max_retries.
    (Pour ce module, les paramètres sont déjà cohérents ; max_retries est un garde-fou.)
    """
    dfa = build_shift_dfa(p)
    if is_nonempty_at_length(dfa, T):
        return dfa
    raise RuntimeError(
        f"Langage vide pour T={T}, params={p}. "
        "Vérifier la cohérence des paramètres (L_min ≤ T - 2·..., etc.)."
    )


# Import numpy différé (évite la dépendance au module level pour les tests unitaires)
try:
    import numpy as np
except ImportError:
    pass
