"""
Tests Module A — automate paramétrique des quarts légaux.

Tests :
  NON-VACUITÉ  : il existe au moins un mot accepté de longueur T
  SPOT-CHECK   : les mots acceptés respectent toutes les contraintes sémantiques
  DÉTERMINISME : même seed → même automate
  ROUND-TRIP   : sérialisation/désérialisation → même DFA
  REFUS        : quelques mots invalides connus sont bien refusés
  ACCEPTATION  : quelques mots valides connus sont bien acceptés
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import pytest
import numpy as np
from ssp_stochastic.automaton import (
    AutomatonParams, build_shift_dfa, is_nonempty_at_length,
    sample_words, check_word_semantics, make_shift_automaton, DFA, ALPHABET,
)

# ── paramètres jouet fixes pour tests reproductibles ─────────────────────────

T_SMALL = 10

PARAMS_SIMPLE = AutomatonParams(
    L_min=4, L_max=6, m=3, n_P=1, meal_len=1,
    a_min=1, b_max=4, win_start=2, win_end=5,
)


# ── non-vacuité ───────────────────────────────────────────────────────────────

def test_nonempty_small():
    """Il existe au moins un mot accepté de longueur T."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    assert is_nonempty_at_length(dfa, T_SMALL), \
        "Langage vide pour PARAMS_SIMPLE et T=10"


def test_nonempty_various_T():
    """Non-vacuité pour plusieurs T."""
    for T in (8, 12, 16, 24):
        p = AutomatonParams(L_min=4, L_max=6, m=3, n_P=1, meal_len=1,
                            a_min=1, b_max=5, win_start=2, win_end=5)
        dfa = build_shift_dfa(p)
        assert is_nonempty_at_length(dfa, T), f"Langage vide pour T={T}"


# ── acceptation de mots connus ─────────────────────────────────────────────────

def test_accept_known_valid():
    """
    PARAMS_SIMPLE : L_min=4, L_max=6, m=3, n_P=1, meal_len=1,
                    a_min=1, b_max=4, win_start=2, win_end=5, T=10.
    Mot valide : OO W W R W P W W OO
      - service = WWRWPWW (longueur 7 ? non, L_max=6)
    Prenons : OO W W R P W W O O  → service=WWRPWW (len=6), W avant R=2≥1 ✓,
              pos_R=3≤4=b_max ✓, P à pos 4 ∈ [2,5] ✓, last != P ✓.
    En notation compacte (T=10): OOWWRPWWOO
    """
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    word = "OOWWRPWWOO"
    assert dfa.accepts(word), f"Mot valide refusé : {word}"
    violations = check_word_semantics(word, PARAMS_SIMPLE)
    assert not violations, f"check_word_semantics donne des violations : {violations}"


def test_accept_minimal_service():
    """Service de longueur L_min."""
    # Service = W R P W (len=4=L_min), a_min=1 (1 W avant R) ✓, P pos 3 ∈ [2,5] ✓
    p = AutomatonParams(L_min=4, L_max=6, m=3, n_P=1, meal_len=1,
                        a_min=1, b_max=3, win_start=2, win_end=5)
    dfa = build_shift_dfa(p)
    # OO WRPW OOOO  → service=WRPW, ok
    word = "OOWRPWOOOO"
    assert len(word) == T_SMALL
    assert dfa.accepts(word), f"Mot valide refusé : {word}"


# ── refus de mots invalides ────────────────────────────────────────────────────

def test_reject_p_first():
    """P en première période de service → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    # service commence par P
    word = "OOPWRWWWOO"
    assert not dfa.accepts(word)


def test_reject_p_last():
    """P en dernière période de service → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    # service = WWRWWP (len=6), P en dernière pos
    word = "OOWWRWWPOO"
    assert not dfa.accepts(word)


def test_reject_no_meal():
    """Pas de repas (aucun R) → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    # service = WWWPWW (len=6, n_P=1 ✓ mais pas de R)
    word = "OOWWWPWWOO"
    assert not dfa.accepts(word)


def test_reject_service_too_short():
    """Service trop court (< L_min) → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    # Service=WRW (len=3 < L_min=4)
    word = "OOOO WRW OOO".replace(" ", "")  # len=10
    word = "OOOOWRWOOO"
    assert not dfa.accepts(word)


def test_reject_two_meals():
    """Deux blocs R → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    # service = WRPRW (deux R, un P)
    word = "OOWRPRWOOO"
    assert not dfa.accepts(word)


def test_reject_wrong_p_count():
    """Mauvais nombre de P → refus."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)  # n_P=1
    # service sans P : WWRWWW
    word = "OOWWRWWWOO"
    assert not dfa.accepts(word)


# ── spot-check sémantique ─────────────────────────────────────────────────────

def test_sampled_words_valid():
    """Tous les mots échantillonnés satisfont check_word_semantics."""
    rng = np.random.default_rng(42)
    p = PARAMS_SIMPLE
    dfa = build_shift_dfa(p)
    words = sample_words(dfa, T_SMALL, n=50, rng=rng)
    assert len(words) > 0, "Aucun mot échantillonné"
    for w in words:
        violations = check_word_semantics(w, p)
        assert not violations, f"Mot {w} invalide : {violations}"


def test_sampled_words_various_params():
    """Spot-check sur plusieurs jeux de paramètres."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        from ssp_stochastic.automaton import sample_automaton_params
        p = sample_automaton_params(rng)
        T = 16
        dfa = build_shift_dfa(p)
        if not is_nonempty_at_length(dfa, T):
            continue  # skip si vide pour ce T
        words = sample_words(dfa, T, n=20, rng=rng, max_attempts=2000)
        for w in words:
            violations = check_word_semantics(w, p)
            assert not violations, f"Params={p}\nMot={w}\nViolations={violations}"


# ── déterminisme ──────────────────────────────────────────────────────────────

def test_determinism():
    """Même paramètres → même DFA (mêmes états, transitions, acceptants)."""
    dfa1 = build_shift_dfa(PARAMS_SIMPLE)
    dfa2 = build_shift_dfa(PARAMS_SIMPLE)
    assert dfa1.num_states == dfa2.num_states
    assert dfa1.initial == dfa2.initial
    assert dfa1.accepting == dfa2.accepting
    assert dfa1.delta == dfa2.delta


def test_determinism_with_seed():
    """Sample params avec même seed → même automate."""
    from ssp_stochastic.automaton import sample_automaton_params
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    p1 = sample_automaton_params(rng1)
    p2 = sample_automaton_params(rng2)
    assert p1 == p2
    dfa1 = build_shift_dfa(p1)
    dfa2 = build_shift_dfa(p2)
    assert dfa1.delta == dfa2.delta


# ── round-trip sérialisation ──────────────────────────────────────────────────

def test_roundtrip_serialization():
    """to_dict() → from_dict() donne un DFA identique."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    d = dfa.to_dict()
    dfa2 = DFA.from_dict(d)
    assert dfa.num_states == dfa2.num_states
    assert dfa.initial == dfa2.initial
    assert dfa.accepting == dfa2.accepting
    assert dfa.delta == dfa2.delta


def test_roundtrip_accepts_same():
    """Après round-trip, accept/reject identiques sur des mots tests."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    dfa2 = DFA.from_dict(dfa.to_dict())
    rng = np.random.default_rng(11)
    words = sample_words(dfa, T_SMALL, n=20, rng=rng)
    for w in words:
        assert dfa.accepts(w) == dfa2.accepts(w)


# ── structure du DFA ─────────────────────────────────────────────────────────

def test_accepting_states_exist():
    """Le DFA a au moins un état acceptant."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    assert len(dfa.accepting) > 0


def test_transitions_use_valid_alphabet():
    """Toutes les transitions utilisent des symboles de ALPHABET."""
    dfa = build_shift_dfa(PARAMS_SIMPLE)
    for (src, sym), dst in dfa.delta.items():
        assert sym in ALPHABET
        assert 0 <= src < dfa.num_states
        assert 0 <= dst < dfa.num_states


# ── make_shift_automaton ──────────────────────────────────────────────────────

def test_make_shift_automaton_ok():
    dfa = make_shift_automaton(PARAMS_SIMPLE, T=T_SMALL)
    assert is_nonempty_at_length(dfa, T_SMALL)


def test_make_shift_automaton_empty_raises():
    """Paramètres impossibles → RuntimeError."""
    # Service doit avoir longueur ≥ L_min = 10, mais T = 8 → impossible
    p = AutomatonParams(L_min=10, L_max=12, m=3, n_P=1, meal_len=1,
                        a_min=1, b_max=9, win_start=2, win_end=11)
    with pytest.raises(RuntimeError):
        make_shift_automaton(p, T=8)
