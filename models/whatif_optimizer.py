"""
models/whatif_optimizer.py — Simulation What-If + Optimisation Multi-Objectifs.

Explore les scénarios et trouve le front de Pareto entre :
  f1 : Minimiser CO₂  (émissions, proxy : retard × mode × distance)
  f2 : Minimiser Coût opérationnel (charge × fréquence × mode)
  f3 : Minimiser Retard moyen

Variables de décision :
  x[0] : Fréquence bus (1-10 rotations/h)
  x[1] : Fréquence tram (1-6 rotations/h)
  x[2] : Nb rames métro (1-20)
  x[3] : Priorité voie bus (0=aucune, 1=partielle, 2=totale)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger.warning("pymoo absent — optimisation désactivée.")


# ── Coefficients empiriques d'émission (g CO₂ / passager-km) ─────────────
CO2_COEFF = {"bus": 89.0, "tram": 12.0, "metro": 14.0}
COUT_COEFF = {"bus": 1.0,  "tram": 2.5,  "metro": 3.0}


# ── Problème pymoo ────────────────────────────────────────────────────────
class TransportOptimizationProblem(ElementwiseProblem):
    """
    Minimise simultanément CO₂, coût et retard moyen.
    Variables continues + entières.
    """

    def __init__(self, baseline: Dict):
        super().__init__(
            n_var=4,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.array([1,  1,  1,  0]),
            xu=np.array([10, 6, 20,  2]),
        )
        self.baseline = baseline   # métriques actuelles du réseau

    def _evaluate(self, x, out, *args, **kwargs):
        freq_bus, freq_tram, nb_metro, prio_bus = x

        # ── f1 : CO₂ (g / passager-km normalisé) ──────────────────────
        utilisation_bus   = min(1.0, self.baseline["demande"] / (freq_bus * 80))
        utilisation_tram  = min(1.0, self.baseline["demande"] / (freq_tram * 200))
        utilisation_metro = min(1.0, self.baseline["demande"] / (nb_metro * 1200))
        co2 = (
            CO2_COEFF["bus"]   * (1 - utilisation_bus)   * freq_bus   +
            CO2_COEFF["tram"]  * (1 - utilisation_tram)  * freq_tram  +
            CO2_COEFF["metro"] * (1 - utilisation_metro) * nb_metro
        )

        # ── f2 : Coût (unités normalisées) ────────────────────────────
        cout = (
            COUT_COEFF["bus"]   * freq_bus   +
            COUT_COEFF["tram"]  * freq_tram  +
            COUT_COEFF["metro"] * nb_metro * 5
        )

        # ── f3 : Retard moyen (secondes) ──────────────────────────────
        retard_base = self.baseline["retard_moyen"]
        bonus_prio  = prio_bus * 0.15   # voie bus dédiée réduit le retard
        bonus_freq  = (freq_bus + freq_tram) / 16 * 0.10
        retard = retard_base * max(0.0, 1.0 - bonus_prio - bonus_freq)

        # ── Contrainte : satisfaction ≥ seuil ─────────────────────────
        satisfaction_estimee = 3.0 + (1 - retard / retard_base) * 1.5
        g1 = 3.5 - satisfaction_estimee   # ≤ 0 obligatoire

        out["F"] = [co2, cout, retard]
        out["G"] = [g1]


# ── Wrapper ────────────────────────────────────────────────────────────────
class WhatIfOptimizer:

    def __init__(self):
        self.pareto_X:  np.ndarray | None = None
        self.pareto_F:  np.ndarray | None = None
        self.scenarios: List[Dict]         = []

    # ── Simulation manuelle d'un scénario ─────────────────────────────
    def simulate_scenario(
        self, baseline: Dict, params: Dict
    ) -> Dict:
        """
        Calcule l'impact d'un scénario (ex: grève = 50 % bus en moins).
        baseline : métriques actuelles
        params   : modifications (ex: {'freq_bus': -0.5, 'incident': True})
        """
        retard_new   = baseline["retard_moyen"]
        co2_new      = baseline["co2"]
        cout_new     = baseline["cout"]
        satisfaction = baseline.get("satisfaction", 3.5)

        if params.get("greve"):
            retard_new   *= 1.8
            satisfaction -= 0.8
        if params.get("pluie"):
            retard_new   *= 1.2
            co2_new      *= 1.1
        if params.get("evenement_sportif"):
            retard_new   *= 1.5
            satisfaction -= 0.3
        if "freq_bus" in params:
            delta = params["freq_bus"]   # ex: -0.3 = -30 %
            retard_new   *= (1 - delta * 0.4)
            co2_new      *= (1 + delta * 0.05)
        if params.get("voie_bus_dediee"):
            retard_new   *= 0.85
            satisfaction += 0.2

        delta_retard = retard_new - baseline["retard_moyen"]
        scenario_result = {
            "retard_moyen":        round(retard_new, 1),
            "co2":                 round(co2_new,    1),
            "cout":                round(cout_new,   1),
            "satisfaction":        round(min(5.0, max(1.0, satisfaction)), 2),
            "delta_retard_s":      round(delta_retard, 1),
            "delta_retard_pct":    round(delta_retard / baseline["retard_moyen"] * 100, 1),
        }
        self.scenarios.append({"params": params, "result": scenario_result})
        return scenario_result

    # ── Optimisation NSGA-II ────────────────────────────────────────────
    def run_pareto(self, baseline: Dict, n_gen: int = 100) -> pd.DataFrame:
        if not PYMOO_AVAILABLE:
            logger.warning("pymoo absent — génération d'un front de Pareto simulé.")
            return self._fake_pareto()

        problem     = TransportOptimizationProblem(baseline)
        algorithm   = NSGA2(
            pop_size=100,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", n_gen)
        result      = minimize(
            problem, algorithm, termination,
            seed=42, verbose=False,
        )
        self.pareto_X = result.X
        self.pareto_F = result.F

        df = pd.DataFrame(
            self.pareto_F,
            columns=["CO2 (g/pkm)", "Coût (u.a.)", "Retard moy. (s)"],
        )
        df[["freq_bus", "freq_tram", "nb_metro", "prio_bus"]] = self.pareto_X
        logger.info(f"Front de Pareto : {len(df)} solutions non-dominées")
        return df

    def _fake_pareto(self) -> pd.DataFrame:
        """Pareto simulé pour la démo quand pymoo est absent."""
        n = 50
        t  = np.linspace(0, 1, n)
        co2  = 80 - 40 * t + np.random.normal(0, 3, n)
        cout = 30 + 30 * t + np.random.normal(0, 2, n)
        ret  = 180 - 80 * t + np.random.normal(0, 10, n)
        return pd.DataFrame({
            "CO2 (g/pkm)": co2,
            "Coût (u.a.)": cout,
            "Retard moy. (s)": ret,
        })

    def get_best_compromise(self) -> Dict | None:
        """Solution au coude du front (distance min au point idéal)."""
        if self.pareto_F is None:
            return None
        F_norm = (self.pareto_F - self.pareto_F.min(0)) / (
            self.pareto_F.max(0) - self.pareto_F.min(0) + 1e-9
        )
        idx = np.argmin(np.linalg.norm(F_norm, axis=1))
        return {
            "index":        int(idx),
            "CO2":          float(self.pareto_F[idx, 0]),
            "cout":         float(self.pareto_F[idx, 1]),
            "retard":       float(self.pareto_F[idx, 2]),
            "freq_bus":     float(self.pareto_X[idx, 0]),
            "freq_tram":    float(self.pareto_X[idx, 1]),
            "nb_metro":     float(self.pareto_X[idx, 2]),
            "prio_bus":     float(self.pareto_X[idx, 3]),
        }
