# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PySCF PBC-TDA references for determinant-state pretraining.

This module deliberately delegates Hartree--Fock and TDA/CIS calculations to
PySCF. It only selects distinct dominant single promotions and adapts their
occupied MO coefficients to JaQMC's existing periodic orbital evaluator.
"""

import dataclasses
import logging
from collections.abc import Sequence
from typing import Literal

import jax
import numpy as np
import pyscf.pbc.scf.kuhf
from jax import numpy as jnp
from pyscf.pbc.lib.kpts_helper import get_kconserv_ria
from pyscf.pbc.tdscf import kuhf as pbc_tda_kuhf

from jaqmc.utils.config import configurable_dataclass

from .scf import PeriodicSCF

logger = logging.getLogger(__name__)


@configurable_dataclass
class PBCTDAReferenceConfig:
    """Configuration for the KUHF-TDA single-promotion reference adapter."""

    kshift: int = 0
    oversample: int = 4
    candidate_topk: int = 8
    min_selected_weight: float = 1e-4
    require_converged: bool = True
    require_same_k: bool = True
    occupation_tolerance: float = 1e-6

    def __post_init__(self):
        if self.kshift != 0:
            raise NotImplementedError(
                "PBC TDA subspace pretraining v1 supports kshift=0 only."
            )
        if not self.require_same_k:
            raise NotImplementedError(
                "PBC TDA subspace pretraining v1 requires same-k promotions."
            )
        if self.oversample < 0:
            raise ValueError("pretrain.tda.oversample must be non-negative")
        if self.candidate_topk < 1:
            raise ValueError("pretrain.tda.candidate_topk must be positive")
        if self.min_selected_weight < 0:
            raise ValueError("pretrain.tda.min_selected_weight must be non-negative")
        if self.occupation_tolerance <= 0:
            raise ValueError("pretrain.tda.occupation_tolerance must be positive")


@dataclasses.dataclass(frozen=True)
class PBCSinglePromotion:
    """One single-determinant proxy selected from a PBC-TDA root."""

    root_index: int
    excitation_energy: float
    spin: Literal["alpha", "beta"]
    k_occ: int
    k_vir: int
    occ_local_index: int
    vir_local_index: int
    occ_mo_index: int
    vir_mo_index: int
    amplitude: complex
    weight: float
    largest_weight: float
    total_weight: float
    participation_ratio: float

    @property
    def key(self) -> tuple[str, int, int, int, int]:
        """Return the determinant-defining promotion identity."""
        return (
            self.spin,
            self.k_occ,
            self.occ_mo_index,
            self.k_vir,
            self.vir_mo_index,
        )


def select_unique_promotions(
    candidates_by_root: Sequence[Sequence[PBCSinglePromotion]],
    n_requested: int,
    *,
    candidate_topk: int,
    min_selected_weight: float,
) -> list[PBCSinglePromotion]:
    """Greedily assign a distinct high-weight promotion to each low root."""
    if n_requested < 0:
        raise ValueError("n_requested must be non-negative")
    selected: list[PBCSinglePromotion] = []
    used: dict[tuple[str, int, int, int, int], int] = {}
    ordered_roots = sorted(
        (root for root in candidates_by_root if root),
        key=lambda root: (root[0].excitation_energy, root[0].root_index),
    )
    for candidates in ordered_roots:
        chosen = None
        for candidate in sorted(candidates, key=lambda item: item.weight, reverse=True)[
            :candidate_topk
        ]:
            if candidate.weight < min_selected_weight:
                continue
            if candidate.key in used:
                logger.info(
                    "TDA root=%s candidate skipped: promotion already assigned "
                    "to state=%s",
                    candidate.root_index,
                    used[candidate.key],
                )
                continue
            chosen = candidate
            break
        if chosen is None:
            logger.info(
                "TDA root=%s skipped: no distinct promotion in top-%s above "
                "weight threshold %.3e",
                candidates[0].root_index,
                candidate_topk,
                min_selected_weight,
            )
            continue
        used[chosen.key] = len(selected) + 1
        selected.append(chosen)
        if len(selected) == n_requested:
            break
    if len(selected) != n_requested:
        raise ValueError(
            "Unable to select enough distinct TDA promotions: requested "
            f"{n_requested}, found {len(selected)}. Increase oversample or "
            "candidate_topk, or inspect the TDA roots."
        )
    return selected


def _validate_occupations(mo_occ, tolerance: float) -> None:
    for spin_occ in mo_occ:
        for k_occ in spin_occ:
            occ = np.asarray(k_occ)
            binary = np.isclose(occ, 0.0, atol=tolerance) | np.isclose(
                occ, 1.0, atol=tolerance
            )
            if not np.all(binary):
                raise NotImplementedError(
                    "PBC TDA subspace pretraining requires integer KUHF "
                    "occupations (approximately 0 or 1)."
                )


def _root_candidates(
    *,
    root_index: int,
    excitation_energy: float,
    amplitudes,
    mo_occ,
    kconserv: np.ndarray,
) -> list[PBCSinglePromotion]:
    raw = []
    for spin_index, (spin, spin_amplitudes) in enumerate(
        (("alpha", amplitudes[0]), ("beta", amplitudes[1]))
    ):
        for k_occ, x_ia in enumerate(spin_amplitudes):
            k_vir = int(kconserv[k_occ])
            occ_abs = np.flatnonzero(np.asarray(mo_occ[spin_index][k_occ]) > 0.9)
            vir_abs = np.flatnonzero(np.asarray(mo_occ[spin_index][k_vir]) < 0.1)
            x_ia = np.asarray(x_ia)
            if x_ia.shape != (len(occ_abs), len(vir_abs)):
                raise ValueError(
                    "Unexpected PySCF TDA amplitude shape for "
                    f"spin={spin}, k={k_occ}: {x_ia.shape}, expected "
                    f"{(len(occ_abs), len(vir_abs))}."
                )
            for occ_local, vir_local in np.ndindex(x_ia.shape):
                amplitude = complex(x_ia[occ_local, vir_local])
                raw.append(
                    (
                        spin,
                        k_occ,
                        k_vir,
                        occ_local,
                        vir_local,
                        int(occ_abs[occ_local]),
                        int(vir_abs[vir_local]),
                        amplitude,
                        float(abs(amplitude) ** 2),
                    )
                )

    total_weight = float(sum(item[-1] for item in raw))
    largest_weight = float(max((item[-1] for item in raw), default=0.0))
    if total_weight > 0:
        normalized = np.asarray([item[-1] / total_weight for item in raw])
        participation_ratio = float(1.0 / np.sum(normalized**2))
    else:
        participation_ratio = float("inf")
    return [
        PBCSinglePromotion(
            root_index=root_index,
            excitation_energy=float(excitation_energy),
            spin=item[0],
            k_occ=item[1],
            k_vir=item[2],
            occ_local_index=item[3],
            vir_local_index=item[4],
            occ_mo_index=item[5],
            vir_mo_index=item[6],
            amplitude=item[7],
            weight=item[8],
            largest_weight=largest_weight,
            total_weight=total_weight,
            participation_ratio=participation_ratio,
        )
        for item in raw
    ]


def run_pbc_tda_promotions(
    mean_field,
    n_requested: int,
    config: PBCTDAReferenceConfig,
) -> list[PBCSinglePromotion]:
    """Run PySCF KUHF-TDA and select distinct dominant promotions."""
    if n_requested < 0:
        raise ValueError("n_requested must be non-negative")
    if n_requested == 0:
        return []
    if not isinstance(mean_field, pyscf.pbc.scf.kuhf.KUHF):
        raise NotImplementedError(
            "PBC TDA subspace pretraining v1 supports KUHF only."
        )
    if not bool(mean_field.converged):
        raise RuntimeError("KUHF must converge before PBC TDA pretraining.")
    _validate_occupations(mean_field.mo_occ, config.occupation_tolerance)
    nkpts = len(mean_field.kpts)
    if config.kshift >= nkpts:
        raise ValueError(
            f"pretrain.tda.kshift={config.kshift} is outside nkpts={nkpts}"
        )
    kconserv = np.asarray(get_kconserv_ria(mean_field.cell, mean_field.kpts))[
        config.kshift
    ]
    same_k = np.array_equal(kconserv, np.arange(nkpts))
    if config.require_same_k and not same_k:
        raise ValueError(
            "PBC TDA subspace pretraining v1 requires same-k/q=0 promotions."
        )

    n_roots = n_requested + config.oversample
    td = pbc_tda_kuhf.TDA(mean_field, kshift_lst=[config.kshift])
    td.nstates = n_roots
    td.kernel()
    converged = np.asarray(td.converged[0], dtype=bool)
    energies = np.asarray(td.e[0])
    roots = td.xy[0]
    logger.info(
        "PBC TDA: hf=%s, nkpts=%s, requested_states=%s, requested_roots=%s, "
        "returned_roots=%s, converged_roots=%s, kshift=%s, same_k=%s",
        type(mean_field).__name__,
        nkpts,
        n_requested + 1,
        n_roots,
        len(roots),
        int(np.sum(converged)),
        config.kshift,
        same_k,
    )
    candidates_by_root = []
    for root_index, (energy, root_xy) in enumerate(zip(energies, roots)):
        is_converged = root_index < len(converged) and bool(converged[root_index])
        if config.require_converged and not is_converged:
            logger.info("TDA root=%s skipped: not converged", root_index)
            continue
        amplitudes, _ = root_xy
        candidates_by_root.append(
            _root_candidates(
                root_index=root_index,
                excitation_energy=float(energy),
                amplitudes=amplitudes,
                mo_occ=mean_field.mo_occ,
                kconserv=kconserv,
            )
        )
    return select_unique_promotions(
        candidates_by_root,
        n_requested,
        candidate_topk=config.candidate_topk,
        min_selected_weight=config.min_selected_weight,
    )


class PeriodicStateOrbitalReference:
    """Lazy ground plus TDA-selected periodic single-determinant references."""

    def __init__(
        self,
        scf: PeriodicSCF,
        n_states: int,
        config: PBCTDAReferenceConfig,
    ):
        if n_states < 1:
            raise ValueError("n_states must be positive")
        self.scf = scf
        self.n_states = n_states
        self.config = config
        self.promotions: tuple[PBCSinglePromotion, ...] = ()
        self._state_coeffs: tuple[list[jax.Array], list[jax.Array]] | None = None

    def prepare(self) -> None:
        """Run TDA after SCF and build state-stacked occupied coefficients."""
        if self._state_coeffs is not None:
            return
        ground = self.scf.get_occupied_mo_coeffs()
        promotions = run_pbc_tda_promotions(
            self.scf.mean_field, self.n_states - 1, self.config
        )
        state_coeffs = []
        for state_index in range(self.n_states):
            coeffs = (
                [np.array(value, copy=True) for value in ground[0]],
                [np.array(value, copy=True) for value in ground[1]],
            )
            if state_index:
                promotion = promotions[state_index - 1]
                spin_index = 0 if promotion.spin == "alpha" else 1
                coeffs[spin_index][promotion.k_occ][
                    :, promotion.occ_local_index
                ] = np.asarray(
                    self.scf.mean_field.mo_coeff[spin_index][promotion.k_vir]
                )[:, promotion.vir_mo_index]
            state_coeffs.append(coeffs)

        self._state_coeffs = (
            [
                jnp.asarray(np.stack([state[0][k] for state in state_coeffs]))
                for k in range(len(ground[0]))
            ],
            [
                jnp.asarray(np.stack([state[1][k] for state in state_coeffs]))
                for k in range(len(ground[1]))
            ],
        )
        self.promotions = tuple(promotions)
        logger.info("Subspace pretrain reference: state=0, reference=KUHF ground")
        for state_index, promotion in enumerate(promotions, start=1):
            logger.info(
                "Subspace pretrain reference: state=%s, tda_root=%s, omega=%.8g, "
                "spin=%s, k_occ=%s, k_vir=%s, occ_local=%s, vir_local=%s, "
                "occ_mo=%s, vir_mo=%s, weight=%.6g, participation_ratio=%.6g",
                state_index,
                promotion.root_index,
                promotion.excitation_energy,
                promotion.spin,
                promotion.k_occ,
                promotion.k_vir,
                promotion.occ_local_index,
                promotion.vir_local_index,
                promotion.occ_mo_index,
                promotion.vir_mo_index,
                promotion.weight,
                promotion.participation_ratio,
            )
            if promotion.weight < 0.1 or promotion.participation_ratio > 3:
                logger.warning(
                    "TDA root=%s is strongly multi-configurational; its "
                    "single-determinant pretrain target is only a weak proxy.",
                    promotion.root_index,
                )

    def _coeffs_for_state(
        self, state_index: jax.Array
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        if self._state_coeffs is None:
            raise RuntimeError("PeriodicStateOrbitalReference.prepare() was not called.")
        return (
            [value[state_index] for value in self._state_coeffs[0]],
            [value[state_index] for value in self._state_coeffs[1]],
        )

    def eval_orbitals(
        self,
        state_index: jax.Array,
        pos: jnp.ndarray,
        nspins: tuple[int, int],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate one state's occupied orbitals with native PBC machinery."""
        return self.scf.eval_orbitals_from_coeffs(
            pos, nspins, self._coeffs_for_state(state_index)
        )

    def eval_slater(
        self,
        state_index: jax.Array,
        pos: jnp.ndarray,
        nspins: tuple[int, int],
    ) -> jnp.ndarray:
        """Evaluate one state's complex Slater log-amplitude."""
        return self.scf.eval_slater_from_coeffs(
            pos, nspins, self._coeffs_for_state(state_index)
        )
