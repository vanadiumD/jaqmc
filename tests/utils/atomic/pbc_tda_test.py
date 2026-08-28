# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.utils.atomic.pbc_tda import (
    PBCTDAReferenceConfig,
    PBCSinglePromotion,
    PeriodicStateOrbitalReference,
    run_pbc_tda_promotions,
    select_unique_promotions,
)


def _promotion(root, energy, key, weight):
    spin, k_occ, occ_mo, k_vir, vir_mo = key
    return PBCSinglePromotion(
        root_index=root,
        excitation_energy=energy,
        spin=spin,
        k_occ=k_occ,
        k_vir=k_vir,
        occ_local_index=0,
        vir_local_index=vir_mo - 1,
        occ_mo_index=occ_mo,
        vir_mo_index=vir_mo,
        amplitude=weight**0.5,
        weight=weight,
        largest_weight=weight,
        total_weight=1.0,
        participation_ratio=1.0,
    )


def test_unique_promotion_selection_uses_next_candidate_on_duplicate():
    promotion_a = ("alpha", 0, 0, 0, 1)
    promotion_b = ("beta", 0, 0, 0, 1)
    roots = [
        [_promotion(0, 0.2, promotion_a, 0.8)],
        [
            _promotion(1, 0.3, promotion_a, 0.7),
            _promotion(1, 0.3, promotion_b, 0.2),
        ],
    ]

    selected = select_unique_promotions(
        roots, 2, candidate_topk=2, min_selected_weight=1e-4
    )

    assert [item.key for item in selected] == [promotion_a, promotion_b]


def test_v1_config_rejects_nonzero_kshift_and_non_same_k_mode():
    with pytest.raises(NotImplementedError, match="kshift=0"):
        PBCTDAReferenceConfig(kshift=1)
    with pytest.raises(NotImplementedError, match="same-k"):
        PBCTDAReferenceConfig(require_same_k=False)


def test_state_reference_replaces_only_selected_occupied_column(monkeypatch):
    ground_alpha = np.array([[1.0], [2.0]])
    ground_beta = np.array([[3.0], [4.0]])
    virtual_alpha = np.array([10.0, 20.0])
    promotion = _promotion(0, 0.5, ("alpha", 0, 0, 0, 1), 0.9)

    class FakeSCF:
        _mo_coeff = ([jnp.asarray(ground_alpha)], [jnp.asarray(ground_beta)])
        mean_field = SimpleNamespace(
            mo_coeff=(
                [np.column_stack([ground_alpha[:, 0], virtual_alpha])],
                [np.column_stack([ground_beta[:, 0], [30.0, 40.0]])],
            )
        )

        def get_occupied_mo_coeffs(self):
            return self._mo_coeff

        def eval_orbitals_from_coeffs(self, pos, nspins, coeffs):
            del pos, nspins
            return coeffs[0][0], coeffs[1][0]

        def eval_slater_from_coeffs(self, pos, nspins, coeffs):
            del pos, nspins
            return jnp.sum(coeffs[0][0]) + jnp.sum(coeffs[1][0])

    monkeypatch.setattr(
        "jaqmc.utils.atomic.pbc_tda.run_pbc_tda_promotions",
        lambda mean_field, n_requested, config: [promotion],
    )
    reference = PeriodicStateOrbitalReference(
        FakeSCF(), 2, PBCTDAReferenceConfig()
    )
    reference.prepare()

    state0 = reference._coeffs_for_state(jnp.array(0))
    state1 = reference._coeffs_for_state(jnp.array(1))
    np.testing.assert_array_equal(state0[0][0], ground_alpha)
    np.testing.assert_array_equal(state0[1][0], ground_beta)
    np.testing.assert_array_equal(state1[0][0][:, 0], virtual_alpha)
    np.testing.assert_array_equal(state1[1][0], ground_beta)


def test_m1_state_reference_uses_ground_without_running_tda():
    ground_alpha = jnp.array([[1.0], [2.0]])
    ground_beta = jnp.array([[3.0], [4.0]])

    class FakeSCF:
        mean_field = object()

        def get_occupied_mo_coeffs(self):
            return [ground_alpha], [ground_beta]

    reference = PeriodicStateOrbitalReference(
        FakeSCF(), 1, PBCTDAReferenceConfig()
    )
    reference.prepare()

    coeffs = reference._coeffs_for_state(jnp.array(0))
    np.testing.assert_array_equal(coeffs[0][0], ground_alpha)
    np.testing.assert_array_equal(coeffs[1][0], ground_beta)
    assert reference.promotions == ()


def test_periodic_tda_selector_on_tiny_gamma_h2():
    from jaqmc.utils.atomic.atom import Atom
    from jaqmc.utils.atomic.scf import PeriodicSCF

    scf = PeriodicSCF(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [1.4, 0.0, 0.0])],
        nelectrons=(1, 1),
        basis="sto-3g",
        lattice_vectors=np.eye(3) * 8.0,
        kpts=np.zeros((1, 3)),
        restricted=False,
        verbose=0,
    )
    scf.run()
    if not scf.mean_field.converged:
        pytest.skip("Tiny periodic KUHF did not converge on this PySCF backend")

    promotions = run_pbc_tda_promotions(
        scf.mean_field,
        1,
        PBCTDAReferenceConfig(oversample=0, candidate_topk=4),
    )

    assert len(promotions) == 1
    promotion = promotions[0]
    assert promotion.excitation_energy > 0
    assert np.isfinite(promotion.weight)
    assert promotion.spin in ("alpha", "beta")
    assert promotion.k_occ == promotion.k_vir == 0
    assert promotion.vir_mo_index != promotion.occ_mo_index
