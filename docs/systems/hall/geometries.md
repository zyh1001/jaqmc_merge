# Hall geometries

`jaqmc hall train` and `jaqmc hall evaluate` select the geometry through
`system.module`. Omitting it retains the spherical system and its existing
configuration. Each geometry uses its own wavefunction, kinetic energy,
potential energy and sampling proposal.

```bash
# Disk training
jaqmc hall train system.module=jaqmc.app.hall.config.disk \
  system.nspins='[6,0]' system.flux=18 system.interaction_strength=1 \
  workflow.save_path=runs/disk

# Torus training with fractional electron coordinates
jaqmc hall train system.module=jaqmc.app.hall.config:HallTorusConfig \
  system.nspins='[2,0]' system.flux=6 system.tau='[0.0,1.0]' \
  workflow.save_path=runs/torus
```

These commands run numerical calculations and belong in compute-node jobs.
The integration-test submission instructions are in the repository's
`INTEGRATION.md`.

## Coordinates and energy conventions

| Geometry | Electron coordinates | Hamiltonian |
| --- | --- | --- |
| Sphere | Angles $(\theta,\phi)$ | Upstream spherical magnetic kinetic energy and chord-distance Coulomb energy |
| Torus | Fractional coordinates $(u,v)$ | Symmetric-gauge magnetic kinetic energy and two-dimensional Ewald Coulomb energy |
| Disk | Cartesian coordinates $(x,y)$ | Planar magnetic kinetic energy, electron repulsion and the potential of a positive background disk |

Torus positions obey $z=L_1(u+\tau v)$, with
$L_1^2\operatorname{Im}\tau=2\pi N_\phi$. `system.flux` specifies $N_\phi$,
and `system.tau` is serialized as `[real, imaginary]`. The Ewald energy
includes the neutralizing background and periodic self-image contribution.
`wf.module=jaqmc.app.hall.wavefunction.torus.mhpo` selects the default ansatz.
The alternate `torus.sigma_mhpo` module retains its original restriction to
rectangular tori and its deprecation warning. Its validated case is the square
torus. It does not add support for arbitrary complex `tau`.

Disk positions use the convention $z=x-iy$. The background radius defaults
to $\sqrt{2\,\mathtt{flux}}$, and `system.d` is its vertical separation from
the electrons. This radius sets the background charge distribution and the
initial sample region. Sampling has no hard radial boundary. The reported
disk potential omits the constant background self-energy. Its original
integration and interpolation settings are retained.

The physical interaction scaling is `system.interaction_strength`.
Fitted symbolic disk states retain their separate `wf.kappa` parameter;
both settings must describe the intended physical calculation.

## Disk observables

The existing `estimators.enabled` flags select `density`, `pair_correlation`,
`one_rdm`, `structure_factor`, `edge_green`, and `wavefunction_scan`.
`short_range` additionally collects distance-resolved energy sums. It requires
energy and angular momentum, and its `kappa` must equal
`system.interaction_strength`.

```bash
jaqmc hall evaluate system.module=jaqmc.app.hall.config.disk \
  workflow.source_path=runs/disk workflow.save_path=runs/disk-observables \
  estimators.enabled.structure_factor=true estimators.enabled.short_range=true
```

`edge_green` uses a separate evaluation with the first electron pinned at
the configured edge radius. Other equilibrium observables should be evaluated
in an independent run with the ordinary sampler. Empty density bins in the
disk pair correlation remain undefined and are reported as `NaN`.

## Wavefunction overlap

Disk overlap uses the original symmetric sampling density
$\rho\propto|\psi|\,|\psi_{\rm ref}|$ through a dedicated command:

```bash
jaqmc hall overlap workflow.source_path=runs/disk-current \
  workflow.reference_path=runs/disk-reference workflow.save_path=runs/disk-overlap \
  system.nspins='[6,0]' system.flux=18
```

Both checkpoints must use the configured wavefunction architecture and parameter
structure. Omitting `workflow.reference_path` performs a self-overlap.
Parameter-free states may also omit `workflow.source_path`. The digest is
`overlap_digest.npz`, retaining `cos_gamma_modulus`, `f_phase`, `cosine`, and
`fubini_distance`.

Torus checkpoint comparisons retain their importance-sampling estimator:

```bash
jaqmc hall evaluate system.module=jaqmc.app.hall.config.torus \
  workflow.source_path=runs/torus workflow.save_path=runs/torus-overlap \
  estimators.enabled.fubini=true
```

It compares the training checkpoints in the source directory against the last
saved step by default. `estimators.fubini.checkpoint_path` and
`estimators.fubini.reference_step` select another directory and reference step.
The result uses the original `fubini:*` keys.

## Checkpoints and validation

The upstream checkpoint format and Sphere parameter trees are retained. Disk
parameter names and envelope modes are retained, including `amplitude_imag`
and `m_learnable2_old`. A checkpoint created before an extra parameter existed
still needs an explicit parameter conversion. No parameters are silently filled.

Histogram and wavefunction-scan evaluation states are adapted to the current
upstream device-state layout. Start fresh evaluation output directories when
using a migrated training checkpoint. Old accumulated evaluation states are
not converted.

Source comparisons load saved original modules under the current upstream
framework and compare identical coordinates and parameters. They cover the
migrated mathematical implementations, derivatives and local energies.
They do not establish equivalence between two historical training environments.
All numerical tests remain pending until submitted on compute nodes.
