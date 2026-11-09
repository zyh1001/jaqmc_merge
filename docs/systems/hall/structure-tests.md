# Geometry interfaces and short training checks

## Configuration and component selection

Hall configuration follows the `Usercyk/jaqmc` `hall/torus` organization at
commit `8c418a4c4835c1e185e5bd4d3afcb46e168a97ba`. A single
`jaqmc.app.hall.config` module exports a common `HallGeometryConfig` and the
three geometry configurations. Each geometry keeps its own wavefunction and
observable modules. The merged Sphere implementation supplies the current
spinor interface, network options and angular-momentum features.

| Geometry | `system.module` | Default wavefunction class |
| --- | --- | --- |
| Sphere | `jaqmc.app.hall.config:HallSphereConfig` | `wavefunction.sphere.mhpo:SphereMHPO` |
| Disk | `jaqmc.app.hall.config:HallDiskConfig` | `wavefunction.disk.mhpo:DiskMHPO` |
| Torus | `jaqmc.app.hall.config:HallTorusConfig` | `wavefunction.torus.mhpo:TorusMHPO` |

Wavefunction paths in the table are relative to `jaqmc.app.hall`.
The geometry-specific `mhpo` module remains a shorthand for its public class.
Omitting `system.module` selects Sphere. `HallConfig` and
`HallSphericalConfig` are aliases of `HallSphereConfig`.

Sphere observables export `SphereOneRDM`, `SpherePairCorrelation` and
`SpherePenalizedLoss`. The same naming pattern applies to Disk and Torus
observables. Sphere's old class names and wavefunction/observable module
paths re-export the same implementations. The spinor adapter remains at
`wavefunction.base:HallWavefunction` and is also exported by `sphere.base`.

The configuration subpackages have been consolidated into `config.py`.
Before using an older configuration, replace its system module as follows:

| Previous system module | Current system module |
| --- | --- |
| `jaqmc.app.hall.config.disk` or `config.disk:HallDiskConfig` | `jaqmc.app.hall.config:HallDiskConfig` |
| `jaqmc.app.hall.config.torus` or `config.torus:HallTorusConfig` | `jaqmc.app.hall.config:HallTorusConfig` |
| `jaqmc.app.hall.config.spherical` or `config.sphere` | `jaqmc.app.hall.config:HallSphereConfig` |
| `jaqmc.app.hall.config:HallConfig` | Still supported; new configurations use `HallSphereConfig` |

`config.disk` and the other removed submodules are no longer importable.
Historical configurations and checkpoints are kept intact. An existing YAML
configuration can select the current system with a trailing CLI override:

```bash
jaqmc hall train --yml path/to/train_config.yaml \
  system.module=jaqmc.app.hall.config:HallDiskConfig
```

This command executes training and belongs in a compute-node job.
Checkpoint parameter names and shapes are preserved. Existing analysis
readers normalize their already-supported historical module spelling in
memory and leave the original configuration files untouched.

## Four small training cases

All cases use two spin-up electrons, `interaction_strength=1`, seed 21,
32 walkers and 100 training iterations. The MHPO network has one determinant,
one layer, one attention head and eight features per head. Calculations use
single precision, KFAC with learning rate `0.05`, ten learning-rate warmup
steps and ten sampling burn-in iterations. The normal background integration
and Ewald precision settings are retained.

| Case | Flux | Additional configuration |
| --- | --- | --- |
| `sphere` | 3 | The Laughlin sequence has `flux=3*(N-1)` |
| `disk` | 6 | `num_orbitals=6`, using the existing `flux=3*N` background convention |
| `torus-square` | 6 | `tau=[0,1]` |
| `torus-hex` | 6 | `tau=[0.5,0.8660254037844386]` |

The Torus cases both use `TorusMHPO`. `TorusSigmaMHPO` retains its rectangular
restriction. These short runs check the calculation workflow. They do not
establish convergence to a Laughlin state.

## Submission commands

The shell script defaults to one V100 GPU, one task, three CPU cores and
one hour. It uses this checkout's virtual environment and forwards terminal
arguments. Run the following commands from the checkout root:

```bash
sbatch scripts/integration_tests/hall_structure.sh --mode dry-run
sbatch scripts/integration_tests/hall_structure.sh --mode train
sbatch scripts/integration_tests/hall_structure.sh --mode all
```

`dry-run` calls the real `jaqmc hall train --dry-run` command for each case.
`train` runs training and a two-iteration checkpoint evaluation. `all`
performs both stages. Configuration resolution can initialize the JAX
backend, so all three modes run through the Slurm script on compute nodes.

Select one geometry or override ordinary configuration fields:

```bash
sbatch scripts/integration_tests/hall_structure.sh \
  --mode all --geometry torus-hex workflow.batch_size=64
```

Available geometry choices are `all`, `sphere`, `disk`, `torus-square` and
`torus-hex`. Results default to
`integration_test_results/hall-structure-JOB_ID/`. `--output` selects a new
result directory. The script manages the training, restore and evaluation
paths inside this directory. Other trailing `key=value` settings override
the training configuration and are recorded in the result manifest.

Checkpoints are saved every 50 iterations without a time interval. If the
iteration count is reduced, also lower `train.run.save_step_interval` so
that at least two checkpoints are available to check parameter updates.

## Results and acceptance

Each case retains the input YAML, resolved training configuration, CLI logs,
training statistics, checkpoints and evaluation outputs. The top-level
`summary.json` records the source revision, current Hall and KFAC source hashes,
integration-source versions, environment and each stage's status. The
submission log is `hall-structure-JOB_ID.out` in the checkout root.

A training case passes only if all expected steps are recorded, energy
statistics and saved numerical state are finite, the KFAC update count
matches the configured iteration count, and parameters changed between saved
checkpoints. The final checkpoint uses zero-based step 99 for 100 iterations.
Evaluation must produce finite energy statistics and preserve the restored
parameters exactly. A required failure gives the job a nonzero exit code.
Stages that were not selected or could not start are recorded as `not-run`.

The public-interface regressions are in `tests/app/hall/structure_test.py`
and are included in the existing integration script's Sphere suite:

```bash
sbatch scripts/integration_tests/run.sh sphere
```

The existing CLI dry-run tests also cover all four configurations.
This change was checked statically only. Configuration dry-runs, training,
evaluation and numerical regressions are pending user submission.
