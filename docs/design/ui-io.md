# UI case format: result data

Nefes exchanges cases with its graphical companion (FNetLibUI) through a single YAML document: the network definition, the canvas layout, and — the subject of this document — an optional `data.datasets` section that embeds solver results for display on the network graph.
The section exists so that a solve performed in Nefes can be inspected in the UI without the UI knowing anything about the model: the file is self-describing, and the UI renders whatever it declares.
This document records the schema and the design rules behind it; the writer lives in `nefes.io.yaml_out` (`dump_case`, `save_case`, `save_solution`) and the reader in `nefes.io.yaml_in` (`load_case`, `load_solution`).

The presentation begins with the binding rule that anchors every dataset to the network, then describes the dataset and metadata schema, proceeds to the animated (frame-carrying) variant, and closes with what each analysis emits and why the Nyquist result is summarized rather than plotted.

## The index-binding rule

A dataset is a named group of *items*, and an item is one numeric series bound to the network purely by position: `values[i]` belongs to the element whose generated `index` attribute is `i`, for the item's declared `target` (`node` or `edge`).
The UI validates that an item supplies exactly one value per element of its target and rejects the dataset otherwise, so a length mismatch is caught at load rather than silently misdrawn.
An important remark here is that this rule is the entire coupling between data and model: no item is identified by element name or type, so the UI needs no knowledge of what any series means.

Nefes mean-flow state lives on edges, so solver fields are emitted as edge items; passing `node_data=True` additionally reduces each non-phase edge field onto the nodes as the mean over each node's incident edges, a display convenience until genuinely nodal state exists.
Phase series (marked `phase` in the writer) are excluded from this reduction, since a plain mean of wrapped phases is meaningless.

## Datasets and self-describing metadata

Each dataset carries an `id`, a display `name`, an `includeInSave` flag, its `items`, and optionally a free-form `description` and an ordered list of `info` entries.
An `info` entry mirrors the display fields of a model parameter — `key`, `label`, `value`, optional `unit` and `description` — so the UI can render it generically as `label : value unit` without hardcoding any key.
This is the mechanism by which analysis-specific scalars (a modal frequency, a growth rate, a stability verdict) reach the user: the *data* names them, the UI merely prints them.

## Animated datasets

A dataset may declare a `frames` axis, upon which it becomes *animated*: a named frame variable with one value per frame, given as

```yaml
frames:
  variable: Phase
  unit: deg
  values: [0.0, 10.0, ..., 350.0]
```

and each per-frame item then holds one row of element values per frame, `values[k][i]` binding frame `k` to the element of index `i`.
A flat (un-nested) item inside an animated dataset remains legal and is read as frame-independent.
The writer validates that every per-frame item has exactly one row per frame value and that its rows have equal length; the UI repeats both checks on load, together with the index-binding rule applied per row.

The frame variable is deliberately self-describing, exactly like the `info` entries: phase, frequency, or any swept parameter are all just a `variable` name, a `unit`, and a value list, so the playback machinery in the UI stays independent of the model.
When a displayed item belongs to an animated dataset, the UI shows a playback control (play/pause/stop, frame stepping, speed, loop) and reports the frame variable's current value alongside the frame index; the colormap range is computed over *all* frames, so colors remain comparable as the animation runs.

Two producers are built into `dump_case`:

1. **Eigenmode phase animation** (`eig_animation=True`): for each selected mode, the instantaneous physical perturbation is swept over one phase cycle, with the per-edge frame values given as $\Re\{\widehat{\psi}_e\, e^{\mathrm{i}\theta_k}\}$, where $\widehat{\psi}_e$ is the mode-shape amplitude at edge $e$ and $\theta_k = 2\pi k / n$ is the phase of frame $k$ of $n$ (`eig_frames`, default 36).
   The cycle endpoint is excluded so looped playback is seamless, and the amplitude is relative — the eigenvector's arbitrary normalization — matching the static per-mode snapshot datasets.
2. **Forced-response frequency sweep** (`forced_sweep=True`): the whole solved frequency grid of a forced response folds into a single dataset with frequency (Hz) as the frame variable, one frame per frequency, carrying the same magnitude/phase items as the per-frequency snapshots.
   This is not an animation in the physical sense; the frame machinery simply lets the user scrub through a sweep in one container instead of loading one dataset per frequency.

## What each analysis emits

1. **Mean flow**: one edge item per selected field (mass flow, pressures, temperature, Mach number, ...), plus a companion chemistry dataset (mixture fractions, burnt marker, species mole fractions) when the network transports a composition.
2. **Forced response**: per-frequency snapshot datasets by default, or the animated sweep described above; each carries magnitude and phase of the requested perturbation quantities.
3. **Eigenmodes**: one dataset per mode with the per-edge mode-shape magnitude and phase, and the modal scalars (frequency, growth rate, damping ratio, unstable flag, residual) as `info` metadata; optionally the phase animation.
4. **Nyquist stability** (`nyquist=...`): an items-free dataset carrying the verdict as metadata — stable flag, unstable-mode count, stability margin $\min_\omega |D|$, swept band, source rank, and the onset frequencies where the locus skims the critical point.

The Nyquist case deserves the closing emphasis: the locus $L(\omega)$ and the stability determinant $D(\omega)$ are scalar functions of frequency, not element-bound fields, so they have no values with which to color the network and cannot be expressed as items under the index-binding rule.
The summary dataset therefore records the *conclusions* of the analysis in the case file; the locus itself remains a plotting concern of the Python session that produced it.
