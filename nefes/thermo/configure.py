"""Parse-time thermo configuration (pure Python).

Builds the immutable ``(model_id, tf, ti)`` bundle and a manifest describing the
transported composition (empty for a perfect gas).  The bundle has a fixed dtype
/ contiguity signature across all models so a single compiled ``thermo_update``
serves every backend.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from nefes.thermo.constants import P_REF, R_UNIVERSAL

from .api import EQ_KERNEL, PERFECT_GAS


@dataclass(frozen=True)
class ThermoConfig:
    """Immutable thermo bundle passed read-only through the kernels.

    Attributes
    ----------
    model_id : int
        Integer selecting the gas model (``PERFECT_GAS``, ``EQ_KERNEL``, ...).
    tf : numpy.ndarray
        Packed float64 data block; layout is model-specific (``[cp, R, W]`` for a
        perfect gas, the flat NASA/element arrays for equilibrium).
    ti : numpy.ndarray
        Packed int64 sizes/indices companion to ``tf`` (empty for a perfect gas).
    element_names : list of str
        Labels of the transported band-1 scalars (feed streams / passive scalars).
    species_names : list of str
        Species carried by the species set (empty for a perfect gas).
    species_set : object
        The ``SpeciesSet``, kept parse-time only (never packed or
        compiled) so element/source builders can resolve species-named compositions
        to feed streams and the ``Tt -> h_t`` datum.
    t_init, t_init_frozen : float
        Initial temperature guesses [K] for a deferred equilibrium config whose feed
        streams are discovered and packed at build time.
    """

    model_id: int  # gas-model selector
    tf: np.ndarray  # float64[::1] packed real data
    ti: np.ndarray  # int64[::1] packed sizes/indices
    element_names: List[str] = field(default_factory=list)  # transported-scalar labels
    species_names: List[str] = field(default_factory=list)  # species names carried by the set
    species_set: object = None  # SpeciesSet (parse-time only)
    t_init: float = 3000.0  # equilibrium temperature guess [K]
    t_init_frozen: float = 300.0  # frozen temperature guess [K]
    # declared feed-stream mass fractions (K, n_species) when the streams were named up front
    # (equilibrium(streams=...)); None defers stream discovery to build time (auto-merge of feeds)
    stream_Y: object = None
    # slate reducer for a deferred automatic species set (species_set=None), applied at network build
    reducer: str = "equilibrium_sampling"
    # trace mole-fraction cutoff for the reducer (None -> its default) and the candidate count
    # above which reduction runs (None -> AUTO_REDUCE_THRESHOLD); the deferred-slate size dials
    reduce_threshold: float = None
    reduce_above: int = None
    # ceiling on the kept species count (None -> uncapped) and species to keep regardless of
    # abundance; further deferred-slate dials, applied at network build
    max_species: int = None
    must_species: tuple = ()
    # True for equilibrium(species_set=None): the product slate is (re)derived from the network
    # feeds at every build, so adding a feed after a solve expands the slate as expected
    auto_species_set: bool = False

    @property
    def n_elem(self) -> int:
        """Number of transported band-1 scalars (feed streams / passive scalars)."""
        return len(self.element_names)

    @property
    def stream_mode(self) -> str:
        """Feed-stream mode: ``"declared"`` when the streams were named up front
        (``equilibrium(streams=...)``, a fixed closed basis feeds blend over), else
        ``"auto"`` (the streams are the distinct feed compositions, discovered at build)."""
        return "declared" if self.stream_Y is not None else "auto"

    @property
    def n_species(self) -> int:
        """Number of species carried by the species set (zero for a perfect gas)."""
        return len(self.species_names)


def perfect_gas(R: float = 287.0, gamma: float = 1.4) -> ThermoConfig:
    """Calorically-perfect-gas configuration (default: dry air).

    Parameters
    ----------
    R : float, optional
        Specific gas constant [J/(kg*K)] (default 287.0, dry air).
    gamma : float, optional
        Ratio of specific heats (default 1.4).

    Returns
    -------
    ThermoConfig
        Bundle with ``tf = [cp, R, W]`` where ``W = R_u / R`` is the molar mass
        [kg/mol] and ``cp = gamma R / (gamma - 1)``.
    """
    cp = gamma * R / (gamma - 1.0)
    W = R_UNIVERSAL / R  # molar mass [kg/mol]
    tf = np.ascontiguousarray([cp, R, W], dtype=np.float64)
    ti = np.empty(0, dtype=np.int64)
    return ThermoConfig(model_id=PERFECT_GAS, tf=tf, ti=ti)


def perfect_gas_passive_scalars(n_scalars: int, R: float = 287.0, gamma: float = 1.4, names=None) -> ThermoConfig:
    """Perfect gas that also advects ``n_scalars`` passive conserved scalars.

    The thermodynamics are unchanged (the perfect-gas kernels ignore ``Z_el``);
    each scalar simply adds one band-1 unknown and one source-free transport
    equation per edge, exercising the composition-transport framework without
    invoking chemistry.

    Parameters
    ----------
    n_scalars : int
        Number of passive conserved scalars to advect (must be >= 1).
    R : float, optional
        Specific gas constant [J/(kg*K)] (default 287.0).
    gamma : float, optional
        Ratio of specific heats (default 1.4).
    names : sequence of str, optional
        Scalar labels; defaults to ``scalar0, scalar1, ...``.

    Returns
    -------
    ThermoConfig
        Perfect-gas bundle carrying ``n_scalars`` band-1 scalars.
    """
    if n_scalars < 1:
        raise ValueError("n_scalars must be >= 1")
    cfg = perfect_gas(R, gamma)
    names = list(names) if names is not None else [f"scalar{i}" for i in range(n_scalars)]
    if len(names) != n_scalars:
        raise ValueError("len(names) must equal n_scalars")
    return ThermoConfig(model_id=PERFECT_GAS, tf=cfg.tf, ti=cfg.ti, element_names=names)


def equilibrium(
    species_set=None,
    streams=None,
    basis: str = "mole",
    mode: str = None,
    T_init: float = 3000.0,
    T_init_frozen: float = 300.0,
    reducer: str = "equilibrium_sampling",
    reduce_threshold: float = None,
    reduce_above: int = None,
    max_species: int = None,
    must_species=(),
):
    """Reacting-gas config from a ``nefes.thermo.SpeciesSet``.

    The transported composition is the network's **feed-stream mixture fractions**
    ``xi`` -- one conserved band-1 scalar per distinct injected composition (an
    oxidizer, a diluent, a fuel, ...).  Both per-edge closures reconstruct from
    ``xi`` by a forward blend: the unburnt (``EQ_FROZEN``) side gets its species
    moles ``n_feed = xi @ Nfeed`` and the burnt (``EQ_KERNEL``) side gets the
    elemental ``Z = xi @ Zfeed`` for the HP-equilibrium solve.  The base model is
    ``EQ_KERNEL``; mark approach edges ``EQ_FROZEN`` via
    ``build_problem(..., edge_models=...)``.

    Parameters
    ----------
    species_set : nefes.thermo.SpeciesSet or nefes.thermo.Mechanism, optional
        The species data (NASA-7/9 polynomials, element matrix).  The species_set is
        the standalone authority; this packs its canonical 9-term arrays for the
        compiled kernel.  Leave it ``None`` (the default) to defer to the **automatic**
        product slate: the packaged NASA Glenn / CEA data is the database, and the
        species_set is resolved from the network's feed compositions at build time (the
        candidate products buildable from the fed-in elements, reduced to the non-trace
        species).  A deferred species set carries no streams (``mode="auto"`` only); pass an
        explicit ``species_set`` to name a declared-stream basis or to pin the species set.
    streams : dict of {str: spec}, optional
        The **declared** feed streams ``{label: composition}`` (each ``composition`` a named
        species mixture in ``basis`` units).  These are the network's fixed, named transported
        streams: every feed then states its composition **in terms of these streams** (its
        ``composition={label: amount}``), so a premixed inlet keeps its constituent streams
        separate and their ratio (an equivalence ratio) stays a live composition degree of
        freedom -- and a stream may be declared without any dedicated inlet that injects it
        pure.  Selects ``mode="declared"``; also pins the scalar order/labels.
    basis : {"mole", "mass"}
        Units of the declared ``streams`` compositions (the species mixtures; ignored in
        ``"auto"`` mode, where each feed carries its own basis).
    mode : {"auto", "declared"}, optional
        ``"declared"`` -- the transported streams are exactly the declared ``streams`` (a
        fixed, closed basis; feeds name their composition over these streams and a feed that
        names something else is rejected).  ``"auto"`` -- the streams are the distinct feed
        compositions, discovered and auto-merged at build time (each feed carries a raw species
        mixture).  Default: ``"declared"`` when ``streams`` is given, else ``"auto"``.
    T_init, T_init_frozen : float
        Initial temperature guesses for the equilibrium and frozen solves [K].
    reducer : str, optional
        Registry key of the slate reducer used when the deferred automatic species set
        (``species_set=None``) is resolved at build (default ``"equilibrium_sampling"``;
        ``"none"`` keeps every candidate).  Ignored when an explicit ``species_set`` is given.
    reduce_threshold : float, optional
        Trace mole-fraction cutoff for the automatic slate: a candidate is kept when its peak
        equilibrium mole fraction over the feed-mixing range clears it.  Larger keeps fewer
        species, smaller keeps more; ``None`` (default) uses the reducer's own cutoff.  Applies
        only to the deferred automatic set, and only once the slate is large enough to reduce
        (see ``reduce_above``).
    reduce_above : int, optional
        Candidate count above which reduction runs; below it every candidate is kept.  Lower it
        to trim a lean slate, raise it to keep a broad slate whole.  ``None`` (default) uses the
        built-in threshold.  Applies only to the deferred automatic set.
    max_species : int, optional
        Ceiling on the number of species in the deferred automatic slate (``None`` for no cap).
        Species are ranked by peak equilibrium mole fraction and the slate filled to this many,
        after the feed species, the ``must_species``, and one carrier of every fed-in element.
        A ceiling, not a target: it only discards the lowest-ranked non-trace species, never
        pads the slate with trace ones, so a size sweep pairs it with a loose ``reduce_threshold``.
        Not accepted together with ``reducer="none"``.  Applies only to the deferred automatic set.
    must_species : iterable of str, optional
        Species to keep in the deferred automatic slate regardless of abundance (a marker, a
        pollutant that is trace at equilibrium, or a high-temperature condensed product such as
        graphite ``"C(gr)"``).  Each must be in the database, buildable from the fed-in elements,
        and an eligible equilibrium product (a gas or a condensed species whose data reaches
        combustion temperatures); an ion or a feed-only condensed species is rejected.  Applies
        only to the deferred automatic set.

    See Also
    --------
    ThermoConfig.stream_mode : the resolved mode on the returned config.
    nefes.thermo.autoset.auto_product_set : the automatic-slate policy.
    """
    from .edge_state import pack_equilibrium

    if max_species is not None and str(reducer or "equilibrium_sampling") == "none":
        raise ValueError("max_species cannot be combined with reducer='none' (which keeps every candidate)")

    if species_set is None:
        # Deferred automatic species set: the species set is unknown until the network's feeds are
        # seen, so pack a valid header only (P_ref, T_init, T_init_frozen -- what the reference
        # scaling reads) and resolve the real bundle at build (finalize_thermo).
        if streams is not None or mode == "declared":
            raise ValueError(
                "species_set=None uses the automatic product slate (auto mode); pass an explicit "
                "species_set to declare a stream basis (streams=...) or a fixed species set"
            )
        header = np.array([P_REF, T_init, T_init_frozen], dtype=np.float64)
        return ThermoConfig(
            model_id=EQ_KERNEL,
            tf=header,
            ti=np.empty(0, dtype=np.int64),
            element_names=[],
            species_names=[],
            species_set=None,
            t_init=T_init,
            t_init_frozen=T_init_frozen,
            stream_Y=None,
            reducer=reducer,
            reduce_threshold=reduce_threshold,
            reduce_above=reduce_above,
            max_species=max_species,
            must_species=tuple(must_species),
            auto_species_set=True,
        )

    if mode is None:
        mode = "declared" if streams is not None else "auto"
    if mode not in ("auto", "declared"):
        raise ValueError(f"mode must be 'auto' or 'declared'; got {mode!r}")
    if mode == "declared" and streams is None:
        raise ValueError("mode='declared' requires streams={label: composition} to declare the basis")
    if mode == "auto" and streams is not None:
        raise ValueError("streams=... declares a fixed basis (that is mode='declared'); drop streams= for mode='auto'")

    if mode == "auto":
        stream_Y = np.zeros((0, species_set.n_species))
        labels: List[str] = []
        declared_Y = None  # defer discovery to build time
    else:
        from ..chem.composition import species_mass_fractions

        labels = list(streams.keys())
        stream_Y = np.array([species_mass_fractions(species_set, streams[k], basis) for k in labels], dtype=np.float64)
        declared_Y = stream_Y  # the fixed basis every feed states its composition over

    tf, ti = pack_equilibrium(species_set, stream_Y, T_init, T_init_frozen)
    return ThermoConfig(
        model_id=EQ_KERNEL,
        tf=tf,
        ti=ti,
        element_names=labels,
        species_names=[s.name for s in species_set.species],
        species_set=species_set,
        t_init=T_init,
        t_init_frozen=T_init_frozen,
        stream_Y=declared_Y,
    )
