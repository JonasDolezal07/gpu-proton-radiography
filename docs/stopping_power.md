# Stopping power and energy loss

prad models continuous energy loss via the relativistic Bethe-Bloch formula. When a `[density]`
block is present in a deck, protons deposit energy into the traversed medium at every Boris step.
Absent a density grid the simulation is purely electromagnetic — energy is conserved exactly.

---

## Physics

### Bethe-Bloch mass stopping power

The energy loss per unit areal density for a proton traversing a medium is:

$$
-\frac{dE}{d(\rho x)} = \frac{K z^2 Z/A}{\beta^2}
\left[ \frac{1}{2} \ln\!\frac{2 m_e c^2 \beta^2 \gamma^2 T_\text{max}}{I^2} - \beta^2 \right]
\quad \text{[MeV cm}^2\text{/g]}
$$

where:

| Symbol | Value / meaning |
|---|---|
| $K$ | $4\pi N_A r_e^2 m_e c^2 = 0.307075$ MeV cm²/g |
| $z$ | projectile charge number (1 for protons) |
| $Z/A$ | target effective atomic number / mass ratio |
| $\beta, \gamma$ | relativistic $v/c$ and Lorentz factor |
| $T_\text{max}$ | maximum kinetic energy transferable to a free electron in one collision |
| $I$ | mean excitation energy of the target material |

$T_\text{max}$ for a proton (mass $M_p$) on an electron (mass $m_e$):

$$
T_\text{max} = \frac{2 m_e c^2 \beta^2 \gamma^2}{1 + 2\gamma m_e / M_p + (m_e/M_p)^2}
\approx 2 m_e c^2 \beta^2 \gamma^2 \quad (M_p \gg m_e)
$$

The differential energy loss per unit path length then follows from the density $\rho$:

$$
-\frac{dE}{dx} = \rho \cdot \left(-\frac{dE}{d(\rho x)}\right)
\quad \text{[MeV/m when } \rho \text{ is g/cm}^3 \text{ and } x \text{ is cm]}
$$

### Continuous slowing-down approximation (CSDA)

prad uses the **CSDA**: at each simulation step the proton loses the expectation-value energy
for that path length. Stochastic fluctuations around the mean (energy-loss straggling) are
not modelled. The CSDA is standard in deterministic transport codes and is exact in the
limit of many small steps. For the timesteps used in prad (0.1–2 ps, path length ~10–50 µm),
the CSDA is a good approximation for proton energies > 1 MeV.

### Energy update per step

After each Boris push, the shader:

1. Samples the local density $\rho(\mathbf{x})$ from the density texture
2. Looks up the mass stopping power $S(T) = -dE/d(\rho x)$ at the current kinetic energy $T$
3. Computes the path length $\delta x = |\mathbf{v}| \cdot dt$ (physical velocity × timestep)
4. Computes $\delta E = S(T) \cdot \rho \cdot \delta x$ and subtracts from kinetic energy
5. Rescales $|\mathbf{u}|$ (the specific relativistic momentum $\mathbf{u} = \gamma \mathbf{v}$) to
   match the new kinetic energy — the momentum direction is unchanged

The minimum enforced kinetic energy is 1 keV. Protons that fall below this threshold are not
removed from the simulation but are effectively stopped (velocity → near-zero, they eventually
exit the domain).

---

## GPU pipeline

The stopping power table is precomputed on the CPU at load time and uploaded as a 256-entry
`float32` array in a storage buffer (binding 6). Entries are log-spaced from 0.1 MeV to 1000 MeV:

```
index i  →  KE = exp(log(0.1) + i/255 × (log(1000) − log(0.1)))   [MeV]
```

The density grid is a scalar `R32_SFLOAT` 3D texture (binding 5), sampled with trilinear
interpolation. Positions outside the declared grid bounds return density 0 (vacuum) — the density
grid does not need to cover the full simulation volume.

Both bindings default to vacuum (zero density, zero table) when no `[density]` block is present,
so all existing simulations are unaffected.

---

## Configuration

```toml
[density]
path     = "data/plasma.dens"   # path to .dens file, relative to deck
material = "water"              # built-in preset (see table below)
```

For a custom material, set `material = "custom"` and supply the parameters explicitly:

```toml
[density]
path     = "data/plasma.dens"
material = "custom"
z_over_a = 0.4935   # effective Z/A  (e.g. CH plastic: 0.5702)
i_ev     = 78.0     # mean excitation energy I [eV]
```

The `[density]` block is entirely optional. Without it the simulation is purely electromagnetic.

### Opaque absorber mode

Instead of continuous energy loss, the density grid can act as a **binary absorber**: any
particle that enters a voxel above a density threshold is immediately removed from the simulation
(recorded as absorbed, not as a detector hit).

```toml
[density]
path = "data/target.dens"
mode = "opaque"                  # "csda" (default) or "opaque"
opaque_threshold_g_cm3 = 0.1    # density threshold [g/cm³]; default 0.1
```

In opaque mode the `material`, `z_over_a`, and `i_ev` keys are ignored — no stopping power
table is needed. The threshold applies to the sampled voxel density at each Boris step.

This mode is useful for modelling solid targets, mesh grids, or any object that is
geometrically opaque to protons without caring about the exact energy-loss physics.
Validation test 25 demonstrates a fiducial mesh radiograph using this mode.

---

## Built-in materials

| Name | Z/A | I (eV) | Typical use |
|---|---|---|---|
| `water` | 0.5551 | 75.0 | Tissue, biological, standard reference |
| `plastic` | 0.5702 | 57.4 | CH₂ / polypropylene, RCF active layers |
| `beryllium` | 0.4439 | 63.7 | Target foils, cryostat windows |
| `aluminum` | 0.4818 | 166.0 | Filter stacks, target holders |
| `hydrogen` | 0.9922 | 19.2 | Cryogenic H targets |

Z/A and I values are taken from the NIST PSTAR database and Sternheimer et al. (1984).

---

## Density file format (`.dens`)

A `.dens` file holds a scalar 3D density grid. The format mirrors `.bfld` but stores a single
`float32` value per voxel rather than a vector.

### Header (64 bytes, little-endian)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | bytes | Magic: `DENS` |
| 4 | 4 | u32 | Version: 1 |
| 8 | 4 | u32 | nx |
| 12 | 4 | u32 | ny |
| 16 | 4 | u32 | nz |
| 20 | 4 | f32 | x\_min (metres) |
| 24 | 4 | f32 | x\_max |
| 28 | 4 | f32 | y\_min |
| 32 | 4 | f32 | y\_max |
| 36 | 4 | f32 | z\_min |
| 40 | 4 | f32 | z\_max |
| 44 | 20 | — | Padding (zeros) |

### Data

Immediately after the header: `nx × ny × nz` values, `f32` little-endian, C-contiguous
(x outermost, z innermost). Units: **g/cm³**.

### Python helper

```python
import struct
import numpy as np
from pathlib import Path

def write_dens(path, rho, bounds):
    """
    rho:    numpy array (nx, ny, nz), float32, g/cm³
    bounds: (x_min, x_max, y_min, y_max, z_min, z_max) in metres
    """
    nx, ny, nz = rho.shape
    with open(path, "wb") as f:
        f.write(b"DENS")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", *bounds))
        f.write(b"\x00" * 20)           # padding
        f.write(rho.astype("<f4").tobytes())

def read_dens(path):
    """Returns (header dict, rho array (nx,ny,nz) float32)."""
    raw = Path(path).read_bytes()
    assert raw[:4] == b"DENS", "not a .dens file"
    _, nx, ny, nz = struct.unpack_from("<4I", raw, 0)[0], *struct.unpack_from("<3I", raw, 8)
    bounds = struct.unpack_from("<6f", raw, 20)
    rho = np.frombuffer(raw, dtype="<f4", offset=64).reshape(nx, ny, nz).copy()
    return {"nx": nx, "ny": ny, "nz": nz,
            "x_min": bounds[0], "x_max": bounds[1],
            "y_min": bounds[2], "y_max": bounds[3],
            "z_min": bounds[4], "z_max": bounds[5]}, rho
```

### Example: uniform water slab

```python
import numpy as np

# 5 mm water slab centred on the beam path, covering y/z ± 30 mm
rho = np.ones((8, 32, 32), dtype=np.float32)   # ρ = 1 g/cm³ everywhere

write_dens(
    "data/water_slab.dens",
    rho,
    bounds=(0.0, 0.005, -0.03, 0.03, -0.03, 0.03),
)
```

### Example: hollow plasma shell

```python
# Cylindrical shell of density in y-z, uniform along x
nx, ny, nz = 4, 64, 64
rho = np.zeros((nx, ny, nz), dtype=np.float32)

y = np.linspace(-0.02, 0.02, ny)
z = np.linspace(-0.02, 0.02, nz)
Y, Z = np.meshgrid(y, z, indexing="ij")
r = np.sqrt(Y**2 + Z**2)

# Shell between 5 mm and 10 mm radius, density 0.1 g/cm³
shell = (r > 0.005) & (r < 0.010)
for ix in range(nx):
    rho[ix][shell] = 0.1

write_dens(
    "data/plasma_shell.dens",
    rho,
    bounds=(-0.01, 0.01, -0.02, 0.02, -0.02, 0.02),
)
```

---

## Validation

### CSDA range agreement with NIST PSTAR

The mass stopping power table is validated against the NIST PSTAR proton range database.
The CSDA range is computed by numerically integrating $\int_0^{T_0} dT / S(T)$ with the
same Bethe-Bloch formula used by prad:

| Energy (MeV) | prad CSDA range | NIST PSTAR range | Deviation |
|---|---|---|---|
| 14.7 | 2.43 mm | 2.40 mm | +1.3% |
| 10.0 | 1.24 mm | 1.23 mm | +0.8% |
| 5.0 | 0.37 mm | 0.37 mm | < 1% |

The ~1% offset at higher energies is expected: this implementation omits the density effect
correction ($\delta/2$), which lowers stopping power slightly at higher $\beta\gamma$.

### test16 — GPU vs analytic integral

A pencil beam of 14.7 MeV protons passes through a uniform 1 mm water slab ($\rho = 1$ g/cm³).
The mean exit kinetic energy is compared to a numerically-integrated Bethe-Bloch CSDA prediction:

```
Simulated exit KE:  10.868 MeV  (loss 3.832 MeV)
Analytic prediction: 10.944 MeV  (loss 3.756 MeV)
Relative error:      0.7%   [tolerance: 5%]
```

The small residual comes from discretisation of the table and the finite step size (0.5 ps → ~26 µm path per step). Smaller `dt_ps` values converge the result toward the analytic integral.

---

## Interpretation

With a density grid configured, the `hits.bin` file contains each particle's **exit kinetic
energy** after traversing the medium, not the input energy. This enables:

**Density contrast imaging.** Denser regions produce larger energy shifts. The energy map
$\langle T(\mathbf{r}) \rangle$ encodes path-integrated density, complementary to the
deflection map from EM fields.

**Range estimation.** Set up a thickness scan (sweep `scale_B = 0.0` with varying slab
thickness) to locate the Bragg peak position for your geometry and beam energy.

**Combined EM + stopping.** When both a magnetic field and a density grid are present,
EM deflection and energy loss operate simultaneously — the Boris push and CSDA energy
update are applied independently each step, which is valid when both perturbations are small
per step.

---

## Limitations

**CSDA only.** Energy-loss straggling (Bohr/Landau-Vavilov fluctuations) is not modelled.
The CSDA gives the mean energy loss; the actual spread around the mean grows as
$\sigma_E \propto \sqrt{\rho x}$ and becomes significant for thin layers
($\rho x \lesssim$ 1 mg/cm²) or near the Bragg peak.

**No nuclear stopping.** The Bethe-Bloch formula captures electronic stopping only. Nuclear
stopping (proton-nucleus Coulomb scattering, relevant below ~100 keV) is not included.
This has negligible impact for proton energies > 1 MeV.

**No density-effect correction.** The $\delta/2$ density-effect term (relevant above ~100 MeV)
is omitted. This causes ~1–2% overestimation of stopping power at 100 MeV and is negligible
below 50 MeV.

**No charge-exchange or nuclear reactions.** The proton carries charge $+e$ throughout; no
electron capture, stripping, or inelastic nuclear interactions are simulated.

**Single material per deck.** The stopping power table is computed for one material. Composite
targets (e.g. a water layer over an aluminium substrate) are not supported in the current
implementation — use a single equivalent material or run separate simulations.

**Static density.** Like the EM field, the density grid is loaded once and does not evolve.
There is no feedback between the proton beam and the plasma density.
