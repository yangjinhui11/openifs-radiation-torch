# LW flux_down orientation bugfix (2026-07-28)

## Symptom
`rrtm_rrtm_140gp(...)` returned `flux_down[:,-1]` (surface) ≈ 0 W/m²,
which is physically impossible — the surface downward longwave flux must be
~300–400 W/m² for a warm lower troposphere. The OLR (`flux_up[:,0]`) was
correct (~311 W/m²), so the upward path was fine; only the downward flux
profile was wrong.

## Root cause
In `openifs_radiation/rrtm_lw/rtrn1a.py`, the output block applied a single
`torch.flip` (`stack_f`) to **both** the upward and downward flux lists:

```python
stack_f = lambda lst: torch.flip(torch.stack(lst, dim=1), dims=[1])
"totuflux": stack_f(tuf_list),   # correct: up-list is surface-first → flip → TOA-first
"totdflux": stack_f(tdf_list),   # WRONG:   down-list is TOA-first → flip → surface-first
```

The upward list is built **surface-first** (bottom-up) by the up-loop
(`jk = 0 → nlev-1`), so flipping it yields the correct TOA-first order
(index 0 = TOA = OLR, index -1 = surface).

The downward list is built **TOA-first** by the down-loop
(`jk = nlev-1 → 0`), so it is *already* in TOA-first order; flipping it
reverses it to surface-first, placing the physical surface value (~393 W/m²)
at index 1 and a spurious zero at index 0/last.

The Fortran reference (`rrtm_rtrn1a_140gp.F90`) confirms the asymmetry:
- Upward:   `TOTUFLUX(:,0)` = surface boundary; loop fills `0..KLEV` upward.
- Downward: loop `JLEV=KLEV,1,-1` fills `TOTDFLUX(:,JLEV-1)`, i.e. indices
            `0..KLEV-1` are TOA→surface; index `KLEV` is left = 0.

## Fix
Apply `torch.flip` **only** to the upward lists; leave the downward lists
un-flipped (they are already TOA-first). Additionally, mirror the near-surface
value (`index nlev-1`) into the surface half-level slot (`index nlev`), which
the Fortran leaves at zero, so that the heating-rate divergence at the
surface layer is physically correct:

```python
stack_flip  = lambda lst: torch.flip(torch.stack(lst, dim=1), dims=[1])  # up
stack_plain = lambda lst: torch.stack(lst, dim=1)                        # down
totdflux = stack_plain(tdf_list)
totdflux[:, -1] = totdflux[:, -2]   # fill surface half-level for heating rate
```

## Validation (ERA5 137-level, clear sky)
| Quantity              | Before (buggy) | After (fixed) | Physical expectation |
|-----------------------|----------------|---------------|----------------------|
| OLR (tropics)         | 311.5 W/m²     | 311.5 W/m²    | ~310 (unchanged)     |
| Surface LW down       | 0.02 W/m²      | 393 W/m²      | ~350–400 (σT⁴×ε_atm) |
| Surface-layer HR      | -0.59 K/day    | -0.56 K/day   | small cooling        |
| Stratosphere HR       | -306 K/day ✗   | +4.3 K/day ✓  | ozone warming        |

**OLR is bit-exact identical** before and after (311.52282726 → 311.52282726),
so the bit-exact verification claim in the paper (which uses only OLR) is
unaffected. The bit-exact OLR/flux_up verification against Fortran remains
valid; only the previously-unused `flux_down` and `heating_rate` outputs
were corrected.

## Files changed
- `openifs_radiation/rrtm_lw/rtrn1a.py` — output block (lines ~582–595)

## Why this was not caught earlier
The LW gradient verification (`lw_verification.pdf`) and all paper LW results
used `flux_up[:,0]` (OLR), which was always correct. The `flux_down` output
was never consumed by a paper figure until the cloudy-sky CRE analysis
(`gen_cloudy_global.py`) needed the surface downward flux.
