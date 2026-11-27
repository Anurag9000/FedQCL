# Ten-client VRAM admission profile correction — 2026-11-26

## Evidence

The prior candidate estimate was 2,577 MiB for both five- and ten-client runs. A live per-process VRAM survey sampled 14 active ten-client runs across the V100, L40 and H100 devices: the allocations ranged from 2,600 MiB to 3,310 MiB. This establishes that the five-client estimate is too low for ten-client admission. These are live phase samples, not complete lifetime peaks.

## Changes

- `vram_admission.py`: separate five-client and ten-client profiles. Ten-client candidate peak is 3,310 + 710 = 4,020 MiB; the 710 MiB margin is the observed range across the 14 ten-client samples.
- `dispatch_jobs.py`: apply the corresponding profile to remaining growth reservations for active attempts, using client count from the authoritative job configuration and avoiding double-counting currently visible allocations.
- `fleet_state.py`: add a transactional `reprofile` action and raise active ten-client reservations to 4,215,275,520 bytes without lowering any existing reservation. Fourteen attempts were updated.
- `test_vram_admission.py`: cover the distinct ten-client profile, byte-unit active-reservation upgrade, and preservation of larger historical reservations.

## Verification

- `test_vram_admission.py`: 9 tests passed.
- `py_compile` for the admission, dispatcher and fleet-state scripts passed.
- Live `active_reservations` checks completed for each fleet GPU and returned plausible growth estimates.
- No training job was launched during this correction; the ready queue was empty.

## Scope and limitation

This only changes VRAM scheduling estimates and reservations. It does not change experiment parameters or training behavior. The 4,020 MiB estimate remains provisional until late-task and counterfactual-phase peaks are measured.
