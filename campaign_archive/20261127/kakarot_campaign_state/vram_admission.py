"""Small, unit-testable VRAM admission arithmetic for the fleet dispatcher."""


def remaining_growth(expected_peak_bytes, visible_allocation_bytes=None,
                    overrun_margin_bytes=0):
    """Return only memory still expected to be allocated beyond the live snapshot.

    Device-reported free VRAM already excludes every visible allocation. A
    visible process therefore reserves only its predicted growth. A claimed or
    starting process that is not visible yet reserves its complete peak.
    """
    expected = max(0, int(expected_peak_bytes))
    if visible_allocation_bytes is None:
        return expected
    visible = max(0, int(visible_allocation_bytes))
    if visible > expected:
        return max(0, int(overrun_margin_bytes))
    return max(0, expected - visible)


def can_admit(free_bytes, remaining_growth_bytes, candidate_peak_bytes,
              uncertainty_margin_bytes=0):
    """Whether measured free VRAM can cover growth, candidate peak and margin."""
    return (int(free_bytes) >= int(remaining_growth_bytes)
            + int(candidate_peak_bytes) + int(uncertainty_margin_bytes))


PROFILES = {5: (2034, 543), 10: (3310, 710)}


def profile_margin_mib(clients):
    try:
        return PROFILES[int(clients)][1]
    except KeyError:
        raise ValueError(f"no measured campaign profile for {clients} clients")


def candidate_peak_mib(clients):
    """Provisional VRAM peak by client count, based on live real-run samples.

    Five-client profile: 2034 MiB observed on H100 plus the previously
    measured 543 MiB cross-device envelope.

    Ten-client profile: 3310 MiB is the largest live allocation sampled across
    V100, L40 and H100 processes; add a 710 MiB margin, the observed range
    across the 14 current ten-client samples, while complete late-task and
    counterfactual phase peaks are still being observed. This is a
    conservative admission estimate, not a claimed lifetime peak. No
    GPU-utilization data is used.
    """
    try:
        observed_mib, margin_mib = PROFILES[int(clients)]
    except KeyError:
        raise ValueError(f"no measured campaign profile for {clients} clients")
    return observed_mib + margin_mib


def effective_peak_bytes(clients, stored_peak_bytes=0):
    """Keep active reservations at least as large as the current client profile."""
    profiled_bytes = candidate_peak_mib(clients) * 1024**2
    return max(int(stored_peak_bytes or 0), profiled_bytes)
