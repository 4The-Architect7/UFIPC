import pytest
from ufipc_calculator import calculate_ufipc

# Suite of tests for the UFIPC calculation function to ensure robustness and accuracy.

def test_healthy_range():
    """
    Tests if a typical 'healthy' set of parameters returns a score within the expected high range.
    This acts as a basic sanity check for the formula's output.
    """
    score = calculate_ufipc(EIT=500, SDC=80, MAPI=0.9, NSR=0.95)
    assert 6.0 <= score <= 6.5, f"Healthy score {score:.2f} is out of the expected range [6.0, 6.5]"

def test_mcs_range():
    """
    Tests if a typical 'Minimally Conscious State' (MCS) set of parameters returns a score
    within the expected low range.
    """
    score = calculate_ufipc(EIT=50, SDC=15, MAPI=0.2, NSR=0.3)
    assert 0.1 <= score <= 0.5, f"MCS score {score:.2f} is out of the expected range [0.1, 0.5]"

def test_vegetative_state_range():
    """
    Tests if parameters indicative of a vegetative state (very low activity) result
    in a score near zero.
    """
    score = calculate_ufipc(EIT=10, SDC=5, MAPI=0.05, NSR=0.1)
    assert 0.0 <= score <= 0.5, f"Vegetative state score {score:.2f} is out of the expected range [0.0, 0.5]"

# --- Parameter Validation Tests ---

def test_invalid_eit_negative():
    """Tests that a negative EIT value raises a ValueError."""
    with pytest.raises(ValueError, match="EIT .* must be a positive number"):
        calculate_ufipc(EIT=-100, SDC=50, MAPI=0.5, NSR=0.5)

def test_invalid_eit_zero():
    """Tests that a zero EIT value raises a ValueError."""
    with pytest.raises(ValueError, match="EIT .* must be a positive number"):
        calculate_ufipc(EIT=0, SDC=50, MAPI=0.5, NSR=0.5)

def test_invalid_sdc():
    """Tests that a negative SDC value raises a ValueError."""
    with pytest.raises(ValueError, match="SDC .* must be a non-negative number"):
        calculate_ufipc(EIT=100, SDC=-1, MAPI=0.5, NSR=0.5)

def test_invalid_mapi_above_one():
    """Tests that a MAPI value greater than 1 raises a ValueError."""
    with pytest.raises(ValueError, match="MAPI .* must be between 0 and 1"):
        calculate_ufipc(EIT=100, SDC=50, MAPI=1.1, NSR=0.5)

def test_invalid_mapi_below_zero():
    """Tests that a MAPI value less than 0 raises a ValueError."""
    with pytest.raises(ValueError, match="MAPI .* must be between 0 and 1"):
        calculate_ufipc(EIT=100, SDC=50, MAPI=-0.1, NSR=0.5)

def test_invalid_nsr():
    """Tests that an NSR value outside the 0-1 range raises a ValueError."""
    with pytest.raises(ValueError, match="NSR .* must be between 0 and 1"):
        calculate_ufipc(EIT=100, SDC=50, MAPI=0.5, NSR=2.0)

# --- Boundary and Edge Case Tests ---

def test_zero_inputs():
    """
    Tests that if key parameters are zero (SDC, MAPI, or NSR), the resulting score is zero,
    as no complexity can be computed.
    """
    assert calculate_ufipc(EIT=500, SDC=0, MAPI=0.9, NSR=0.95) == 0.0
    assert calculate_ufipc(EIT=500, SDC=80, MAPI=0, NSR=0.95) == 0.0
    assert calculate_ufipc(EIT=500, SDC=80, MAPI=0.9, NSR=0) == 0.0

def test_max_score_clipping():
    """
    Tests that an extremely high, non-physiological set of inputs results in a score
    that is correctly clipped to the maximum of 10.0.
    """
    # Using a very low normalization constant to force a high score
    score = calculate_ufipc(EIT=1e6, SDC=200, MAPI=1.0, NSR=1.0, normalization_constant=1.0)
    assert score == 10.0, "Score should be clipped to 10.0 for extreme inputs"

