import numpy as np
import logging

# Configure basic logging for the module
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ufipc(
    EIT: float,
    SDC: float,
    MAPI: float,
    NSR: float,
    normalization_constant: float = 25.0
) -> float:
    """
    Calculates the Universal Framework for Information Processing Complexity (UFIPC) score.

    This function takes four key biophysical parameters and computes a raw score based on
    a specified formula. The raw score is then normalized to a 0-10 scale to provide a
    standardized measure of cognitive complexity.

    Args:
        EIT (float): Information-Theoretic Energy (bits/Joule). Must be a positive value.
        SDC (float): Signal Discrimination Capacity (bits). Must be a non-negative value.
        MAPI (float): Adaptive Plasticity Index. Must be on a 0-1 scale.
        NSR (float): System Responsiveness. Must be on a 0-1 scale.
        normalization_constant (float, optional): The constant used to scale the raw score.
                                                  Defaults to 25.0, which is derived from
                                                  estimated maximum physiological values.

    Returns:
        float: The normalized UFIPC score, clipped to be within the range [0, 10].

    Raises:
        ValueError: If any of the input parameters are outside their valid ranges.
        
    References:
        Contreras, J. (2025). Universal Framework for Information Processing 
        Complexity (UFIPC). Manuscript in preparation.
    """
    logging.debug(f"Input parameters: EIT={EIT}, SDC={SDC}, MAPI={MAPI}, NSR={NSR}")
    
    # --- Input Validation ---
    if not isinstance(EIT, (int, float)) or EIT <= 0:
        raise ValueError("EIT (Information-Theoretic Energy) must be a positive number.")
    if not isinstance(SDC, (int, float)) or SDC < 0:
        raise ValueError("SDC (Signal Discrimination Capacity) must be a non-negative number.")
    if not isinstance(MAPI, (int, float)) or not (0 <= MAPI <= 1):
        raise ValueError("MAPI (Adaptive Plasticity Index) must be between 0 and 1.")
    if not isinstance(NSR, (int, float)) or not (0 <= NSR <= 1):
        raise ValueError("NSR (System Responsiveness) must be between 0 and 1.")

    # --- Calculation ---
    # The core formula combines the parameters to generate a raw complexity score.
    # log10 is used for EIT to handle its potentially wide dynamic range.
    # tanh and sqrt are used to appropriately weight MAPI and NSR.
    raw_score = np.log10(EIT) * SDC * np.tanh(MAPI) * np.sqrt(NSR)

    # --- Normalization ---
    # The raw score is divided by a constant to map it to a standardized 0-10 scale.
    if normalization_constant <= 0:
        raise ValueError("normalization_constant must be positive.")
    normalized_score = raw_score / normalization_constant
    
    logging.info(f"Calculated raw score: {raw_score:.2f}, Normalized UFIPC score: {normalized_score:.2f}")

    # --- Output ---
    # The final score is clipped to ensure it strictly falls within the [0, 10] range.
    final_score = np.clip(normalized_score, 0, 10)
    logging.debug(f"Final clipped score: {final_score:.2f}")
    return final_score

# --- Example Usage ---
if __name__ == '__main__':
    # Example parameters for a healthy, alert individual
    eit_healthy = 500.0  # bits/Joule
    sdc_healthy = 80.0   # bits
    mapi_healthy = 0.9   # High plasticity
    nsr_healthy = 0.95   # High responsiveness

    # Example parameters for a patient in a minimally conscious state
    eit_mcs = 50.0
    sdc_mcs = 15.0
    mapi_mcs = 0.2
    nsr_mcs = 0.3

    try:
        print("\n--- Calculating Healthy Score ---")
        ufipc_score_healthy = calculate_ufipc(eit_healthy, sdc_healthy, mapi_healthy, nsr_healthy)
        print(f"Healthy Individual's UFIPC Score: {ufipc_score_healthy:.2f}")

        print("\n--- Calculating MCS Score ---")
        ufipc_score_mcs = calculate_ufipc(eit_mcs, sdc_mcs, mapi_mcs, nsr_mcs)
        print(f"Minimally Conscious State UFIPC Score: {ufipc_score_mcs:.2f}")

        print("\n--- Testing Invalid Input ---")
        # Example of invalid input to demonstrate error handling
        calculate_ufipc(-100, 50, 0.5, 0.5)

    except ValueError as e:
        # Using logging to capture the error as well
        logging.error(f"Error during calculation: {e}")
        print(f"\nCaught expected error: {e}")

