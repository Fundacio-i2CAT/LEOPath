import pickle
import pprint  # For pretty printing dictionaries

# --- Configuration ---
RESULTS_FILENAME = "starlink_many_gs_results.pkl"
TIME_STEP_S = 60  # Must match the simulation step used
OFFSET_S = 0  # Must match the simulation offset used

# --- Load the data ---
try:
    with open(RESULTS_FILENAME, "rb") as f:
        # Loads the list of state dictionaries (or None values)
        all_states = pickle.load(f)
    print(f"Successfully loaded {len(all_states)} states from {RESULTS_FILENAME}")

except FileNotFoundError:
    print(f"Error: File not found - {RESULTS_FILENAME}")
    print("Did the simulation script run successfully and create the file?")
    exit()
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

# --- Inspect the data ---

if not all_states:
    print("The loaded state list is empty.")
    exit()

# Example 1: Check the state at a specific time step (e.g., the first step, index 0)
step_index = 0
time_s = OFFSET_S + step_index * TIME_STEP_S
print(f"\n--- State at Step {step_index} (t = {time_s}s) ---")

state_at_step = all_states[step_index]

if state_at_step is None:
    print("State calculation failed at this step.")
else:
    # Check if 'fstate' key exists
    if "fstate" in state_at_step:
        fstate_at_step = state_at_step["fstate"]
        print("F-State:")
        pprint.pprint(fstate_at_step)
    # if "bandwidth" in state_at_step:
    #     print("\nBandwidth State:")
    #     pprint.pprint(state_at_step["bandwidth"])
    # else:
    #     print("Bandwidth dictionary not found in the state for this step.")
