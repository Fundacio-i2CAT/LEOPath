import pickle
import argparse
import sys
import pprint


def explore_pickle_file(filepath, list_items, print_index, print_all):
    """
    Loads and explores a pickle file based on command-line arguments.

    Args:
        filepath (str): The path to the pickle file.
        list_items (bool): If True, prints the number of items in the loaded data.
        print_index (int): The index of the item to print. If None, this option is ignored.
        print_all (bool): If True, prints all items in the loaded data.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from: {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}", file=sys.stderr)
        sys.exit(1)
    except pickle.UnpicklingError:
        print(
            f"Error: Could not unpickle the file. It might be corrupted or not a pickle file: {filepath}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}", file=sys.stderr)
        sys.exit(1)

    if (
        not isinstance(data, list)
        and not isinstance(data, tuple)
        and (list_items or print_index is not None or print_all)
    ):
        print(
            "Warning: Loaded data is not a list or tuple. Treating it as a single item.",
            file=sys.stderr,
        )
        if list_items:
            print("Number of items: 1")
        if print_index is not None:
            if print_index == 0:
                print(f"\n--- Data (Item at index 0) ---")
                pprint.pprint(data)
            else:
                print(
                    f"Error: Index {print_index} is out of range for a single item (only index 0 is valid).",
                    file=sys.stderr,
                )
        if print_all:
            print(f"\n--- All Data ---")
            pprint.pprint(data)
        return

    if isinstance(data, (list, tuple)):
        num_items = len(data)
        if list_items:
            print(f"Number of top-level items in the file: {num_items}")

        if print_index is not None:
            if 0 <= print_index < num_items:
                print(f"\n--- Item at index {print_index} ---")
                pprint.pprint(data[print_index])
            else:
                print(
                    f"Error: Index {print_index} is out of range. File contains {num_items} items (indices 0 to {num_items - 1}).",
                    file=sys.stderr,
                )

        if print_all:
            if num_items > 20:  # Arbitrary threshold to ask for confirmation
                confirm = input(
                    f"The file contains {num_items} items. Are you sure you want to print all of them? (yes/no): "
                ).lower()
                if confirm != "yes":
                    print("Aborted printing all items.")
                    return
            print(f"\n--- All {num_items} Items ---")
            for i, item in enumerate(data):
                print(f"\n--- Item at index {i} ---")
                pprint.pprint(item)
                if i < num_items - 1:  # Add a separator for readability
                    print("-" * 20)
    elif not list_items and print_index is None and not print_all:
        # If no specific action is requested other than loading, print a summary or type
        print(f"Data loaded. It is of type: {type(data)}")
        print("Use --list, --index, or --all to explore further.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and print data from a Python pickle file.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )
    parser.add_argument("filepath", help="Path to the pickle file to explore.")
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List the number of top-level items in the pickle file (if it's a list/tuple).",
    )
    parser.add_argument(
        "-i", "--index", type=int, metavar="N", help="Print the item at the specified index N."
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Print all items in the pickle file.\n"
        "You will be prompted for confirmation if there are many items.",
    )

    args = parser.parse_args()

    if not args.list and args.index is None and not args.all:
        print(
            "No action specified. Use -l, -i, or -a to explore the file. See --help for more options."
        )
        # Optionally, you could default to printing basic info or the first item.
        # For now, it requires an explicit action.

    explore_pickle_file(args.filepath, args.list, args.index, args.all)
