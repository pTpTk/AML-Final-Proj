"""Build Elo and five-game histories from the corrected chronological matches."""

from src.features import load_match_history, TRAINING_PATH


def main():
    table = load_match_history()
    table.to_csv(TRAINING_PATH, index=False)
    print(f"Saved {len(table):,} matches and {table.shape[1]} columns to {TRAINING_PATH}.")


if __name__ == "__main__":
    main()
