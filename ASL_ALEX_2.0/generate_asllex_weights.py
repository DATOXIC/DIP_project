import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. COLUMNS USED FROM ASL-LEX 2.0
# ============================================================
# We intentionally use only the dominant-hand phonological fields.
# These columns are mapped to the four feature groups of the project.
FEATURE_GROUPS = {
    "A": [
        "Handshape.2.0",
        "SelectedFingers.2.0",
        "Flexion.2.0",
        "Spread.2.0",
        "ThumbPosition.2.0",
        "ThumbContact.2.0",
    ],
    "P": [
        "MajorLocation.2.0",
        "MinorLocation.2.0",
        "SecondMinorLocation.2.0",
    ],
    "JM": [
        "Movement.2.0",
        "RepeatedMovement.2.0",
        "FlexionChange.2.0",
        "SpreadChange.2.0",
    ],
    "N": [
        "UlnarRotation.2.0",
    ],
}

OUTPUT_GROUP_ORDER = ["P", "A", "JM", "N"]


# ============================================================
# 2. LOAD DATA
# ============================================================
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load ASL-LEX CSV. latin1 handles the supplied file safely."""
    return pd.read_csv(csv_path, encoding="latin1", low_memory=False)


def validate_columns(df: pd.DataFrame) -> None:
    """Check that every required column exists before calculation."""
    required = ["EntryID"]
    for columns in FEATURE_GROUPS.values():
        required.extend(columns)

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in signdata.csv:\n"
            + "\n".join(missing)
        )


# ============================================================
# 3. FIND THE TARGET SIGN
# ============================================================
def normalize_sign_name(value: str) -> str:
    """Normalize a sign name for case-insensitive matching."""
    return str(value).strip().lower()


def find_target_row(df: pd.DataFrame, sign_name: str) -> pd.Series:
    """Find one ASL-LEX sign by EntryID."""
    normalized_target = normalize_sign_name(sign_name)
    entry_ids = df["EntryID"].astype(str).map(normalize_sign_name)

    matches = df[entry_ids == normalized_target]

    if matches.empty:
        suggestions = sorted(df["EntryID"].dropna().astype(str).unique())
        similar = [name for name in suggestions if normalized_target in name.lower()]

        message = f"Sign '{sign_name}' was not found in EntryID."
        if similar:
            message += " Similar entries:\n" + "\n".join(similar[:20])
        raise ValueError(message)

    if len(matches) > 1:
        raise ValueError(
            f"Sign '{sign_name}' appears more than once in EntryID. "
            "The script expects one phonological entry per sign."
        )

    return matches.iloc[0]


# ============================================================
# 4. FEATURE COMPARISON
# ============================================================
def value_is_missing(value) -> bool:
    """Return True for NaN/None/blank values."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def group_similarity(target_row: pd.Series, candidate_row: pd.Series, columns: list[str]):
    """
    Compare two signs within one feature group.

    Missing fields are ignored. The score is:
        number of matching known fields / number of comparable fields

    Returns None when no field in the group is available for comparison.
    """
    comparisons = []

    for column in columns:
        target_value = target_row[column]
        candidate_value = candidate_row[column]

        if value_is_missing(target_value) or value_is_missing(candidate_value):
            continue

        target_text = str(target_value).strip().lower()
        candidate_text = str(candidate_value).strip().lower()

        comparisons.append(1.0 if target_text == candidate_text else 0.0)

    if not comparisons:
        return None

    return float(np.mean(comparisons))


def overall_similarity(group_similarities: dict[str, float | None]) -> float:
    """
    Average available group similarities equally.

    Equal group contribution is deliberate: A has more columns than N, but
    we do not want the larger number of columns in A to automatically make
    A dominate the definition of a neighbor.
    """
    available = [value for value in group_similarities.values() if value is not None]

    if not available:
        return 0.0

    return float(np.mean(available))


# ============================================================
# 5. FIND PHONOLOGICAL NEIGHBORS
# ============================================================
def find_neighbors(
    df: pd.DataFrame,
    target_row: pd.Series,
    top_k: int = 30,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Find the top-K most phonologically similar signs."""
    results = []

    target_id = normalize_sign_name(target_row["EntryID"])

    for _, candidate_row in df.iterrows():
        candidate_id = normalize_sign_name(candidate_row["EntryID"])

        if candidate_id == target_id:
            continue

        group_scores = {}

        for group_name, columns in FEATURE_GROUPS.items():
            group_scores[group_name] = group_similarity(
                target_row,
                candidate_row,
                columns,
            )

        similarity = overall_similarity(group_scores)

        if similarity < min_similarity:
            continue

        result = {
            "EntryID": candidate_row["EntryID"],
            "overall_similarity": similarity,
        }

        for group_name in OUTPUT_GROUP_ORDER:
            result[f"similarity_{group_name}"] = group_scores[group_name]

        results.append(result)

    neighbors = pd.DataFrame(results)

    if neighbors.empty:
        raise ValueError("No phonological neighbors were found.")

    neighbors = neighbors.sort_values(
        by="overall_similarity",
        ascending=False,
        kind="stable",
    )

    return neighbors.head(top_k).reset_index(drop=True)


# ============================================================
# 6. CONVERT NEIGHBORHOOD INTO CONTRASTIVE STRENGTH
# ============================================================
def calculate_contrast_scores(neighbors: pd.DataFrame) -> dict[str, float]:
    """
    Calculate how strongly each group differs from the target across its neighbors.

    For one neighbor:
        contrast = 1 - similarity

    Neighbors receive a similarity-based confidence weight:
        q = overall_similarity

    This emphasizes closer neighbors while still using the whole selected
    neighborhood.
    """
    weights = neighbors["overall_similarity"].to_numpy(dtype=float)

    # Prevent a completely dissimilar neighbor from having zero contribution.
    weights = np.maximum(weights, 1e-8)

    scores = {}

    for group_name in OUTPUT_GROUP_ORDER:
        similarity_values = neighbors[f"similarity_{group_name}"].to_numpy(dtype=float)

        valid_mask = ~np.isnan(similarity_values)

        if not np.any(valid_mask):
            scores[group_name] = 0.0
            continue

        valid_similarity = similarity_values[valid_mask]
        valid_weights = weights[valid_mask]

        contrast_values = 1.0 - valid_similarity

        weighted_average = np.average(
            contrast_values,
            weights=valid_weights,
        )

        scores[group_name] = float(weighted_average)

    return scores


# ============================================================
# 7. NORMALIZE INTO DTW WEIGHTS
# ============================================================
def normalize_scores(
    scores: dict[str, float],
    smoothing: float = 0.01,
) -> dict[str, float]:
    """
    Convert contrast scores into weights that sum to 1.

    A small smoothing value avoids an exact zero weight when the dataset
    contains no observed contrast for one group.
    """
    adjusted = {}

    for group_name in OUTPUT_GROUP_ORDER:
        adjusted[group_name] = max(float(scores.get(group_name, 0.0)), 0.0) + smoothing

    total = sum(adjusted.values())

    if total <= 0:
        raise ValueError("Could not normalize contrast scores.")

    normalized = {
        group_name: adjusted[group_name] / total
        for group_name in OUTPUT_GROUP_ORDER
    }

    return normalized


# ============================================================
# 8. BUILD ONE SIGN PROFILE
# ============================================================
def build_sign_profile(
    df: pd.DataFrame,
    sign_name: str,
    top_k: int,
    min_similarity: float,
    smoothing: float,
) -> tuple[dict, pd.DataFrame]:
    """Build one sign's ASL-LEX-derived phonological prior profile."""
    target_row = find_target_row(df, sign_name)

    neighbors = find_neighbors(
        df=df,
        target_row=target_row,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    contrast_scores = calculate_contrast_scores(neighbors)
    weights = normalize_scores(
        contrast_scores,
        smoothing=smoothing,
    )

    profile = {
        "sign": str(target_row["EntryID"]),
        "method": "ASL-LEX-derived phonological prior weights",
        "parameters": {
            "top_k_neighbors": top_k,
            "minimum_neighbor_similarity": min_similarity,
            "smoothing": smoothing,
        },
        "contrast_scores": contrast_scores,
        "weights": weights,
        "neighbors": [
            {
                "EntryID": str(row["EntryID"]),
                "similarity": round(float(row["overall_similarity"]), 6),
            }
            for _, row in neighbors.iterrows()
        ],
    }

    return profile, neighbors


# ============================================================
# 9. MAIN PROGRAM
# ============================================================
def parse_sign_list(sign_argument: str) -> list[str]:
    """Read comma-separated sign names."""
    signs = []

    for item in sign_argument.split(","):
        cleaned = item.strip()
        if cleaned:
            signs.append(cleaned)

    if not signs:
        raise ValueError("No sign names were provided.")

    return signs


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASL-LEX-derived P/A/JM/N weights for ASL signs."
    )

    parser.add_argument(
        "--csv",
        default="signdata.csv",
        help="Path to ASL-LEX signdata.csv",
    )

    parser.add_argument(
        "--signs",
        required=True,
        help="Comma-separated EntryID values, e.g. ME,YOU,HELLO",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of nearest phonological neighbors to use",
    )

    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Ignore neighbors below this overall similarity",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.01,
        help="Small value added before weight normalization",
    )

    parser.add_argument(
        "--json-output",
        default="sign_weights.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--csv-output",
        default="sign_weights_report.csv",
        help="Output CSV report",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    if not 0.0 <= args.min_similarity <= 1.0:
        raise ValueError("--min-similarity must be between 0 and 1.")

    if args.smoothing < 0:
        raise ValueError("--smoothing cannot be negative.")

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {csv_path}")

    df = load_dataset(str(csv_path))
    validate_columns(df)

    signs = parse_sign_list(args.signs)

    all_profiles = {}
    report_rows = []

    for sign_name in signs:
        profile, neighbors = build_sign_profile(
            df=df,
            sign_name=sign_name,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            smoothing=args.smoothing,
        )

        canonical_name = profile["sign"]
        all_profiles[canonical_name] = profile

        row = {
            "sign": canonical_name,
            "contrast_P": profile["contrast_scores"]["P"],
            "contrast_A": profile["contrast_scores"]["A"],
            "contrast_JM": profile["contrast_scores"]["JM"],
            "contrast_N": profile["contrast_scores"]["N"],
            "weight_P": profile["weights"]["P"],
            "weight_A": profile["weights"]["A"],
            "weight_JM": profile["weights"]["JM"],
            "weight_N": profile["weights"]["N"],
            "neighbor_count": len(neighbors),
        }

        report_rows.append(row)

        print("\n" + "=" * 60)
        print(f"Sign: {canonical_name}")
        print("Contrast scores:")
        for group_name in OUTPUT_GROUP_ORDER:
            print(f"  {group_name:>2}: {profile['contrast_scores'][group_name]:.6f}")

        print("Weights:")
        for group_name in OUTPUT_GROUP_ORDER:
            print(f"  {group_name:>2}: {profile['weights'][group_name]:.6f}")

        print("Top neighbors:")
        for item in profile["neighbors"][:10]:
            print(
                f"  {item['EntryID']:<25} "
                f"similarity={item['similarity']:.4f}"
            )

    with open(args.json_output, "w", encoding="utf-8") as file:
        json.dump(all_profiles, file, indent=2, ensure_ascii=False)

    report = pd.DataFrame(report_rows)
    report.to_csv(args.csv_output, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"Saved JSON: {args.json_output}")
    print(f"Saved CSV : {args.csv_output}")


if __name__ == "__main__":
    main()
