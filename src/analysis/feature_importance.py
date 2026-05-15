from __future__ import annotations

import pandas as pd


def feature_importance_table(model, feature_names: list[str], top_k: int = 30) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    return table.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
