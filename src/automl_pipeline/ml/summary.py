from pandas import DataFrame


def summarize(df: DataFrame) -> dict:
    overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "total_cells": df.size,
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 1),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }

    col_stats = []
    for col in df.columns:
        s = df[col]
        stats = {
            "name": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null_pct": round(s.isna().mean() * 100, 1),
            "unique": int(s.nunique()),
        }
        if s.dtype.kind in "ifc":
            stats["mean"] = round(float(s.mean()), 3)
            stats["std"] = round(float(s.std(ddof=0)), 3)
            stats["min"] = round(float(s.min()), 3)
            stats["max"] = round(float(s.max()), 3)
        elif s.dtype.kind in "uib":
            stats["min"] = int(s.min()) if not s.isna().all() else None
            stats["max"] = int(s.max()) if not s.isna().all() else None
        elif s.dtype.kind in "O":
            top = s.mode()
            stats["top"] = str(top.iloc[0]) if not top.empty else ""
        col_stats.append(stats)

    return {"overview": overview, "columns": col_stats}
