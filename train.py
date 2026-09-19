"""Chronological NBA game-winner training pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

BASE_STATS = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV"]
ROLL_WINDOWS = (3, 5)


def season_ids(start_season: str, end_season: str) -> list[str]:
    start = int(start_season[:4])
    end = int(end_season[:4])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start, end + 1)]


def fetch_games(start_season: str, end_season: str, pause: float = 0.65) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in season_ids(start_season, end_season):
        result = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]
        result["SEASON_ID_LABEL"] = season
        frames.append(result)
        time.sleep(pause)
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


def add_team_pregame_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    grouped = df.groupby("TEAM_ID", group_keys=False)

    df["HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["REST_DAYS"] = grouped["GAME_DATE"].diff().dt.days.clip(lower=0, upper=7)
    df["REST_DAYS"] = df["REST_DAYS"].fillna(3)
    df["B2B"] = (df["REST_DAYS"] <= 1).astype(int)
    df["GAME_NUMBER"] = grouped.cumcount() + 1
    df["WIN"] = (df["WL"] == "W").astype(int)

    df["WIN_PCT"] = grouped["WIN"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )

    for stat in BASE_STATS:
        for window in ROLL_WINDOWS:
            df[f"{stat}_L{window}"] = grouped[stat].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
            )

    # Count consecutive wins strictly before the current game.
    def prior_streak(values: pd.Series) -> pd.Series:
        out: list[int] = []
        streak = 0
        for value in values.shift(1).fillna(0).astype(int):
            streak = streak + 1 if value else 0
            out.append(streak)
        return pd.Series(out, index=values.index)

    df["WIN_STREAK"] = grouped["WIN"].apply(prior_streak).reset_index(level=0, drop=True)
    return df


def build_matchups(team_games: pd.DataFrame) -> pd.DataFrame:
    featured = add_team_pregame_features(team_games)
    home = featured[featured["HOME"] == 1].copy()
    away = featured[featured["HOME"] == 0].copy()

    keep = [
        "GAME_ID", "GAME_DATE", "SEASON_ID_LABEL", "TEAM_ID", "TEAM_ABBREVIATION",
        "WIN", "REST_DAYS", "B2B", "GAME_NUMBER", "WIN_STREAK", "WIN_PCT",
    ] + [f"{stat}_L{window}" for stat in BASE_STATS for window in ROLL_WINDOWS]

    home = home[keep].add_prefix("HOME_").rename(
        columns={"HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE"}
    )
    away = away[keep].add_prefix("AWAY_").rename(
        columns={"AWAY_GAME_ID": "GAME_ID", "AWAY_GAME_DATE": "GAME_DATE"}
    )

    games = home.merge(away, on=["GAME_ID", "GAME_DATE"], validate="one_to_one")
    games["TARGET_HOME_WIN"] = games["HOME_WIN"]

    numeric_pairs = [
        "REST_DAYS", "B2B", "GAME_NUMBER", "WIN_STREAK", "WIN_PCT",
    ] + [f"{stat}_L{window}" for stat in BASE_STATS for window in ROLL_WINDOWS]

    for name in numeric_pairs:
        games[f"DIFF_{name}"] = games[f"HOME_{name}"] - games[f"AWAY_{name}"]

    games["REST_ADVANTAGE"] = games["HOME_REST_DAYS"] - games["AWAY_REST_DAYS"]
    return games.sort_values("GAME_DATE").reset_index(drop=True)


def train_model(games: pd.DataFrame, test_season: str, output: Path) -> None:
    feature_columns = [c for c in games.columns if c.startswith("DIFF_")]
    feature_columns += ["HOME_REST_DAYS", "AWAY_REST_DAYS", "REST_ADVANTAGE"]

    usable = games.dropna(subset=feature_columns + ["TARGET_HOME_WIN"]).copy()
    train = usable[usable["HOME_SEASON_ID_LABEL"] != test_season]
    test = usable[usable["HOME_SEASON_ID_LABEL"] == test_season]
    if train.empty or test.empty:
        raise ValueError("The requested split produced an empty train or test set.")

    model = XGBClassifier(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_columns], train["TARGET_HOME_WIN"])
    predictions = model.predict(test[feature_columns])

    print(f"Train games: {len(train):,}")
    print(f"Test games:  {len(test):,}")
    print(f"Accuracy:    {accuracy_score(test['TARGET_HOME_WIN'], predictions):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(test["TARGET_HOME_WIN"], predictions))
    print(classification_report(test["TARGET_HOME_WIN"], predictions, digits=4))

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_columns}, output)
    print(f"Saved model bundle to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", default="2014-15")
    parser.add_argument("--test-season", default="2023-24")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/nba_xgb.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv:
        raw = pd.read_csv(args.input_csv, parse_dates=["GAME_DATE"])
    else:
        raw = fetch_games(args.start_season, args.test_season)
        if args.export_csv:
            args.export_csv.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(args.export_csv, index=False)

    matchups = build_matchups(raw)
    train_model(matchups, args.test_season, args.output)


if __name__ == "__main__":
    main()
