"""
SkillCorner Open Data loading for the MTE pipeline.

VERIFIED against the actual checkout at D:\\SkillCorner_opendata (2026-11-01).
Schema facts (confirmed from the repo README and by inspecting the files):

  matches.json  - a LIST of match dicts, each with "id", "home_team"{id,
                  short_name}, "away_team"{...}, "status", "date_time".

  data/matches/<id>/<id>_match.json - match metadata: "players"[] each with
                  id, team_id, trackable_object, player_role{name,
                  position_group, acronym} (goalkeeper = name == "Goalkeeper"),
                  and "match_periods"[] with per-period start_frame/end_frame.
                  IMPORTANT: tracking player_data.player_id == players[].id
                  (NOT trackable_object).

  data/matches/<id>/<id>_tracking_extrapolated.jsonl - one JSON per frame:
                  frame (int, 10 fps), timestamp ("HH:MM:SS.cc" or null),
                  period (1/2, null on the padding frames at the start),
                  ball_data{x, y, z, is_detected},
                  possession{player_id, group},
                  player_data[{x, y, player_id, is_detected}]  (no velocity!)

  data/matches/<id>/<id>_dynamic_events.csv - ~250 columns (event_type,
                  team_id, x_start/y_start, frame_start, attacking_side, ...).

  data/matches/<id>/<id>_phases_of_play.csv - frame_start/frame_end, period,
                  attacking_side (left_to_right/right_to_left),
                  team_in_possession_id, team_in_possession_phase_type (8
                  values: build_up/chaotic/create/direct/finish/quick_break/
                  set_play/transition), team_out_of_possession_phase_type (8).

  data/aggregates/ - season-level player aggregates (physical / passing /
                  obrag), NOT consumed by the MTE pipeline.

  Coordinates: METRES, origin at the pitch CENTRE, x = long axis (~+-52 m
  with out-of-bounds coverage beyond the lines), y = short axis (~+-34 m).
  "left_to_right" means attacking toward +x.  Tracking has NO velocity ->
  finite-difference the positions at 10 Hz.

The functions below mirror this schema exactly; the dataset does NOT need to
be copied into this project - pass the SkillCorner checkout via
config.SKILLCORNER_DATA_DIR (or --data-dir).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import config

FRAME_RATE_HZ = 10.0
GOALKEEPER_ROLE_NAME = "Goalkeeper"
N_OUTFIELD = 10          # outfield players per team in the GVRNN view


def match_dir(data_dir: Path, match_id) -> Path:
    return Path(data_dir) / "matches" / str(match_id)


def load_match_index(data_dir: Path) -> list:
    """matches.json -> list of match dicts (the file IS a JSON list)."""
    with open(Path(data_dir) / "matches.json") as f:
        return json.load(f)


def load_match_metadata(data_dir: Path, match_id) -> dict:
    """{id}_match.json -> dict (players, match_periods, home/away team)."""
    p = match_dir(data_dir, match_id) / f"{match_id}_match.json"
    with open(p) as f:
        return json.load(f)


def load_skillcorner_tracking(data_dir: Path, match_id) -> dict:
    """Parse {id}_tracking_extrapolated.jsonl into numpy arrays.

    Returns dict:
        pos        (T, N, 2) player positions [m], NaN where the player was
                          off the pitch (substituted out / not yet on)
        detected   (T, N) bool, True when measured (not extrapolated)
        player_ids (N,) int player ids == match.json players[].id  (the
                          tracking player_id key is the PLAYER id, NOT the
                          trackable_object id - verified 22/22)
        presence   (N,) int frames with a real position (playing time),
                          computed BEFORE any fill
        ball_pos   (T, 2),  ball_z (T,)
        frame      (T,) int original frame numbers (10 fps grid)
        period     (T,) int (1/2)
        frame_times (T,) float seconds (= frame / 10)

    Frames with period is None (padding - the length is NOT fixed at 10,
    some matches have thousands) and frames with EMPTY player_data (dead-ball
    stretches: no players AND null ball - tracking is all-or-nothing per
    frame) are dropped; T ~ 43k per match after the drop.  Original frame
    numbers are kept so event/phase alignment survives.
    """
    p = match_dir(data_dir, match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    valid = []
    with open(p) as f:
        for line in f:
            fr = json.loads(line)
            if fr.get("period") is None:
                continue                         # padding frames
            if not fr["player_data"]:
                continue                         # dead-ball frames (no tracking)
            valid.append(fr)

    T = len(valid)
    frame = np.array([fr["frame"] for fr in valid], dtype=np.int64)
    period = np.array([fr["period"] for fr in valid], dtype=np.int64)

    # distinct player ids in order of first appearance
    seen = {}
    for fr in valid:
        for pd_ in fr["player_data"]:
            pid = pd_["player_id"]
            if pid not in seen:
                seen[pid] = len(seen)
    N = len(seen)
    ids = [None] * N
    for pid, i in seen.items():
        ids[i] = pid

    pos = np.full((T, N, 2), np.nan, dtype=np.float32)
    detected = np.zeros((T, N), dtype=bool)
    for t, fr in enumerate(valid):
        for pd_ in fr["player_data"]:
            i = seen[pd_["player_id"]]
            if pd_.get("x") is not None and pd_.get("y") is not None:
                pos[t, i] = (float(pd_["x"]), float(pd_["y"]))
                detected[t, i] = bool(pd_.get("is_detected"))

    ball_pos = np.full((T, 2), np.nan, dtype=np.float32)
    ball_z = np.full(T, np.nan, dtype=np.float32)
    for t, fr in enumerate(valid):
        b = fr.get("ball_data") or {}
        if b.get("x") is not None:
            ball_pos[t] = (float(b["x"]), float(b["y"]))
            ball_z[t] = float(b["z"]) if b.get("z") is not None else np.nan

    # presence (frames with a real position) - computed on the RAW data, so
    # it reflects true playing time, not any fill
    presence = (~np.isnan(pos[:, :, 0])).sum(axis=0).astype(np.int64)

    return {"pos": pos, "detected": detected, "player_ids": np.array(ids),
            "presence": presence, "ball_pos": ball_pos, "ball_z": ball_z,
            "frame": frame, "period": period,
            "frame_times": frame.astype(np.float32) / FRAME_RATE_HZ}


def _ffill_bfill(x: np.ndarray, label: str) -> np.ndarray:
    """Forward-fill then back-fill NaN along the frame axis (per column),
    reporting how many values were filled.  Raises if NaN remains (a fully
    empty column - should never happen after the dead-ball frame drop)."""
    x = np.asarray(x, dtype=float)
    nan_before = int(np.isnan(x).sum())
    if nan_before == 0:
        return x
    import pandas as pd
    shape = x.shape
    flat = pd.DataFrame(x.reshape(len(x), -1)).ffill().bfill().values
    out = np.array(flat.reshape(shape), copy=True)   # guaranteed fresh/writable
    if np.isnan(out).any():
        raise ValueError(f"{label}: NaN remains even after fill - check the "
                         f"tracking data")
    print(f"[data_loading] {label}: filled {nan_before} NaN values "
          f"(forward/backward fill)")
    return out


def _diff_velocity(x: np.ndarray, hz: float) -> np.ndarray:
    """Finite-difference velocity in m/s (tracking has none); first row
    duplicates the second to keep length T."""
    v = np.full_like(x, np.nan, dtype=np.float32)
    d = np.diff(x, axis=0) * hz
    v[1:] = d
    v[0] = v[1] if len(v) > 1 else 0.0
    return v


def attack_direction_by_team(data_dir: Path, match_id) -> dict:
    """{(period, team_id): +1 (left_to_right, toward +x) or -1} from the
    phases-of-play file (mode of attacking_side over each team's phases)."""
    p = match_dir(data_dir, match_id) / f"{match_id}_phases_of_play.csv"
    import csv
    dirs = defaultdict(Counter)
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("attacking_side") and row.get("team_in_possession_id"):
                key = (int(float(row["period"])), int(row["team_in_possession_id"]))
                dirs[key][row["attacking_side"]] += 1
    return {k: (+1 if c.most_common(1)[0][0] == "left_to_right" else -1)
            for k, c in dirs.items()}


def export_gvrnn_npz(
    data_dir: Path,
    match_id,
    team_id: int,
    out_npz: Path,
    player_ids: Optional[list] = None,
    roles: Optional[dict] = None,
) -> dict:
    """Select ONE team's 10 outfield players + the ball (goalkeeper excluded,
    per the paper) and write the tracking npz consumed by gvrnn.train /
    gvrnn.extract.  The team is re-oriented so it ALWAYS attacks toward +x
    (x is mirrored in periods where its attacking_side is right_to_left;
    y is unchanged - the short axis is symmetric under the end switch).

    Node selection:
      * player_ids given -> use exactly those player ids IN THE GIVEN ORDER
        (custom node set, e.g. the substitute-analysis node sets: 7 full-match
        + filler + sub + replaced starter).  Non-team / goalkeeper ids are
        skipped with a warning.
      * None -> the top-10 outfield players by TRACKING PRESENCE (frames with
        a position = playing time), not by match.json list order - that order
        mixes in substitutes; the top-10 by presence are the real starters.

    The npz carries (in addition to the tracking arrays):
      player_ids (N,) int64  node index -> player id
      live_mask  (T, N) bool True where the position was real (BEFORE the
                  off-pitch forward/backward fill) - used for last-N-live
                  window selection downstream
      frame      (T,) int64  original 10 fps frame numbers (exact ints)
    plus a sidecar <name>.meta.json describing the node set.  Off-pitch
    stretches are forward/backward-filled (players subbed off / not yet on
    are frozen at their last position); the fill count is reported.
    """
    meta = load_match_metadata(data_dir, match_id)
    trk = load_skillcorner_tracking(data_dir, match_id)

    outfield_all = [p for p in meta["players"]
                    if int(p["team_id"]) == int(team_id)
                    and p["player_role"]["name"] != GOALKEEPER_ROLE_NAME
                    and p.get("id") is not None]
    # NOTE the join key: tracking player_data.player_id == match.json
    # players[].id (NOT trackable_object), verified empirically.
    col_of = {pid: i for i, pid in enumerate(trk["player_ids"])}
    presence = {}
    for p in outfield_all:
        i = col_of.get(int(p["id"]))
        presence[int(p["id"])] = (0 if i is None else
                                  int(trk["presence"][i]))

    if player_ids is not None:
        by_id = {int(p["id"]): p for p in meta["players"]}
        outfield = []
        for pid in player_ids:
            p = by_id.get(int(pid))
            if p is None or int(p["team_id"]) != int(team_id):
                print(f"[data_loading WARN] player {pid} is not a team "
                      f"{team_id} player in match.json; skipped", flush=True)
                continue
            if p["player_role"]["name"] == GOALKEEPER_ROLE_NAME:
                print(f"[data_loading WARN] player {pid} is the goalkeeper "
                      f"(excluded by the paper); skipped", flush=True)
                continue
            outfield.append(p)
        selection = "custom"
    else:
        outfield = sorted(outfield_all, key=lambda p: presence[int(p["id"])],
                          reverse=True)[:N_OUTFIELD]
        if len(outfield) < N_OUTFIELD:
            print(f"[data_loading WARN] team {team_id} has only "
                  f"{len(outfield)} outfield players in match.json",
                  flush=True)
        selection = "presence-top10"

    player_ids = [int(p["id"]) for p in outfield]
    col_pairs = [(pid, col_of[pid]) for pid in player_ids if pid in col_of]
    if len(col_pairs) != len(player_ids):
        missing = [pid for pid in player_ids if pid not in col_of]
        print(f"[data_loading WARN] {len(missing)} selected players never "
              f"appear in tracking: {missing}", flush=True)
    player_ids = [pid for pid, _ in col_pairs]
    cols = [c for _, c in col_pairs]

    pos_raw = trk["pos"][:, cols]
    live_mask = ~np.isnan(pos_raw[:, :, 0])      # BEFORE any fill
    # fill only the SELECTED players' gaps (players subbed off / not yet on
    # are frozen at their last position); the fill count is reported per
    # export so the amount of synthetic data is visible
    pos = _ffill_bfill(pos_raw.copy(), f"{match_id} team {team_id} selected players")
    ball = _ffill_bfill(trk["ball_pos"].copy(), f"{match_id} ball position")
    vel = _diff_velocity(pos, FRAME_RATE_HZ)
    ball_vel = _diff_velocity(ball, FRAME_RATE_HZ)

    # attack-direction re-orientation per period
    sign = attack_direction_by_team(data_dir, match_id)
    periods = np.unique(trk["period"])
    for pv in periods:
        s = sign.get((int(pv), int(team_id)), +1)
        if s < 0:
            m = trk["period"] == pv
            pos[m, :, 0] *= -1
            vel[m, :, 0] *= -1
            ball[m, 0] *= -1
            ball_vel[m, 0] *= -1
        elif s == +1 and (int(pv), int(team_id)) not in sign:
            print(f"[data_loading WARN] no possession phase found for team "
                  f"{team_id} in period {pv}; assuming +x attack", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, pos=pos, vel=vel,
                        ball_pos=ball, ball_vel=ball_vel,
                        team_ids=np.zeros(pos.shape[1], dtype=int),
                        period_of_frame=trk["period"],
                        frame_times=trk["frame_times"],
                        player_ids=np.array(player_ids, dtype=np.int64),
                        live_mask=live_mask,
                        frame=trk["frame"].astype(np.int64))

    # sidecar describing the node set (id -> node index, live windows, role)
    players_meta = []
    for node, pid in enumerate(player_ids):
        live = np.flatnonzero(live_mask[:, node])
        first = int(trk["frame"][live[0]]) if len(live) else None
        last = int(trk["frame"][live[-1]]) if len(live) else None
        role = "auto"
        if roles and pid in roles:
            role = roles[pid]
        players_meta.append({"id": pid, "node": node,
                             "presence": int(live_mask[:, node].sum()),
                             "first_frame": first, "last_frame": last,
                             "role": role})
    sidecar = {
        "match_id": match_id, "team_id": team_id, "selection": selection,
        "frame_rate_hz": FRAME_RATE_HZ, "n_frames": int(len(trk["frame"])),
        "n_players": int(len(player_ids)),
        "players": players_meta,
    }
    sidecar_path = out_npz.with_suffix(".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    meta_out = {"match_id": match_id, "team_id": team_id,
                "n_players": pos.shape[1], "frames": len(trk["frame"]),
                "frame_rate_hz": FRAME_RATE_HZ,
                "player_ids": player_ids, "sidecar": str(sidecar_path)}
    print(f"[data_loading] exported {out_npz}: {meta_out}")
    return meta_out


def load_events(data_dir: Path, match_id) -> dict:
    """{id}_dynamic_events.csv and {id}_phases_of_play.csv as raw rows
    (used by the phase-conditioned extension of PROTOCOL.md scheme 1; NOT
    consumed by the MTE pipeline itself)."""
    import csv
    out = {}
    for name in ("dynamic_events.csv", "phases_of_play.csv"):
        p = match_dir(data_dir, match_id) / f"{match_id}_{name}"
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        out[name] = rows
        print(f"[data_loading] {name}: {len(rows)} rows, "
              f"columns {list(rows[0].keys())[:6]}... " if rows else "empty")
    return out


def main(argv=None):
    """Probe the SkillCorner checkout and export one team's GVRNN view:

        python -m data_loading --data-dir D:/SkillCorner_opendata/data \\
            --match 1886347 --team 4177 --out artifacts/m1/tracking_team0.npz
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--data-dir", default=str(config.SKILLCORNER_DATA_DIR))
    ap.add_argument("--match", type=int, default=1886347)
    ap.add_argument("--team", type=int, default=None)
    ap.add_argument("--out", default="artifacts/tracking_team0.npz")
    ap.add_argument("--player-ids", default=None,
                    help="comma-separated custom 10-player node set "
                         "(node order = this order); omit for top-10-presence")
    args = ap.parse_args(argv)

    idx = load_match_index(Path(args.data_dir))
    m = next((x for x in idx if int(x["id"]) == args.match), None)
    if m is None:
        raise SystemExit(f"match {args.match} not in matches.json")
    if args.team is None:
        args.team = int(m["home_team"]["id"])
    print(f"[data_loading] {m['home_team']['short_name']} vs "
          f"{m['away_team']['short_name']} ({args.match})")
    pids = None
    if args.player_ids:
        pids = [int(x) for x in args.player_ids.split(",")]
    export_gvrnn_npz(Path(args.data_dir), args.match, args.team,
                     config.abs_path(args.out), player_ids=pids)


if __name__ == "__main__":
    main()
