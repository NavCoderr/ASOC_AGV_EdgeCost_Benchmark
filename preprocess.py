# -*- coding: utf-8 -*-

from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


RAW_LOG_FILE = "naveen12features_title.csv"
NODE_FILE = "Node_F3.csv"
EDGE_FILE = "Edge_Distances3_.csv"
OUT_DIR = "outputs"

SESSION_THRESH = 100_000.0
ENERGY_SCALE = 3.6
MIN_MOVE_SPEED = 0.09
SNAP_RADIUS = 1.5
FFILL_GAP = 10
MAX_SEG_SECONDS = 120
MIN_MOVE_PTS = 2

C = SimpleNamespace(
    RAW_LOG_FILE=RAW_LOG_FILE,
    NODE_FILE=NODE_FILE,
    EDGE_FILE=EDGE_FILE,
    OUT_DIR=OUT_DIR,
    SESSION_THRESH=SESSION_THRESH,
    ENERGY_SCALE=ENERGY_SCALE,
    MIN_MOVE_SPEED=MIN_MOVE_SPEED,
    SNAP_RADIUS=SNAP_RADIUS,
    FFILL_GAP=FFILL_GAP,
    MAX_SEG_SECONDS=MAX_SEG_SECONDS,
    MIN_MOVE_PTS=MIN_MOVE_PTS,
)


def _parse_ts(s):
    s = str(s).strip()
    if s in ("", "nan"):
        return np.nan

    p = s.split(":")
    try:
        if len(p) == 2:
            return float(p[0]) * 60 + float(p[1])
        if len(p) == 3:
            return float(p[0]) * 3600 + float(p[1]) * 60 + float(p[2])
    except Exception:
        return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def _n(x):
    return pd.to_numeric(x, errors="coerce")


def _b(s):
    return (
        s.astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
        .fillna(0.0)
    )


def _nc(c):
    return str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def load_and_split(path):
    print(f"[preprocess] Loading {Path(path).name} ...")

    df = pd.read_csv(path, low_memory=False)
    print(f"[preprocess] Raw rows: {len(df)}")

    df["t_sec"] = df["timestamp"].map(_parse_ts)
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec").reset_index(drop=True)

    cum = _n(df["Cumulative energy consumption"])
    df["session"] = np.where(cum < C.SESSION_THRESH, 2, 1)

    for sid in [1, 2]:
        n = (df["session"] == sid).sum()
        t0 = df.loc[df["session"] == sid, "t_sec"].min()
        t1 = df.loc[df["session"] == sid, "t_sec"].max()
        print(f"  Session {sid}: {n} rows  t={t0:.0f}–{t1:.0f} s")

    return df


def clean_session(df_all, sid, out_dir):
    d = df_all[df_all["session"] == sid].copy()

    n0 = len(d)
    d = d.drop_duplicates(subset=["t_sec", "X-coordinate", "Y-coordinate"])
    d = d.sort_values("t_sec").reset_index(drop=True)

    print(f"  S{sid}: dedup {n0} → {len(d)} rows  ({n0 - len(d)} duplicates removed)")

    d["t_sec"] = d["t_sec"] - d["t_sec"].min()

    e_cum = _n(d["Cumulative energy consumption"]).ffill().bfill()
    e_smooth = e_cum.cummax()
    e_delta = e_smooth.diff().fillna(0).clip(lower=0)
    d["energy_J_step"] = e_delta.values * C.ENERGY_SCALE

    sc_col = "Safety - Front Scanner Protective Zone Active"
    d["scanner_active"] = _b(d[sc_col]) if sc_col in d.columns else 0.0

    d["t_1hz"] = np.floor(d["t_sec"]).astype(int)

    num_cols = [
        c for c in [
            "X-coordinate",
            "Y-coordinate",
            "Speed",
            "power consumption",
            "current consuption",
            "Heading",
            "Position confidence",
            "Battery value",
            "RIGHT DRIVE SIGNALS.ActualSpeed_R",
            "LEFT DRIVE SIGNALS.ActualSpeed_L",
        ]
        if c in d.columns
    ]

    cat_cols = [c for c in ["Going to ID", "Current segment"] if c in d.columns]

    agg = {c: "mean" for c in num_cols}
    agg["energy_J_step"] = "sum"
    agg["scanner_active"] = "max"

    for c in cat_cols:
        agg[c] = "last"

    d1 = d.groupby("t_1hz", as_index=False).agg(agg).sort_values("t_1hz")

    t_max = int(d1["t_1hz"].max())
    full = pd.DataFrame({"t_1hz": np.arange(0, t_max + 1)})
    d1 = full.merge(d1, on="t_1hz", how="left")

    for c in num_cols:
        if c in d1.columns:
            d1[c] = (
                d1[c]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )

    d1["energy_J_step"] = d1["energy_J_step"].fillna(0)
    d1["scanner_active"] = d1["scanner_active"].fillna(0)
    d1["t_sec"] = d1["t_1hz"]
    d1["session"] = sid

    diffs = d1["t_1hz"].diff().dropna()
    all_ok = bool((diffs == 1).all())

    d1["interval_check"] = d1["t_1hz"].diff().fillna(1)
    d1["is_1hz_ok"] = d1["interval_check"] == 1

    move = d1[_n(d1["Speed"]) >= C.MIN_MOVE_SPEED].reset_index(drop=True)

    print(
        f"  S{sid}: 1Hz rows={len(d1)}  move={len(move)}  "
        f"dur={t_max / 60:.1f} min  all_1s_ok={all_ok}"
    )

    out = Path(out_dir)
    d1.to_csv(out / f"check_1hz_session{sid}.csv", index=False)

    print(f"  → saved: check_1hz_session{sid}.csv")

    return d1, move


def run_preprocess(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(C.RAW_LOG_FILE, low_memory=False)
    df_raw.head(200).to_csv(out_dir / "check_raw_sample.csv", index=False)

    print("[preprocess] Saved check_raw_sample.csv")

    df = load_and_split(C.RAW_LOG_FILE)

    print("\n[preprocess] Cleaning sessions to 1 Hz ...")

    s1_1hz, s1_move = clean_session(df, sid=1, out_dir=out_dir)
    s2_1hz, s2_move = clean_session(df, sid=2, out_dir=out_dir)

    combined = pd.concat([s1_move, s2_move], ignore_index=True)
    combined.to_csv(out_dir / "check_1hz_combined_move.csv", index=False)

    print("\n[preprocess] Done.")
    print(f"  Session 1: {len(s1_1hz)} rows (1Hz, {s1_1hz['t_1hz'].max() / 60:.1f} min)")
    print(f"  Session 2: {len(s2_1hz)} rows (1Hz, {s2_1hz['t_1hz'].max() / 60:.1f} min)")

    return s1_1hz, s2_1hz, s1_move, s2_move


def _compute_node_stats(raw_csv, node_x, node_y, node_ids, snap_r=0.8):
    stats = {int(nid): {"avg_speed": 0.22, "stop_pct": 0.30} for nid in node_ids}

    try:
        df = pd.read_csv(raw_csv, low_memory=False)

        df["t_sec"] = df["timestamp"].map(_parse_ts)
        df = df.dropna(subset=["t_sec"])

        cum = _n(df["Cumulative energy consumption"])
        df = df[cum >= C.SESSION_THRESH].copy()

        df = df.drop_duplicates(subset=["t_sec", "X-coordinate", "Y-coordinate"])
        df["spd"] = _n(df["Speed"]).fillna(0.0)
        df["x"] = _n(df["X-coordinate"])
        df["y"] = _n(df["Y-coordinate"])

        for i, nid in enumerate(node_ids):
            d2 = np.sqrt(
                (df["x"] - float(node_x[i])) ** 2
                + (df["y"] - float(node_y[i])) ** 2
            )

            near = df[d2 < snap_r]

            if len(near) < 5:
                continue

            mov = near[near["spd"] >= C.MIN_MOVE_SPEED]

            stats[int(nid)] = {
                "avg_speed": float(mov["spd"].mean()) if len(mov) > 0 else 0.22,
                "stop_pct": float((near["spd"] < 0.01).mean()),
            }

        print(
            f"[graph] Node data-stats loaded  "
            f"(station stop_pct avg: {np.mean([v['stop_pct'] for v in stats.values()]):.3f})"
        )

    except Exception as e:
        print(f"[graph] Node stats fallback: {e}")

    return stats


def load_graph(raw_csv=None):
    nodes = pd.read_csv(C.NODE_FILE, sep=None, engine="python")
    edges = pd.read_csv(C.EDGE_FILE, sep=None, engine="python")

    nm = {_nc(c): c for c in nodes.columns}

    def pn(*cs):
        for c in cs:
            if _nc(c) in nm:
                return nm[_nc(c)]
        return None

    nid_c = pn("node", "nodeid", "node_id", "id")
    x_c = pn("xcoordinate", "x", "x-coordinate")
    y_c = pn("ycoordinate", "y", "y-coordinate")

    tmp = pd.DataFrame(
        {
            "nid": _n(nodes[nid_c]),
            "x": _n(nodes[x_c]),
            "y": _n(nodes[y_c]),
        }
    ).dropna()

    tmp["nid"] = tmp["nid"].astype(int)
    tmp = tmp.drop_duplicates("nid").sort_values("nid").reset_index(drop=True)

    nids = tmp["nid"].to_numpy(int)
    nx = tmp["x"].to_numpy(np.float32)
    ny = tmp["y"].to_numpy(np.float32)

    ni2i = {int(n): i for i, n in enumerate(nids)}
    N = len(nids)

    nr = (
        nodes.copy()
        .assign(**{nid_c: _n(nodes[nid_c])})
        .dropna(subset=[nid_c])
        .assign(**{nid_c: lambda d: d[nid_c].astype(int)})
        .drop_duplicates(nid_c)
        .set_index(nid_c)
        .reindex(nids)
        .reset_index()
    )

    def sc(col, d=0.0):
        if col in nr.columns:
            return _n(nr[col]).fillna(d).to_numpy(np.float32)
        return np.full(N, d, np.float32)

    tc = sc("Type_Corridor")
    ti = sc("Type_Intersection")
    ts2 = sc("Type_Station")
    ch = (sc("charging_flag") > 0).astype(np.float32)

    node_type = np.zeros(N, int)
    node_type[tc > 0] = 1
    node_type[ts2 > 0] = 2

    em = {_nc(c): c for c in edges.columns}

    def pe(*cs):
        for c in cs:
            if _nc(c) in em:
                return em[_nc(c)]
        return None

    uc = pe("from", "u", "u_node_id", "start")
    vc = pe("to", "v", "v_node_id", "end")
    dc = pe("distance", "edge_distance", "dist", "d")

    et = pd.DataFrame(
        {
            "u": _n(edges[uc]),
            "v": _n(edges[vc]),
            "d": _n(edges[dc]).fillna(0.0),
        }
    ).dropna(subset=["u", "v"])

    et["u"] = et["u"].astype(int)
    et["v"] = et["v"].astype(int)

    eu = et["u"].to_numpy(np.int64)
    ev = et["v"].to_numpy(np.int64)
    ed = et["d"].to_numpy(np.float32)

    in_deg = np.zeros(N, np.float32)
    out_deg = np.zeros(N, np.float32)
    adj_t = defaultdict(list)

    for u, v in zip(eu, ev):
        out_deg[ni2i[int(u)]] += 1
        in_deg[ni2i[int(v)]] += 1
        adj_t[int(u)].append(int(v))

    in_n = in_deg / (in_deg.max() + 1e-6)
    out_n = out_deg / (out_deg.max() + 1e-6)
    is_term = ((in_deg + out_deg) <= 2).astype(np.float32)

    n_sn = np.array(
        [
            sum(1 for v in adj_t[int(nid)] if node_type[ni2i[v]] == 2)
            for nid in nids
        ],
        np.float32,
    )

    n_cn = np.array(
        [
            sum(1 for v in adj_t[int(nid)] if node_type[ni2i[v]] == 1)
            for nid in nids
        ],
        np.float32,
    )

    n_sn /= n_sn.max() + 1e-6
    n_cn /= n_cn.max() + 1e-6

    xn = (nx - nx.min()) / (nx.max() - nx.min() + 1e-6)
    yn = (ny - ny.min()) / (ny.max() - ny.min() + 1e-6)

    dm = sc("dist_mean")
    dm_n = dm / (dm.max() + 1e-6)

    dmax = sc("dist_max")
    dmin = sc("dist_min")
    drn = (dmax - dmin).clip(0) / ((dmax - dmin).max() + 1e-6)

    avg_spd = np.full(N, 0.22, np.float32)
    stop_pc = np.full(N, 0.30, np.float32)

    if raw_csv:
        nstats = _compute_node_stats(raw_csv, nx, ny, nids)

        for i, nid in enumerate(nids):
            avg_spd[i] = nstats[int(nid)]["avg_speed"]
            stop_pc[i] = nstats[int(nid)]["stop_pct"]

    avg_spd_n = avg_spd / (avg_spd.max() + 1e-6)

    nf = np.stack(
        [
            xn,
            yn,
            in_n,
            out_n,
            ch,
            tc,
            ti,
            ts2,
            dm_n,
            drn,
            avg_spd_n,
            stop_pc,
            n_sn,
            n_cn,
            is_term,
        ],
        axis=1,
    ).astype(np.float32)

    uidx = np.array([ni2i[int(u)] for u in eu], np.int64)
    vidx = np.array([ni2i[int(v)] for v in ev], np.int64)
    ei = np.stack([uidx, vidx], axis=0)

    edge_set = {(int(u), int(v)) for u, v in zip(eu, ev)}
    dist_map = {(int(u), int(v)): float(d) for u, v, d in zip(eu, ev, ed)}
    chargers = [int(n) for n, c in zip(nids, ch) if c > 0]

    print(
        f"[graph] nodes={N}  edges={len(eu)}  "
        f"node_feat_dim={nf.shape[1]}  chargers={chargers}"
    )

    print(
        f"[graph] types: station={int((node_type == 2).sum())}  "
        f"corridor={int((node_type == 1).sum())}  "
        f"intersection={int((node_type == 0).sum())}"
    )

    return {
        "node_ids": nids,
        "node_x": nx,
        "node_y": ny,
        "node_feat": nf,
        "node_type": node_type,
        "eu": eu,
        "ev": ev,
        "ed": ed,
        "ni2i": ni2i,
        "edge_index": ei,
        "edge_set": edge_set,
        "dist_map": dist_map,
        "chargers": chargers,
    }


def snap_to_graph(df_1hz, G):
    x = _n(df_1hz["X-coordinate"]).to_numpy()
    y = _n(df_1hz["Y-coordinate"]).to_numpy()

    snapped = []

    for px, py in zip(x, y):
        if not (np.isfinite(px) and np.isfinite(py)):
            snapped.append(np.nan)
            continue

        d = np.sqrt((G["node_x"] - px) ** 2 + (G["node_y"] - py) ** 2)
        j = int(np.argmin(d))

        if d[j] <= C.SNAP_RADIUS:
            snapped.append(int(G["node_ids"][j]))
        else:
            snapped.append(np.nan)

    df = df_1hz.copy()
    df["node_id"] = snapped
    df["node_id"] = df["node_id"].ffill(limit=C.FFILL_GAP)

    pct = 100 * df["node_id"].notna().mean()
    sid = int(df["session"].iloc[0]) if "session" in df.columns else "?"

    print(f"  S{sid}: snap {pct:.1f}%  (radius={C.SNAP_RADIUS}m, ffill={C.FFILL_GAP}s)")

    return df


def extract_edge_samples(df_snapped, G, sid):
    nids = _n(df_snapped["node_id"]).to_numpy()

    if "t_sec" in df_snapped.columns:
        t_s = df_snapped["t_sec"].to_numpy()
    else:
        t_s = df_snapped["t_1hz"].to_numpy()

    sp = _n(df_snapped.get("Speed", pd.Series(np.zeros(len(df_snapped))))).fillna(0).to_numpy()
    e1 = _n(df_snapped.get("energy_J_step", pd.Series(np.zeros(len(df_snapped))))).fillna(0).to_numpy()
    pw = _n(df_snapped.get("power consumption", pd.Series(np.zeros(len(df_snapped))))).fillna(0).to_numpy()
    cur = _n(df_snapped.get("current consuption", pd.Series(np.zeros(len(df_snapped))))).fillna(0).to_numpy()
    hd = _n(df_snapped.get("Heading", pd.Series(np.full(len(df_snapped), np.nan)))).to_numpy()
    rs = _n(df_snapped.get("RIGHT DRIVE SIGNALS.ActualSpeed_R", pd.Series(np.full(len(df_snapped), np.nan)))).to_numpy()
    ls = _n(df_snapped.get("LEFT DRIVE SIGNALS.ActualSpeed_L", pd.Series(np.full(len(df_snapped), np.nan)))).to_numpy()
    sc = _n(df_snapped.get("scanner_active", pd.Series(np.zeros(len(df_snapped))))).fillna(0).to_numpy()

    wd = np.where(np.isfinite(rs) & np.isfinite(ls), np.abs(rs - ls), 0.0)

    msp = sp[sp >= C.MIN_MOVE_SPEED]
    vref = float(np.percentile(msp, 95)) if len(msp) > 10 else 0.2
    vref = max(vref, 1e-3)

    def hdiff(a):
        if len(a) < 2:
            return 0.0

        d = np.diff(a.astype(float))
        return float(np.mean(np.abs((d + 180) % 360 - 180)))

    rows = []
    n = len(df_snapped)
    i = 0

    while i < n - 2:
        if not np.isfinite(nids[i]):
            i += 1
            continue

        u = int(nids[i])
        j = i

        while j < n and np.isfinite(nids[j]) and int(nids[j]) == u:
            j += 1

        if j >= n or not np.isfinite(nids[j]):
            i = j
            continue

        v = int(nids[j])

        if v == u or (u, v) not in G["edge_set"]:
            i = j
            continue

        k = j

        while (
            k < n
            and np.isfinite(nids[k])
            and int(nids[k]) == v
            and (k - j) < 3
        ):
            k += 1

        dt = k - i

        if dt <= 2 or dt > C.MAX_SEG_SECONDS:
            i = j
            continue

        seg_sp = sp[i:k]
        seg_e1 = e1[i:k]
        seg_pw = pw[i:k]
        seg_cur = cur[i:k]
        seg_wd = wd[i:k]
        seg_sc = sc[i:k]
        seg_hd = hd[i:k]

        mm = (seg_sp >= C.MIN_MOVE_SPEED) & np.isfinite(seg_sp)

        if mm.sum() < C.MIN_MOVE_PTS:
            i = j
            continue

        ms = float(np.mean(seg_sp[mm]))
        em = float(np.sum(seg_e1[mm]))
        tm = int(mm.sum())

        pw_m = seg_pw[mm][np.isfinite(seg_pw[mm])]

        mpw = float(np.mean(pw_m)) if len(pw_m) else float(em / max(tm, 1))
        spw = float(np.std(pw_m)) if len(pw_m) else 0.0
        ppw = float(np.max(pw_m)) if len(pw_m) else 0.0

        if em == 0 and mpw > 0:
            em = mpw * tm

        cm_ = seg_cur[mm][np.isfinite(seg_cur[mm])]

        mc = float(np.mean(cm_)) if len(cm_) else 0.0
        sc2 = float(np.std(cm_)) if len(cm_) else 0.0

        dist = float(G["dist_map"][(u, v)])
        slow = float(np.clip(1.0 - ms / vref, 0, 1))
        stds = float(np.std(seg_sp[mm]))

        if np.isfinite(seg_wd[mm]).any():
            wdm = float(np.mean(seg_wd[mm][np.isfinite(seg_wd[mm])]))
        else:
            wdm = 0.0

        stopr = float(np.sum(seg_sp < 0.01) / max(dt, 1))
        scanr = float(np.mean(seg_sc))

        hh = seg_hd[mm][np.isfinite(seg_hd[mm])]
        ti = hdiff(hh) if len(hh) >= 2 else 0.0

        rows.append(
            {
                "session": sid,
                "u_node_id": u,
                "v_node_id": v,
                "edge_distance": dist,
                "time_s": float(tm),
                "energy_J": em,
                "mean_speed": ms,
                "slowdown_idx": slow,
                "std_speed": stds,
                "turn_intensity": ti,
                "mean_power_W": mpw,
                "std_power_W": spw,
                "peak_power_W": ppw,
                "mean_current": mc,
                "std_current": sc2,
                "wheel_diff_mean": wdm,
                "stop_ratio": stopr,
                "scanner_ratio": scanr,
                "obs_t_start": float(t_s[i]),
            }
        )

        i = j

    out = pd.DataFrame(rows)

    ue = out[["u_node_id", "v_node_id"]].drop_duplicates().shape[0]
    ze = int((out["energy_J"] == 0).sum())

    print(f"  S{sid}: {len(out)} edge samples  {ue} unique edges  zero_energy={ze}")

    return out


def run_extraction(s1_1hz, s2_1hz, G, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[extractor] Snapping to graph nodes ...")

    s1_snapped = snap_to_graph(s1_1hz, G)
    s2_snapped = snap_to_graph(s2_1hz, G)

    s1_snapped.to_csv(out_dir / "check_snapped_s1.csv", index=False)
    s2_snapped.to_csv(out_dir / "check_snapped_s2.csv", index=False)

    print("  Saved check_snapped_s1/s2.csv")

    print("[extractor] Extracting edge samples ...")

    e1 = extract_edge_samples(s1_snapped, G, sid=1)
    e2 = extract_edge_samples(s2_snapped, G, sid=2)
    ec = pd.concat([e1, e2], ignore_index=True)

    e1.to_csv(out_dir / "edge_samples_s1.csv", index=False)
    e2.to_csv(out_dir / "edge_samples_s2.csv", index=False)
    ec.to_csv(out_dir / "edge_samples_combined.csv", index=False)

    ue = ec[["u_node_id", "v_node_id"]].drop_duplicates().shape[0]

    print("\n[extractor] Saved edge_samples_s1/s2/combined.csv")
    print(f"  Total: {len(ec)} samples  Unique: {ue}/102  Never observed: {102 - ue}")

    return e1, e2, ec


if __name__ == "__main__":
    s1, s2, _, _ = run_preprocess(C.OUT_DIR)
    G = load_graph(raw_csv=C.RAW_LOG_FILE)
    run_extraction(s1, s2, G, C.OUT_DIR)