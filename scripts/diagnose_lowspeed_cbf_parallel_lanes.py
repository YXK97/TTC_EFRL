import argparse
import csv
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defmarl.env.mve import MVEEnvState
from defmarl.env.mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.scaling_lowspeed import scaling_calc


def make_state(x: float, y: float, v: float, delta: float = 0.0):
    return jnp.array([x, y, 1.0, 0.0, v, delta], dtype=jnp.float32)


def logged_to_raw_cost(cost: float) -> float:
    return cost - 1.0 if cost > 1.0 else cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-min", type=float, default=-30.0)
    parser.add_argument("--x-max", type=float, default=30.0)
    parser.add_argument("--num", type=int, default=241)
    parser.add_argument("--ego-lane", type=int, default=1)
    parser.add_argument("--obst-lane", type=int, default=0)
    parser.add_argument("--ego-v", type=float, default=20.0 / 3.6)
    parser.add_argument("--obst-v", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lowspeed_cbf_parallel_lanes.csv"))
    args = parser.parse_args()

    env = MVELaneChangeAndOverTake_LowSpeed_CBF(num_agents=1, max_step=1)
    lane_centers = np.asarray(env.params["lane_centers"])
    ego_y = float(lane_centers[args.ego_lane])
    obst_y = float(lane_centers[args.obst_lane])

    action = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    goal = make_state(100.0, ego_y, args.ego_v)[None, :]
    rel_xs = np.linspace(args.x_min, args.x_max, args.num)
    rows = []

    for rel_x in rel_xs:
        ego = make_state(0.0, ego_y, args.ego_v)[None, :]
        obst = make_state(float(rel_x), obst_y, args.obst_v)[None, :]
        graph = env.get_graph(MVEEnvState(agent=ego, goal=goal, obstacle=obst))

        cost, cost_real = env.get_cost(graph, action)
        alpha = scaling_calc(
            ego[0],
            obst[0],
            env.params["ego_bb_size"],
            env.params["ego_lr"],
            env.params["obst_bb_size"],
            env.params["obst_lr"],
        )

        cbf_cost = float(cost[0, 1])
        cost_real_obst = float(cost_real[0, 1])
        alpha_components = 1.0 - np.asarray(cost_real[0])
        cost_components = np.asarray(cost[0])
        rows.append(
            {
                "rel_x_obst_minus_ego_m": float(rel_x),
                "ego_y_m": ego_y,
                "obst_y_m": obst_y,
                "ego_v_mps": args.ego_v,
                "obst_v_mps": args.obst_v,
                "alpha": float(alpha),
                "alpha_obst": float(alpha_components[1]),
                "alpha_bound_low": float(alpha_components[2]),
                "alpha_bound_high": float(alpha_components[3]),
                "cost_real_obst": cost_real_obst,
                "cost_real_bound_low": float(cost_real[0, 2]),
                "cost_real_bound_high": float(cost_real[0, 3]),
                "cbf_cost_obst_logged": cbf_cost,
                "cbf_cost_obst_raw_est": logged_to_raw_cost(cbf_cost),
                "cbf_cost_bound_low_logged": float(cost_components[2]),
                "cbf_cost_bound_high_logged": float(cost_components[3]),
                "is_unsafe_real": cost_real_obst >= 0.0,
                "is_unsafe_cbf_logged": cbf_cost >= 0.0,
                "is_any_unsafe_real": bool(np.max(np.asarray(cost_real[0])) >= 0.0),
                "is_any_unsafe_cbf_logged": bool(np.max(cost_components) >= 0.0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    unsafe_real = sum(row["is_unsafe_real"] for row in rows)
    unsafe_cbf = sum(row["is_unsafe_cbf_logged"] for row in rows)
    any_unsafe_real = sum(row["is_any_unsafe_real"] for row in rows)
    any_unsafe_cbf = sum(row["is_any_unsafe_cbf_logged"] for row in rows)
    print(f"lane_centers={lane_centers.tolist()}")
    print(f"ego lane {args.ego_lane}: y={ego_y:.3f}, obstacle lane {args.obst_lane}: y={obst_y:.3f}")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"real unsafe rows: {unsafe_real}/{len(rows)}")
    print(f"cbf unsafe rows:  {unsafe_cbf}/{len(rows)}")
    print(f"any-component real unsafe rows: {any_unsafe_real}/{len(rows)}")
    print(f"any-component cbf unsafe rows:  {any_unsafe_cbf}/{len(rows)}")
    print("sample rows near side-by-side rel_x=0:")
    for row in rows:
        if abs(row["rel_x_obst_minus_ego_m"]) <= 1.0:
            print(row)


if __name__ == "__main__":
    main()
