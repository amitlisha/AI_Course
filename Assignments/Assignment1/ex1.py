import ex1_check
import search
import utils
from collections import deque


id = ["No numbers - I'm special!"]


class WateringProblem(search.Problem):
    """This class implements a pressure plate problem"""

    def __init__(self, initial):
        """Constructor only needs the initial state.
        Don't forget to set the goal or implement the goal test"""

        # Grid / static environment
        self.size = initial["Size"]
        self.walls = frozenset(initial.get("Walls", set()))

        # Static tap positions (used by heuristic)
        self.tap_positions = tuple(initial.get("Taps", {}).keys())

        self._dist_to_tap = self._bfs_from_all_taps(self.tap_positions)

        self._tap_distances = {}
        for tap_pos in initial.get("Taps", {}):
            self._tap_distances[tap_pos] = self._bfs_from(tap_pos)

        self._reachable_to_plant = self._bfs_from_all_plants(initial.get("Plants", {}))

        # --- Build initial state (same immutable tuple format you already had) ---
        taps_state = tuple(
            sorted((i, j, wu) for (i, j), wu in initial.get("Taps", {}).items())
        )
        plants_state = tuple(
            sorted((i, j, need) for (i, j), need in initial.get("Plants", {}).items())
        )
        robots_state = tuple(
            sorted(
                (rid, i, j, load, capacity)
                for rid, (i, j, load, capacity) in initial.get("Robots", {}).items()
            )
        )

        self._single_tr_lb_cache = {}

        initial_state = (taps_state, plants_state, robots_state)
        search.Problem.__init__(self, initial_state)

    def _bfs_from_all_plants(self, plants_dict):
        """
        Multi-source BFS starting from all plant positions.
        We use this to find all cells from which at least one plant is reachable.
        Returns a set of reachable cells.
        """
        rows, cols = self.size
        walls = self.walls

        from collections import deque

        q = deque()
        visited = set()

        # Initialize with all plant positions
        for i, j in plants_dict.keys():
            visited.add((i, j))
            q.append((i, j))

        while q:
            i, j = q.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if (ni, nj) in walls:
                    continue
                if (ni, nj) in visited:
                    continue
                visited.add((ni, nj))
                q.append((ni, nj))

        return visited

    def _bfs_from(self, start):
        """
        BFS from a starting cell over the static grid (walls, bounds).
        Returns a dict: cell -> shortest distance in moves.
        """
        rows, cols = self.size
        walls = self.walls

        dist = {start: 0}
        q = deque([start])

        while q:
            i, j = q.popleft()
            d = dist[(i, j)]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if (ni, nj) in walls:
                    continue
                if (ni, nj) in dist:
                    continue
                dist[(ni, nj)] = d + 1
                q.append((ni, nj))

        return dist

    def _bfs_from_all_taps(self, tap_positions):
        """
        Multi-source BFS from all taps simultaneously.
        Returns a dict: cell -> distance to nearest tap.
        """
        rows, cols = self.size
        walls = self.walls

        dist = {}
        q = deque()

        # Initialize queue with all taps at distance 0
        for pos in tap_positions:
            dist[pos] = 0
            q.append(pos)

        while q:
            i, j = q.popleft()
            d = dist[(i, j)]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if (ni, nj) in walls:
                    continue
                if (ni, nj) in dist:
                    continue
                dist[(ni, nj)] = d + 1
                q.append((ni, nj))

        return dist

    def _state_to_components(self, state):
        """Helper: converts a state tuple back to dictionaries."""
        taps_state, plants_state, robots_state = state

        taps = {(i, j): wu for (i, j, wu) in taps_state}
        plants = {(i, j): need for (i, j, need) in plants_state}
        robots = {
            rid: (i, j, load, capacity) for (rid, i, j, load, capacity) in robots_state
        }

        return taps, plants, robots

    def _components_to_state(self, taps, plants, robots):
        """Helper: converts dictionaries to a hashable state tuple."""
        taps_state = tuple(sorted((i, j, wu) for (i, j), wu in taps.items()))
        plants_state = tuple(sorted((i, j, need) for (i, j), need in plants.items()))
        robots_state = tuple(
            sorted(
                (rid, i, j, load, capacity)
                for rid, (i, j, load, capacity) in robots.items()
            )
        )
        return (taps_state, plants_state, robots_state)

    def successor(self, state):
        """Generates the successor states returns [(action, achieved_state), ...]"""
        successors = []

        taps, plants, robots = self._state_to_components(state)
        rows, cols = self.size
        walls = self.walls

        # --- Global info for pruning ---
        total_need = sum(need for need in plants.values() if need > 0)
        total_load = sum(load for (_, _, load, _) in robots.values())

        # If total_load >= total_need, we already carry enough water to satisfy all plants.
        # Any further LOAD is useless in an optimal plan.
        forbid_loads = total_need > 0 and total_load >= total_need

        # Map occupied cells to robot ids to prevent collisions
        occupied = {(i, j): rid for rid, (i, j, load, capacity) in robots.items()}

        reachable_to_plant = self._reachable_to_plant

        for rid, (i, j, load, capacity) in robots.items():
            # Movement actions
            moves = [
                ("UP", (-1, 0)),
                ("DOWN", (1, 0)),
                ("LEFT", (0, -1)),
                ("RIGHT", (0, 1)),
            ]

            for move_name, (di, dj) in moves:
                ni, nj = i + di, j + dj

                # Check grid bounds
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue

                # Check walls and other robots
                if (ni, nj) in walls:
                    continue
                if (ni, nj) in occupied:
                    continue

                # --- Region pruning: don't move into cells from which no plant is reachable ---
                if reachable_to_plant and (ni, nj) not in reachable_to_plant:
                    continue

                robots_new = dict(robots)
                robots_new[rid] = (ni, nj, load, capacity)

                new_state = self._components_to_state(taps, plants, robots_new)
                action_str = f"{move_name}{{{rid}}}"
                successors.append((action_str, new_state))

            pos = (i, j)

            # Load action: robot on a tap, tap has water, robot not at capacity
            if (
                pos in taps
                and load < capacity
                and taps[pos] > 0
                and not forbid_loads  # pruning: don't load if we already have enough water globally
            ):
                taps_new = dict(taps)
                taps_new[pos] = taps[pos] - 1

                robots_new = dict(robots)
                robots_new[rid] = (i, j, load + 1, capacity)

                new_state = self._components_to_state(taps_new, plants, robots_new)
                successors.append((f"LOAD{{{rid}}}", new_state))

            # Pour action: robot on a plant, robot has water, plant still needs water
            if pos in plants and load > 0 and plants[pos] > 0:
                plants_new = dict(plants)
                plants_new[pos] = plants[pos] - 1

                robots_new = dict(robots)
                robots_new[rid] = (i, j, load - 1, capacity)

                new_state = self._components_to_state(taps, plants_new, robots_new)
                successors.append((f"POUR{{{rid}}}", new_state))

        return successors

    def goal_test(self, state):
        """given a state, checks if this is the goal state, compares to the created goal state returns True/False"""
        taps_state, plants_state, robots_state = state

        # Goal: all plants have received the required water, i.e. remaining requirement is zero.
        for i, j, need in plants_state:
            if need != 0:
                return False
        return True

    def h_astar(self, node):
        """
        Admissible A* heuristic.

        - For general cases (multi taps/robots):
            h = remaining LOADs + remaining POURs
                + min(robot -> tap)
                + max(tap -> thirsty plant)

        - For the special case of 1 tap & 1 robot:
            add strong lower bounds on future tap<->plant tours:
            (a) grouping units into tours (D_cycles)
            (b) per-plant 'trips' bound (LB_trips_per_plant).
        """
        state = node.state
        taps_state, plants_state, robots_state = state

        # --- Rebuild simple dicts from tuples ---
        plants = {(i, j): need for (i, j, need) in plants_state}
        robots = {rid: (i, j, load, cap) for (rid, i, j, load, cap) in robots_state}

        # --- 1. Remaining need & loads ---
        total_need = sum(need for need in plants.values() if need > 0)
        if total_need == 0:
            return 0

        total_load = sum(load for (_, _, load, _) in robots.values())
        remaining_loads = max(total_need - total_load, 0)
        remaining_pours = total_need

        h = remaining_loads + remaining_pours

        # If we don't need any more loads OR no taps/robots -> no tap-based movement bound
        if remaining_loads == 0 or not self.tap_positions or not robots:
            return h

        dist_to_tap = self._dist_to_tap

        # --- 2. D_RT: min robot -> nearest tap distance ---
        min_robot_to_tap = None
        for rid, (ri, rj, load, cap) in robots.items():
            d = dist_to_tap.get((ri, rj))
            if d is None:
                continue
            if min_robot_to_tap is None or d < min_robot_to_tap:
                min_robot_to_tap = d

        if min_robot_to_tap is None:
            min_robot_to_tap = 0

        # --- 3. Generic plant distance bound: D_TP_single = farthest thirsty plant from taps ---
        d_max = 0
        plant_entries = []  # we'll also reuse this for the snake-specialized part
        for pos, need in plants.items():
            if need <= 0:
                continue
            d = dist_to_tap.get(pos)
            if d is None:
                continue
            plant_entries.append((d, need))
            if d > d_max:
                d_max = d

        D_TP_single = d_max if plant_entries else 0

        # --- 4. Extra strong bounds only for 1 tap & 1 robot ---
        D_cycles = 0
        LB_trips_per_plant = 0

        if len(robots) == 1 and len(self.tap_positions) == 1 and plant_entries:
            # Capacity of the single robot
            (_, (ri, rj, load, C_max)) = next(iter(robots.items()))
            C_max = max(C_max, 1)

            # We clamp free_units the same way as in the computation
            free_units = min(total_load, total_need)

            # Build a canonical key: only plant distances & needs matter, not positions or order.
            # plant_entries is [(d, need), ...]; we sort it to make the key canonical.
            key_plant = tuple(
                sorted((d, need) for (d, need) in plant_entries if need > 0)
            )
            cache_key = (free_units, C_max, key_plant)

            cached = self._single_tr_lb_cache.get(cache_key)
            if cached is not None:
                D_cycles, LB_trips_per_plant = cached
            else:
                # --- Heavy computation done only once per pattern ---

                # Build multiset of unit distances for remaining plant needs
                unit_dists = []
                for d, need in plant_entries:
                    if need > 0:
                        unit_dists.extend([d] * need)

                if unit_dists:
                    unit_dists.sort(reverse=True)

                    if free_units < len(unit_dists):
                        units_for_taps = unit_dists[
                            free_units:
                        ]  # these must come from taps
                        N = len(units_for_taps)

                        # (a) Grouped tours bound (D_cycles)
                        K = (N + C_max - 1) // C_max  # number of tours
                        group_max = []
                        for t in range(K):
                            idx = t * C_max
                            if idx < N:
                                group_max.append(units_for_taps[idx])

                        if group_max:
                            S = sum(group_max)
                            g_max = max(group_max)
                            # Total tour length >= 2*S - g_max (last tour need not return)
                            D_cycles = 2 * S - g_max

                        # (b) Per-plant "trips" bound
                        for d, need in plant_entries:
                            if need <= 0:
                                continue
                            k_p = (need + C_max - 1) // C_max  # ceil(need / C_max)
                            L_p = (2 * k_p - 1) * d
                            if L_p > LB_trips_per_plant:
                                LB_trips_per_plant = L_p

                # Store in cache (even if both are 0 – that’s still a valid result)
                self._single_tr_lb_cache[cache_key] = (D_cycles, LB_trips_per_plant)

        # Final tap->plant movement LB: generic max-distance OR the cycles bound OR per-plant bound
        D_TP = max(D_TP_single, D_cycles, LB_trips_per_plant)

        h += min_robot_to_tap + D_TP
        return h


def create_watering_problem(game):
    print("<<create_watering_problem")
    """ Create a pressure plate problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return WateringProblem(game)


# ---------- Experiment / Testing Utilities ----------

import ex1_check
import search
import utils


# ---------- Experiment / Testing Utilities ----------


def experiment_heuristics_astar():
    """
    Run A* with several heuristics on a selection of problems from ex1_check and
    collect results into a pandas DataFrame.

    Heuristics compared:
      - h_improved_astar (h_astar): max(remaining_water, movement_lb)

    Timeout:
      - 1 minute per run, enforced inside the heuristic (no multiprocessing, no signals)

    Extra debug fields per run:
      - heuristic_calls:    how many times the heuristic was evaluated
      - heuristic_calls_per_sec
      - h_min, h_max, h_avg: stats over heuristic values
      - max_depth_seen:     max node.depth seen by the heuristic
      - max_g_seen:         max node.path_cost seen
      - max_f_seen:         max (g + h) seen
    """
    import time
    import pandas as pd

    TIMEOUT = 60  # seconds

    class TimeoutException(Exception):
        """Raised when a single search run exceeds TIMEOUT seconds."""

        pass

    # List of (problem_name, problem_dict, optimal_solution_length_or_None)
    problems = [
        ("Problem_pdf", ex1_check.Problem_pdf, 20),
        ("problem1", ex1_check.problem1, 8),
        ("problem2", ex1_check.problem2, 20),
        ("problem3", ex1_check.problem3, 28),
        ("problem4", ex1_check.problem4, 13),
        ("problem5_deadend", ex1_check.problem5_deadend, None),
        ("problem6", ex1_check.problem6, 8),
        ("problem7", ex1_check.problem7, 20),
        ("problem_hard1", ex1_check.problem_hard1, 31),
        ("problem_hard2", ex1_check.problem_hard2, 24),
        ("problem_hard3", ex1_check.problem_hard3, 42),
        ("problem_hard4", ex1_check.problem_hard4, 25),
        ("problem_hard5", ex1_check.problem_hard5, 29),
        ("problem_hard6", ex1_check.problem_hard6, 33),
        ("problem_load", ex1_check.problem_load, 65),
        ("problem_10x10_single", ex1_check.problem_10x10_single, 106),
        ("problem_12x12_snake", ex1_check.problem_12x12_snake, 249),
        ("problem_12x12_snake_hard", ex1_check.problem_12x12_snake_hard, 343),
    ]

    # Heuristics to compare: name -> function-getter on WateringProblem
    def get_h_improved(p):
        return p.h_astar

    heuristics = [
        ("h_improved_astar", get_h_improved),
    ]

    rows = []
    total = len(problems) * len(heuristics)
    counter = 0

    for prob_name, prob_def, optimal_len in problems:
        for h_name, h_getter in heuristics:
            counter += 1
            print(f"[{counter}/{total}] Running {prob_name} with {h_name} ...")

            problem = create_watering_problem(prob_def)
            base_h = h_getter(problem)

            start_time = time.time()

            # --- instrumentation for heuristic/debug stats ---
            stats = {
                "calls": 0,
                "sum_h": 0.0,
                "min_h": float("inf"),
                "max_h": float("-inf"),
                "max_depth": 0,
                "max_g": 0.0,
                "max_f": 0.0,
            }

            def timed_h(node, _base_h=base_h, _start=start_time, _stats=stats):
                # timeout check
                if time.time() - _start > TIMEOUT:
                    raise TimeoutException()

                h_val = _base_h(node)

                # update stats
                _stats["calls"] += 1
                _stats["sum_h"] += h_val
                if h_val < _stats["min_h"]:
                    _stats["min_h"] = h_val
                if h_val > _stats["max_h"]:
                    _stats["max_h"] = h_val

                depth = getattr(node, "depth", 0)
                if depth > _stats["max_depth"]:
                    _stats["max_depth"] = depth

                g = getattr(node, "path_cost", 0.0)
                if g > _stats["max_g"]:
                    _stats["max_g"] = g

                f = g + h_val
                if f > _stats["max_f"]:
                    _stats["max_f"] = f

                return h_val

            solved = False
            solution_length = None
            err = None
            result = None

            try:
                result = search.astar_search(problem, timed_h)
            except TimeoutException:
                err = "TIMEOUT"
            except Exception as e:
                err = str(e)

            runtime = time.time() - start_time

            if result and isinstance(result[0], search.Node):
                solved = True
                path = result[0].path()[::-1]
                actions = [pi.action for pi in path][1:]
                solution_length = len(actions)
            elif err is None:
                # No node returned and no explicit error → treat as no solution
                err = "NO_SOLUTION"

            # derive stats
            calls = stats["calls"]
            h_min = None if stats["min_h"] == float("inf") else stats["min_h"]
            h_max = None if stats["max_h"] == float("-inf") else stats["max_h"]
            h_avg = stats["sum_h"] / calls if calls > 0 else None
            calls_per_sec = calls / runtime if runtime > 0 and calls > 0 else None

            rows.append(
                {
                    "problem": prob_name,
                    "heuristic": h_name,
                    "solved": solved,
                    "solution_length": solution_length,
                    "optimal_length": optimal_len,
                    "runtime_sec": runtime,
                    "error": err,
                    # --- new debug fields ---
                    "heuristic_calls": calls,
                    "heuristic_calls_per_sec": calls_per_sec,
                    "h_min": h_min,
                    "h_max": h_max,
                    "h_avg": h_avg,
                    "max_depth_seen": stats["max_depth"],
                    "max_g_seen": stats["max_g"],
                    "max_f_seen": stats["max_f"],
                }
            )

    df = pd.DataFrame(rows)
    print(df)
    return df


if __name__ == "__main__":
    # Default behavior (as in the original file): run the checker.
    ex1_check.main()
    # If you want to run the heuristic experiments manually, you can call:
    # experiment_heuristics_astar()
