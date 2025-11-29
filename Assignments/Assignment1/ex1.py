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
        self._h_astar_cache = {}
        self._h_gbfs_cache = {}

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
        dist_to_tap = self._dist_to_tap

        # Single-tap info (used for corridor forcing when robot is empty)
        single_tap_mode = len(self.tap_positions) == 1
        tap_has_water = False
        if single_tap_mode:
            tap_pos = next(iter(self.tap_positions))
            tap_has_water = taps.get(tap_pos, 0) > 0

        for rid, (i, j, load, capacity) in robots.items():
            # Movement actions
            moves = [
                ("UP", (-1, 0)),
                ("DOWN", (1, 0)),
                ("LEFT", (0, -1)),
                ("RIGHT", (0, 1)),
            ]

            # --- Corridor forcing: when empty, force a shortest-path move to the tap ---
            force_tap_corridor = False
            if single_tap_mode and load == 0 and total_need > 0 and tap_has_water:
                d_here = dist_to_tap.get((i, j))
                # If we are not already on the tap and the tap is reachable from here,
                # an optimal solution can always choose to go monotonically "downhill"
                # in dist_to_tap until reaching the tap.
                if d_here is not None and d_here > 0:
                    force_tap_corridor = True
                else:
                    force_tap_corridor = False
            else:
                d_here = None  # not used

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

                # --- Apply corridor forcing when appropriate ---
                if force_tap_corridor:
                    d_next = dist_to_tap.get((ni, nj))
                    # Only allow moves that strictly reduce dist_to_tap
                    # (i.e., stay on some shortest path to the tap).
                    if d_next is None or d_next != d_here - 1:
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

        This version restores the full logic (including D_cycles) but avoids
        rebuilding dicts on every call by operating directly on the tuple state.
        Also caches state -> heuristic value.
        """
        state = node.state

        # --- Cache check ---
        cached = self._h_astar_cache.get(state)
        if cached is not None:
            return cached

        taps_state, plants_state, robots_state = state

        # --- 1. Remaining need & loads ---
        total_need = sum(need for (_, _, need) in plants_state if need > 0)
        if total_need == 0:
            self._h_astar_cache[state] = 0
            return 0

        total_load = sum(load for (_, _, _, load, _) in robots_state)
        remaining_loads = max(total_need - total_load, 0)
        remaining_pours = total_need

        h_val = remaining_loads + remaining_pours

        # If we don't need any more loads OR no taps/robots -> no tap-based movement bound
        if remaining_loads == 0 or not self.tap_positions or not robots_state:
            self._h_astar_cache[state] = h_val
            return h_val

        dist_to_tap = self._dist_to_tap

        # --- 2. D_RT: min robot -> nearest tap distance ---
        min_robot_to_tap = None
        for _, ri, rj, load, cap in robots_state:
            d = dist_to_tap.get((ri, rj))
            if d is None:
                continue
            if min_robot_to_tap is None or d < min_robot_to_tap:
                min_robot_to_tap = d

        if min_robot_to_tap is None:
            min_robot_to_tap = 0

        # --- 3. Generic plant distance bound: D_TP_single = farthest thirsty plant from taps ---
        d_max = 0
        plant_entries = []  # (distance_from_nearest_tap, need)
        for pi, pj, need in plants_state:
            if need <= 0:
                continue
            d = dist_to_tap.get((pi, pj))
            if d is None:
                continue
            plant_entries.append((d, need))
            if d > d_max:
                d_max = d

        D_TP_single = d_max if plant_entries else 0

        # --- 4. Extra strong bounds only for 1 tap & 1 robot ---
        D_cycles = 0
        LB_trips_per_plant = 0

        if len(robots_state) == 1 and len(self.tap_positions) == 1 and plant_entries:
            # Capacity of the single robot
            (_, ri, rj, load, C_max) = robots_state[0]
            C_max = max(C_max, 1)

            # free_units: units not forced to originate from taps
            free_units = min(total_load, total_need)

            # Build a canonical key: only plant distances & needs matter, not positions or order.
            key_plant = tuple(
                sorted((d, need) for (d, need) in plant_entries if need > 0)
            )
            cache_key = (free_units, C_max, key_plant)

            cached_pair = self._single_tr_lb_cache.get(cache_key)
            if cached_pair is not None:
                D_cycles, LB_trips_per_plant = cached_pair
            else:
                # --- Heavy computation done only once per (free_units, C_max, pattern) ---

                # Multiset of unit distances (one entry per remaining unit of need)
                unit_dists = []
                for d, need in plant_entries:
                    if need > 0:
                        unit_dists.extend([d] * need)

                if unit_dists:
                    unit_dists.sort(reverse=True)

                    if free_units < len(unit_dists):
                        # units_for_taps: those units that must still be fetched from taps
                        units_for_taps = unit_dists[free_units:]
                        N = len(units_for_taps)

                        # (a) Grouped tours bound (D_cycles)
                        K = (
                            N + C_max - 1
                        ) // C_max  # number of tours with capacity C_max
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

        h_val += min_robot_to_tap + D_TP

        # --- Store in cache and return ---
        self._h_astar_cache[state] = h_val
        return h_val

    def h_gbfs(self, node):
        """
        Greedy Best-First Search heuristic (can be non-admissible).

        Idea:
        - Count remaining interactions (LOAD + POUR), like in A*.
        - Add a cheap estimate of movement:
          * min robot->tap distance (using precomputed dist_to_tap).
          * approximate total tap->plant travel, based on:
                sum_over_plants(need * dist_to_nearest_tap(plant)) / avg_capacity

        This is cheap (O(#robots + #plants)) and correlates well with how much work
        remains, but it is allowed to overestimate (GBFS doesn't require admissibility).
        We also cache state -> heuristic value.
        """
        state = node.state

        # --- Cache check ---
        cached = self._h_gbfs_cache.get(state)
        if cached is not None:
            return cached

        taps_state, plants_state, robots_state = state

        # --- 1. Remaining water need ---
        total_need = 0
        for _, _, need in plants_state:
            if need > 0:
                total_need += need

        if total_need == 0:
            self._h_gbfs_cache[state] = 0
            return 0

        # If there are no robots, just return something large-ish.
        if not robots_state:
            val = 10 * total_need
            self._h_gbfs_cache[state] = val
            return val

        # --- 2. Current total load on robots ---
        total_load = 0
        total_capacity = 0
        for _, _, _, load, cap in robots_state:
            total_load += load
            total_capacity += cap

        remaining_loads = max(total_need - total_load, 0)
        remaining_pours = total_need

        # Base interaction cost (LOAD + POUR)
        h_val = remaining_loads + remaining_pours

        # If there are no taps, we can't use tap distances for movement.
        if not self.tap_positions or not self._dist_to_tap:
            self._h_gbfs_cache[state] = h_val
            return h_val

        dist_to_tap = self._dist_to_tap

        # --- 3. Min robot -> nearest tap distance (D_RT) ---
        min_robot_to_tap = None
        for _, ri, rj, load, cap in robots_state:
            d = dist_to_tap.get((ri, rj))
            if d is None:
                continue
            if (min_robot_to_tap is None) or (d < min_robot_to_tap):
                min_robot_to_tap = d

        if min_robot_to_tap is None:
            min_robot_to_tap = 0

        # --- 4. Approximate total tap -> plant movement ---
        sum_need_dist = 0
        for pi, pj, need in plants_state:
            if need <= 0:
                continue
            d = dist_to_tap.get((pi, pj))
            if d is None:
                continue
            sum_need_dist += need * d

        # Use an average capacity to approximate how many "tours" are needed.
        avg_capacity = max(1, total_capacity // max(1, len(robots_state)))
        approx_tap_to_plants = sum_need_dist // avg_capacity

        # Final GBFS heuristic
        h_val += min_robot_to_tap + approx_tap_to_plants

        # --- Store in cache and return ---
        self._h_gbfs_cache[state] = h_val
        return h_val


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


def experiment_search_methods():
    """
    Unified benchmark runner for both A* (with h_astar) and GBFS (with h_gbfs).
    Produces one combined results DataFrame.
    """
    import time
    import pandas as pd

    TIMEOUT = 60  # Seconds per run

    class TimeoutException(Exception):
        pass

    # List of problems: (name, dict, optimal_solution_len)
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

    # Algorithms to run: label, search_fn, heuristic_getter
    methods = [
        ("A*", search.astar_search, lambda P: P.h_astar),
        ("GBFS", search.greedy_best_first_graph_search, lambda P: P.h_gbfs),
    ]

    rows = []
    total = len(problems) * len(methods)
    counter = 0

    for prob_name, prob_def, optimal_len in problems:
        for alg_name, search_fn, h_get in methods:
            counter += 1
            print(f"[{counter}/{total}] {alg_name} on {prob_name}")

            problem = create_watering_problem(prob_def)
            base_h = h_get(problem)

            start_time = time.time()
            stats = {
                "calls": 0,
                "sum_h": 0,
                "min_h": float("inf"),
                "max_h": float("-inf"),
                "max_depth": 0,
                "max_g": 0,
                "max_f": 0,
            }

            # Decorated heuristic with timeout + instrumentation
            def H(node, _h=base_h, _start=start_time, _stats=stats):
                if time.time() - _start > TIMEOUT:
                    raise TimeoutException()

                v = _h(node)
                _stats["calls"] += 1
                _stats["sum_h"] += v
                _stats["min_h"] = min(_stats["min_h"], v)
                _stats["max_h"] = max(_stats["max_h"], v)

                d = getattr(node, "depth", 0)
                g = getattr(node, "path_cost", 0)
                f = g + v
                _stats["max_depth"] = max(_stats["max_depth"], d)
                _stats["max_g"] = max(_stats["max_g"], g)
                _stats["max_f"] = max(_stats["max_f"], f)

                return v

            solved, solution_length, err = False, None, None

            try:
                result = search_fn(problem, H)
            except TimeoutException:
                err = "TIMEOUT"
                result = None
            except Exception as e:
                err = str(e)
                result = None

            runtime = time.time() - start_time

            if result and isinstance(result[0], search.Node):
                solved = True
                path = result[0].path()[::-1]
                actions = [n.action for n in path][1:]
                solution_length = len(actions)
            elif err is None:
                err = "NO_SOLUTION"

            calls = stats["calls"] or 1
            h_avg = stats["sum_h"] / calls
            calls_per_sec = calls / runtime if runtime else None

            rows.append(
                dict(
                    problem=prob_name,
                    algorithm=alg_name,
                    solved=solved,
                    solution_length=solution_length,
                    optimal_length=optimal_len,
                    runtime_sec=runtime,
                    error=err,
                    heuristic_calls=calls,
                    heuristic_calls_per_sec=calls_per_sec,
                    h_min=stats["min_h"],
                    h_max=stats["max_h"],
                    h_avg=h_avg,
                    max_depth_seen=stats["max_depth"],
                    max_g_seen=stats["max_g"],
                    max_f_seen=stats["max_f"],
                )
            )

    df = pd.DataFrame(rows)
    print(df)
    return df


if __name__ == "__main__":
    # Default behavior (as in the original file): run the checker.
    ex1_check.main()
    # If you want to run the heuristic experiments manually, you can call:
    # experiment_heuristics_astar()
