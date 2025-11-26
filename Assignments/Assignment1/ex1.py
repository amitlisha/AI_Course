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
        # initial is the init_state dictionary as described in the assignment.
        # We store static parts (size, walls) on the problem instance and keep
        # the dynamic parts (taps, plants, robots) inside the state.
        self.size = initial["Size"]
        self.walls = frozenset(initial.get("Walls", set()))

        # Each component of the state is immutable so that the whole state is hashable.
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

        initial_state = (taps_state, plants_state, robots_state)

        self._precompute_distances()

        search.Problem.__init__(self, initial_state)

    def _precompute_distances(self):
        """Precompute shortest path distances between all free cells, respecting walls."""
        rows, cols = self.size
        walls = self.walls
        self._dist = {}  # (i,j) -> dict[(i2,j2)] = distance

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for si in range(rows):
            for sj in range(cols):
                if (si, sj) in walls:
                    continue  # we never stand on walls

                start = (si, sj)
                d = {start: 0}
                q = deque([start])

                while q:
                    i, j = q.popleft()
                    for di, dj in moves:
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < rows and 0 <= nj < cols):
                            continue
                        if (ni, nj) in walls:
                            continue
                        if (ni, nj) not in d:
                            d[(ni, nj)] = d[(i, j)] + 1
                            q.append((ni, nj))

                self._dist[start] = d

    def grid_dist(self, a, b):
        """Shortest-path distance between two cells, or a big number if unreachable."""
        return self._dist.get(a, {}).get(b, float("inf"))

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
        """Generates the successor states returns [(action, achieved_states, ...)]"""
        successors = []

        taps, plants, robots = self._state_to_components(state)
        rows, cols = self.size
        walls = self.walls

        # Map occupied cells to robot ids to prevent collisions
        occupied = {(i, j): rid for rid, (i, j, load, capacity) in robots.items()}

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

                robots_new = dict(robots)
                robots_new[rid] = (ni, nj, load, capacity)

                new_state = self._components_to_state(taps, plants, robots_new)
                action_str = f"{move_name}{{{rid}}}"
                successors.append((action_str, new_state))

            pos = (i, j)

            # Load action: robot on a tap, tap has water, robot not at capacity
            if pos in taps and load < capacity and taps[pos] > 0:
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

    # ---------- Heuristics for A* and GBFS ----------

    # def _max_min_robot_to_plant_distance(self, plants_state, robots_state):
    #     """
    #     Helper for heuristic: for each plant that still needs water, compute the
    #     minimum Manhattan distance from any robot to that plant; then take the max
    #     over plants.

    #     This is a lower bound on the movement cost, ignoring walls and taps.
    #     """
    #     # Collect robot positions
    #     robot_positions = [(i, j) for (rid, i, j, load, capacity) in robots_state]
    #     if not robot_positions:
    #         return 0

    #     # Only plants that still need water
    #     need_plants = [(i, j, need) for (i, j, need) in plants_state if need > 0]
    #     if not need_plants:
    #         return 0

    #     max_min_dist = 0
    #     for pi, pj, need in need_plants:
    #         min_dist = min(abs(pi - ri) + abs(pj - rj) for (ri, rj) in robot_positions)
    #         if min_dist > max_min_dist:
    #             max_min_dist = min_dist
    #     return max_min_dist

    # def h_astar(self, node):
    #     """This is the heuristic. It gets a node (not a state)
    #     and returns a goal distance estimate"""
    #     # Improved admissible heuristic:
    #     # 1. Lower bound on POUR actions: total remaining water units.
    #     # 2. Lower bound on movement: for each plant that still needs water,
    #     #    compute the minimum Manhattan distance from any robot; then take
    #     #    the maximum over plants (some plant is "farthest" from robots).
    #     #
    #     # The true remaining cost must be at least:
    #     #   - the number of remaining POUR actions, and
    #     #   - the distance to reach the farthest plant.
    #     #
    #     # Taking max of two lower bounds is still a valid lower bound.
    #     taps_state, plants_state, robots_state = node.state

    #     # (1) Remaining POURs
    #     remaining_water = sum(need for (i, j, need) in plants_state)

    #     # (2) Movement lower bound
    #     movement_lb = self._max_min_robot_to_plant_distance(plants_state, robots_state)

    #     return max(remaining_water, movement_lb)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def h_astar(self, node):
        """
        A* heuristic for the watering problem using precomputed grid distances.

        h(s) = pours_needed + loads_needed + movement_needed, where:
        - pours_needed = total remaining water units plants still need
        - loads_needed = extra LOAD actions needed beyond water already carried
        - movement_needed = distance (in moves) needed to bring water
                            to the farthest plant, using problem.grid_dist
        """
        taps_state, plants_state, robots_state = node.state

        # --- Collect plants that still need water ---
        plants = [(i, j, need) for (i, j, need) in plants_state if need > 0]
        if not plants:
            # All plants satisfied -> goal
            return 0

        # --- Non-move lower bound: pours + extra loads ---
        # total_need = how many units of water all plants still need
        total_need = 0
        for _, _, need in plants:
            total_need += need

        # robots + total_carry = how much water is currently inside robots
        total_carry = 0
        robots = []
        for rid, i, j, load, cap in robots_state:
            robots.append((rid, i, j, load, cap))
            total_carry += load

        pours_needed = total_need
        loads_needed = max(0, total_need - total_carry)

        # --- Movement lower bound: farthest plant from any water source ---
        # Water sources are: taps that still have water + robots that currently carry water
        water_sources = []
        for i, j, wu in taps_state:
            if wu > 0:
                water_sources.append((i, j))

        for rid, i, j, load, cap in robots:
            if load > 0:
                water_sources.append((i, j))

        movement_needed = 0
        if water_sources:
            # For each plant, find distance to its closest source, then take max over plants
            for pi, pj, need in plants:
                d = min(self.grid_dist((pi, pj), src) for src in water_sources)
                # If instance is solvable, d should be finite
                if d > movement_needed:
                    movement_needed = d
        else:
            # No water anywhere yet -> we don't enforce a movement LB (still admissible).
            movement_needed = 0

        # Final heuristic: lower bound on total remaining actions
        return pours_needed + loads_needed + movement_needed


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

            # Wrap the heuristic so it checks the wall-clock time on each call
            def timed_h(node, _base_h=base_h, _start=start_time):
                if time.time() - _start > TIMEOUT:
                    raise TimeoutException()
                return _base_h(node)

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

            rows.append(
                {
                    "problem": prob_name,
                    "heuristic": h_name,
                    "solved": solved,
                    "solution_length": solution_length,
                    "optimal_length": optimal_len,
                    "runtime_sec": runtime,
                    "error": err,
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
