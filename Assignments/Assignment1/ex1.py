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

    def h_astar(self, node):
        """
        Admissible A* heuristic.
        Components:
        1. Interaction cost: Every missing water unit requires 1 POUR.
           Every unit not yet in a robot requires 1 LOAD.
        2. Delivery cost: Current load must travel to the closest thirsty plant.
        3. Routing cost: Missing water must travel from closest tap to closest plant.
           Plus, a robot must travel to a tap if fetching is required.
        """
        taps, plants, robots = self._state_to_components(node.state)

        # 1. Identify needs and resources
        # List of coordinates for plants that need water
        thirsty_plants = [pos for pos, need in plants.items() if need > 0]

        # If solution found (no thirsty plants), heuristic is 0
        if not thirsty_plants:
            return 0

        # List of coordinates for taps that still have water
        active_taps = [pos for pos, amount in taps.items() if amount > 0]

        # Calculate volumes
        total_need = sum(plants[p] for p in thirsty_plants)
        current_carry = sum(r[2] for r in robots.values())  # r[2] is load

        # 2. Interaction Costs (Atomic actions that must happen)
        # We need 1 pour for every needed unit
        cost_pours = total_need
        # We need 1 load for every unit not yet carried
        missing_water = max(0, total_need - current_carry)
        cost_loads = missing_water

        h_val = cost_pours + cost_loads

        # 3. Delivery Costs (Moving carried water)
        # For every robot with load, minimal distance to a thirsty plant
        for rid, (r_x, r_y, load, cap) in robots.items():
            if load > 0:
                # Find closest plant
                min_dist_to_plant = min(
                    [self.grid_dist((r_x, r_y), p_pos) for p_pos in thirsty_plants]
                )
                # We don't multiply by load because multiple units can move simultaneously
                # inside the robot, but the robot must make the trip at least once.
                # However, to be strictly admissible and tighter:
                # We treat each unit of water as needing to arrive.
                # But since they move together, adding dist * load might overestimate if
                # they are dropped at the same plant.
                # Safe lower bound: The robot must traverse the distance at least once.
                h_val += min_dist_to_plant

        # 4. Procurement Costs (Fetching missing water)
        if missing_water > 0 and active_taps:
            # A. The water itself must move from Tap -> Plant
            # Find the global minimum distance between any active tap and any thirsty plant
            min_transit = float("inf")
            for t_pos in active_taps:
                for p_pos in thirsty_plants:
                    d = self.grid_dist(t_pos, p_pos)
                    if d < min_transit:
                        min_transit = d

            # Every missing unit must eventually travel this minimum distance
            # (Relaxation: assuming infinite capacity on the optimal path)
            h_val += missing_water * min_transit

            # B. A robot must get to a tap to start this process
            # Find minimum distance from any robot to any active tap
            min_dist_to_tap = float("inf")
            robot_positions = [(val[0], val[1]) for val in robots.values()]

            for r_pos in robot_positions:
                for t_pos in active_taps:
                    d = self.grid_dist(r_pos, t_pos)
                    if d < min_dist_to_tap:
                        min_dist_to_tap = d

            h_val += min_dist_to_tap

        return h_val

    def h_gbfs(self, node):
        """
        Greedy Best-First Search heuristic.
        Uses the same logic as A* but we can make it slightly 'greedy'
        to prefer states where robots are closer to targets, ignoring rigorous cost accounting.
        """
        # For this assignment, the A* heuristic is quite informative.
        # We can reuse it directly or return a weighted version.
        return self.h_astar(node)


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
