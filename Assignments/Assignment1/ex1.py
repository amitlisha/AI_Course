import ex1_check
import search
import utils

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

        search.Problem.__init__(self, initial_state)

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
        """This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate"""
        # Simple admissible heuristic:
        # At least one POUR action is required for each unit of remaining
        # water needed by the plants, regardless of movement and loading.
        # So the sum of all remaining water requirements is a lower bound
        # on the remaining cost.
        taps_state, plants_state, robots_state = node.state
        return sum(need for (i, j, need) in plants_state)

    def h_gbfs(self, node):
        """This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate"""
        # Simple heuristic for GBFS:
        # Count how many plants still need any water at all.
        # This encourages the search to reduce the number of unfinished plants.
        taps_state, plants_state, robots_state = node.state
        return sum(1 for (i, j, need) in plants_state if need > 0)


def create_watering_problem(game):
    print("<<create_watering_problem")
    """ Create a pressure plate problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return WateringProblem(game)


if __name__ == "__main__":
    ex1_check.main()
