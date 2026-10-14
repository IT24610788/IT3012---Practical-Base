import random
from collections import deque
import heapq
import math
from logic_engine import KnowledgeBase


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """An agent that computes paths using standard search algorithms."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'  # Switched to AStar to test Lab 07 logic
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    def sense_and_act(self, percept: dict) -> str:
        # If the plan is empty, we need to calculate a new path
        if not self.plan:
            agent_pos = percept['agent_pos']
            all_food = percept.get('all_food', [])

            # Edge case: If there's no food left, stay still
            if not all_food:
                return 'Stop'

            # Find the closest food pellet using Manhattan distance
            closest_food = min(
                all_food,
                key=lambda f: abs(f[0] - agent_pos[0]) +
                abs(f[1] - agent_pos[1])
            )

            # Formulate the search problem for the algorithms
            class PositionSearchProblem:
                def get_start_state(self):
                    return agent_pos

                def is_goal_state(self, state):
                    return state == closest_food

                def get_successors(self, state):
                    x, y = state
                    successors = []
                    # Assuming percept provides walls, default to empty set if not
                    walls = set(percept.get('walls', []))
                    # 4-way movement actions
                    for action, dx, dy in [('Up', 0, -1), ('Down', 0, 1), ('Left', -1, 0), ('Right', 1, 0)]:
                        next_state = (x + dx, y + dy)
                        if next_state not in walls:
                            # Return tuple: (next_state, action, cost)
                            successors.append((next_state, action, 1))
                    return successors

            problem = PositionSearchProblem()

            # Define sample facts as per Lab 07 Step 3.3
            tile_facts = {
                (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],  # deduces Retreat: blocked
                (0, 1): ['TargetVisible', 'HasDust'],                        # SafeToEngage only: allowed
            }

            # Execute the search method matching self.active_algo
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(problem)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(problem)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(problem)
            elif self.active_algo == 'AStar':
                self.plan = self.astar_search(
                    problem, closest_food, heuristic_type='manhattan', tile_facts=tile_facts)

        # Return the first action from the plan
        if self.plan:
            return self.plan.pop(0)

        return 'Stop'  # Fallback if search fails

    def bfs_search(self, problem):
        """
        Breadth-First Search: Explores the shallowest nodes first.
        Uses a FIFO queue via deque.popleft().
        Maintains a 'reached' set to operate as a Graph Search.
        """
        queue = deque([(problem.get_start_state(), [])])
        reached = set([problem.get_start_state()])

        while queue:
            state, path = queue.popleft()

            if problem.is_goal_state(state):
                return path

            for next_state, action, cost in problem.get_successors(state):
                if next_state not in reached:
                    reached.add(next_state)
                    queue.append((next_state, path + [action]))

        return []

    def dfs_search(self, problem):
        """
        Depth-First Search: Explores the deepest nodes first.
        Uses a LIFO stack via list.pop().
        Maintains a 'reached' set to operate as a Graph Search.
        """
        stack = [(problem.get_start_state(), [])]
        reached = set()

        while stack:
            state, path = stack.pop()

            if problem.is_goal_state(state):
                return path

            if state not in reached:
                reached.add(state)
                for next_state, action, cost in problem.get_successors(state):
                    if next_state not in reached:
                        stack.append((next_state, path + [action]))

        return []

    def ucs_search(self, problem):
        """
        Uniform Cost Search: Explores nodes ordered by total path cost g(n).
        Uses a Priority Queue via heapq.heappop().
        Maintains a 'reached' dictionary to map states to their lowest known cost.
        """
        pq = []
        count = 0
        heapq.heappush(pq, (0, count, problem.get_start_state(), []))
        reached = {}

        while pq:
            current_cost, _, state, path = heapq.heappop(pq)

            if problem.is_goal_state(state):
                return path

            if state not in reached or current_cost < reached[state]:
                reached[state] = current_cost

                for next_state, action, step_cost in problem.get_successors(state):
                    new_cost = current_cost + step_cost
                    if next_state not in reached or new_cost < reached.get(next_state, float('inf')):
                        count += 1
                        heapq.heappush(
                            pq, (new_cost, count, next_state, path + [action]))

        return []

    def astar_search(self, problem, goal_pos, heuristic_type='manhattan', tile_facts=None):
        """
        A* Search: Explores nodes ordered by f(n) = g(n) + h(n).
        Uses a Priority Queue via heapq.heappop().
        Integrates Knowledge Base to verify logical feasibility of tiles.
        """
        if tile_facts is None:
            tile_facts = {}

        pq = []
        count = 0
        start_state = problem.get_start_state()

        # Ensure states are tuples for dictionary hashing
        start_state = tuple(start_state)
        goal_pos = tuple(goal_pos)

        # Calculate initial h(n)
        h_cost = (self.manhattan_distance(start_state, goal_pos) if heuristic_type == 'manhattan'
                  else self.euclidean_distance(start_state, goal_pos))

        # Priority queue tuple: (f_cost, g_cost, count, state, path)
        heapq.heappush(pq, (h_cost, 0, count, start_state, []))
        reached = {}

        while pq:
            f_cost, current_cost, _, state, path = heapq.heappop(pq)
            state_tuple = tuple(state)

            if problem.is_goal_state(state_tuple):
                return path

            if state_tuple not in reached or current_cost < reached[state_tuple]:
                reached[state_tuple] = current_cost

                for next_state, action, step_cost in problem.get_successors(state):
                    next_tuple = tuple(next_state)

                    # --- LOGICAL FEASIBILITY CHECK ---
                    self.kb.clear_facts()
                    # Feed that tile's facts into the KB
                    for fact in tile_facts.get(next_tuple, []):
                        self.kb.tell_fact(fact)
                    
                    self.kb.forward_chain()
                    
                    # If 'Retreat' is deduced, mark the tile as Infeasible
                    if 'Retreat' in self.kb.facts:
                        continue
                    # ---------------------------------

                    new_g = current_cost + step_cost
                    h_val = (self.manhattan_distance(next_tuple, goal_pos) if heuristic_type == 'manhattan'
                             else self.euclidean_distance(next_tuple, goal_pos))
                    new_f = new_g + h_val

                    if next_tuple not in reached or new_g < reached.get(next_tuple, float('inf')):
                        count += 1
                        heapq.heappush(
                            pq, (new_f, new_g, count, next_tuple, path + [action]))

        return []

    def manhattan_distance(self, pos, goal):
        """Calculates Manhattan distance between pos and goal."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """Calculates Euclidean (straight-line) distance between pos and goal."""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)