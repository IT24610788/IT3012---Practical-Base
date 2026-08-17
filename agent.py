# agent.py
import random
from collections import deque
import heapq


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

    def bfs_search(self, problem):
        """
        Breadth-First Search: Explores the shallowest nodes first.
        Uses a FIFO queue via deque.popleft().
        Maintains a 'reached' set to operate as a Graph Search.
        """
        # queue stores tuples of (state, path_of_actions)
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
        # stack stores tuples of (state, path_of_actions)
        stack = [(problem.get_start_state(), [])]
        reached = set()

        while stack:
            state, path = stack.pop()

            if problem.is_goal_state(state):
                return path

            # To avoid cycles, only expand if we haven't reached this state
            if state not in reached:
                reached.add(state)
                for next_state, action, cost in problem.get_successors(state):
                    # Minor optimization to prevent pushing obviously visited nodes
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
        # Push (cost, tie_breaker, state, path)
        # The tie_breaker ensures we don't accidentally try to compare 'state' objects if costs tie
        count = 0
        heapq.heappush(pq, (0, count, problem.get_start_state(), []))

        # Dictionary acts as our 'reached' set but also stores the best cost to reach a state
        reached = {}

        while pq:
            current_cost, _, state, path = heapq.heappop(pq)

            if problem.is_goal_state(state):
                return path

            # If this state is unvisited, or we found a strictly cheaper way here
            if state not in reached or current_cost < reached[state]:
                reached[state] = current_cost

                for next_state, action, step_cost in problem.get_successors(state):
                    new_cost = current_cost + step_cost

                    # Only push to heap if it's unvisited or we found a cheaper path to it
                    if next_state not in reached or new_cost < reached.get(next_state, float('inf')):
                        count += 1
                        heapq.heappush(
                            pq, (new_cost, count, next_state, path + [action]))

        return []
