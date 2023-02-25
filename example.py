from __future__ import annotations
from problem import Problem, Plan, Task, State, apply_effects
from dataclasses import dataclass
from itertools import count
from heapq import heappush, heappop

@dataclass
class Tree:
    ''' Represents a state in the search space, with the information how it was 
        reached. '''
    parent: Tree
    task: Task
    state: State
    cost: 0

class PriorityQueue(object):
    ''' A queue that sorts items by priorities, breaking ties with insertion order. '''
    def __init__(self, priority=lambda x:x):
        self.priority = priority
        # The counter creates unique ids for the items to break ties. 
        # An increasing counter means sticking to insertion order. 
        self._count = count()
        # The list used as heap for heapq
        self._data = list()

    def __len__(self):
        return len(self._data)

    def push(self, item):
        prioritized_item = (self.priority(item), next(self._count), item)
        heappush(self._data, prioritized_item)

    def pop(self):
        return heappop(self._data)[2]

def plan_breadth_first_search(problem: Problem) -> Plan:
    ''' Performs tasks in a BFS manner until the final state is reached.
        IMPORTANT: This only calculates a correct plan for simple problems, 
        but is incorrect when a box must be relocated multiple times. '''
    queue = PriorityQueue(lambda x: x.cost)
    queue.push(Tree(None, None, problem.state, 0))
    visited = set()
    
    while queue:
        if len(visited) >= 10000:
            print(f"Aborting search after {len(visited)} states.")
            break
        tree: Tree = queue.pop()
        state = tree.state
        # If the state was visited before, ignore it, to prevent search loops.
        if state in visited:
            continue
        # Plan found
        if is_solved(problem, tree):
            return extract_plan(tree);
        # Otherwise, expand the state
        for task in candidate_tasks(problem, state):
            if task.is_satisfied(state):
                new_state = apply_effects(state, [task])
                queue.push(Tree(tree, task, new_state, tree.cost + 1))
        # Mark the state visited
        visited.add(state)
    return None

def is_solved(problem: Problem, tree: Tree):
    ''' Checks if the problem was solved, by comparing the state of all boxes 
        with the expected state after all problem tasks would be executed.
        IMPORTANT: This is incorrect and would only work on problems where 
        the boxes are not relocated multiple times. '''
    return (tree.state.image(problem.boxes) ==
        apply_effects(problem.state, problem.tasks).image(problem.boxes))

def extract_plan(tree: Tree):
    tasks: list[Task] = []
    while tree.task:
        tasks.append(tree.task)
        tree = tree.parent
    tasks.reverse()
    # Now we try to parallelize the plan, by checking what the earliest timestep is
    plan = Plan()
    for task in tasks:
        agent = task.agent
        time = earliest_time_for(task, plan)
        agent_plan = plan.tasks_of(agent)
        # tree now is the root node with the initial state
        node = agent_plan[-1].node if len(agent_plan) > 0 else tree.state[agent]
        if len(agent_plan) > 0:
            node = agent_plan[-1].node
        while len(agent_plan) < time:
            agent_plan.append(Task.wait(agent, node))
        agent_plan.append(task)
    return plan

def earliest_time_for(task: Task, plan: Plan):
    for time in range(plan.length()-1, -1, -1):
        if any_conflict(task, plan.tasks_at(time)):
            return time + 1
    return 0

def any_conflict(task: Task, othertasks: list[Task]):
    for othertask in othertasks:
        if (task.node == othertask.node
            or task.agent == othertask.agent
            or task.box == othertask.box):
            return True
    return False

def candidate_tasks(problem: Problem, state: State):
    for task in problem.tasks:
        yield task
    for agent in problem.agents:
        for node in problem.graph.successors(state[agent]):
            yield Task.move(agent, node)
