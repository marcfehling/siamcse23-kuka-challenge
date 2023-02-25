'''
This module defines classes to represent and work with intralogistic problems.
'''
from __future__ import absolute_import
import networkx
import yaml
import os
from networkx.classes.digraph import DiGraph
from enum import Enum
from _collections_abc import Iterable
from operator import xor
from functools import reduce
from itertools import combinations
from numpy.core.fromnumeric import clip
from collections import Counter
from networkx.algorithms.shortest_paths.astar import astar_path
from numpy.linalg.linalg import norm
from _collections import defaultdict


class State:
    ''' This class maps agents and boxes to their location. It also includes 
        the graph to check the precondition of move tasks uniformly. '''
    def __init__(self, graph: DiGraph, agents: set[str], boxes: set[str], mapping: dict):
        self.graph = graph
        self.agents = agents
        self.boxes = boxes
        self.nodes = set(graph.nodes())
        self._mapping = mapping

    def __getitem__(self, key):
        return self._mapping.get(key, None)
    
    def __setitem__(self, key, value):
        if not key in self.agents and not key in self.boxes: 
            raise KeyError(key)
        self._mapping[key] = value
    
    def __hash__(self):
        return reduce(xor, map(hash, self._mapping.items()), 0)
        
    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return self._mapping == other._mapping
        return False
    
    def __str__(self) -> str:
        return str(self._mapping)
    
    def __repr__(self) -> str:
        return repr(self._mapping)
    
    def clone(self):
        ''' Creates a clone whose mapping can me modified independently of this 
            state. The graph, agents and boxes are shared. '''
        return State(self.graph, self.agents, self.boxes, self._mapping.copy())
    
    def domain(self) -> set:
        ''' Returns the domain of this state mapping. '''
        return set(self._mapping.keys())
    
    def image(self, subdomain: Iterable = None) -> set:
        ''' Returns the image (the value range) of the state mapping, maybe 
            restricted to the given subdomain.'''
        if subdomain:
            return set([self[k] for k in subdomain])
        return set(self._mapping.values())

    def location_of_box(self, box: str) -> str | None:
        ''' Returns the location of the given box in this state, which can be 
            an agent, a node or nowhere. '''
        return self[box]
    
    def node_of_agent(self, agent: str) -> str | None:
        ''' Returns the node where the given agent is in this state. '''
        return self[agent]

    def agents_on_node(self, node: str) -> set[str]:
        ''' Returns all agents at the given node. If the returned sets size is 
            greater than 1, there is a conflict. '''
        return {a for a in self.agents if self[a] == node}

    def boxes_on_node(self, node: str) -> set[str]:
        ''' Returns all boxes on the given node. If the returned sets size is 
            greater than 1, there is a conflict. '''
        return {b for b in self.boxes if self[b] == node}
    
    def boxes_on_agent(self, agent: str) -> set[str]:
        ''' Returns all boxes on the given agent. If the returned sets size is 
            greater than 1, there is a conflict. '''
        return {b for b in self.boxes if self[b] == agent}

    def has_box(self, agentOrNode):
        return any(self[b] == agentOrNode for b in self.boxes)


class TaskType(Enum):
    
    MOVE = 'move'
    PICK = 'pick'
    DROP = 'drop'
    LOAD = 'load'
    UNLOAD = 'unload'
    WAIT = 'wait'
    
    @classmethod
    def _missing_(cls, value):
        ''' Implements an ignore-case lookup '''
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class Task:
    def __init__(self, typ: TaskType, args: tuple[str]):
        self.type = typ
        self._args = args
    
    def __repr__(self):
        return '(' + self.type.value + ',' + ','.join(self._args) + ')'
    
    def __hash__(self):
        return hash(self.type) ^ hash(self._args)
    
    def __eq__(self, other):
        if isinstance(other, Task):
            return self.type == other.type and self._args == other._args
        return False
    
    @property
    def agent(self) -> str:
        ''' Returns the agent this task is assigned to. '''
        return self._args[0]
    
    @property
    def node(self) -> str:
        ''' Returns the node of this task. For moves, it is the target node. 
            For pick, drop, load, unload and wait tasks, it is the node where
            the agent currently is. '''
        return self._args[1]
    
    @property
    def box(self) -> str | None:
        ''' Returns the box affected by this task. Move and wait tasks don't 
            affect boxes. '''
        return self._args[2] if self.type not in {TaskType.MOVE, TaskType.WAIT} else None
    
    def to_list(self) -> list[str]:
        ''' Represents this task as list, for example [move, a1, n3] '''
        return [self.type.value] + [*self._args]
    
    @staticmethod
    def from_list(array: list[str]):
        ''' Constructs a Task from a list representation. The first element is 
            the task type, the others are the arguments. '''
        return Task(TaskType(array[0]), tuple(array[1:]))
    
    @staticmethod
    def move(agent:str, node:str):
        ''' Constructs a move task with the given arguments. '''
        return Task(TaskType.MOVE, (agent, node))

    @staticmethod
    def wait(agent:str, node:str):
        ''' Constructs a wait task with the given arguments. '''
        return Task(TaskType.WAIT, (agent, node))
    
    def is_satisfied(self, state: State) -> bool:
        ''' Checks if the precondition of this task is satisfied in the given 
            state. '''
        match self.type:
            case TaskType.MOVE:
                return (state.graph.has_successor(state[self.agent], self.node)
                    and not state.agents_on_node(self.node)
                    and not (state.boxes_on_node(self.node) 
                             and state.boxes_on_agent(self.agent)))
            case TaskType.PICK:
                return (state[self.agent] == self.node
                    and state[self.box] == self.node
                    and not state.boxes_on_agent(self.agent))
            case TaskType.DROP:
                return (state[self.agent] == self.node
                    and state[self.box] == self.agent
                    and not state.boxes_on_node(self.node))
            case TaskType.LOAD:
                return (state[self.agent] == self.node
                    and state[self.box] == None
                    and not state.boxes_on_agent(self.agent))
            case TaskType.UNLOAD:
                return (state[self.agent] == self.node
                    and state[self.box] == self.agent)
            case TaskType.WAIT:
                return state[self.agent] == self.node


    def apply_effect(self, state: State):
        ''' Changes the given state by applying this  
            state. '''
        match self.type:
            case TaskType.MOVE:
                state[self.agent] = self.node
            case TaskType.PICK:
                state[self.box] = self.agent
            case TaskType.DROP:
                state[self.box] = self.node
            case TaskType.LOAD:
                state[self.box] = self.agent
            case TaskType.UNLOAD:
                state[self.box] = None

    
def apply_effects(state: State, tasks: Iterable):
    state = state.clone()
    for t in tasks:
        t.apply_effect(state)
    return state


class Problem:
    def __init__(self, name: str, graph: DiGraph, agents: set[str], boxes: set[str], state: dict, tasks: list[Task]):
        self.name = name
        self.graph = graph
        # Fill the state dict with all domain objects and override with the configured state
        state = {x:None for x in agents | boxes} | state
        self.state = State(graph, agents, boxes, state)
        ''' Returns the initial state of this problem. '''
        self.tasks = tuple(tasks)
        ''' Returns the tasks of this problem. '''

    @property
    def nodes(self):
        ''' Returns the set of nodes of this problem. '''
        return self.state.nodes
    
    @property
    def agents(self):
        ''' Returns the set of agents of this problem. '''
        return self.state.agents
    
    @property
    def boxes(self):
        ''' Returns the set of boxes of this problem. '''
        return self.state.boxes

    def tasks_of(self, agent: str):
        ''' Returns the tasks for the given agent. The ordering of the 
            sublist is consistent with the total task list. '''
        return [t for t in self.tasks if t.agent == agent]
    
    def __str__(self):
        return 'Nodes: ' + str(self.nodes) + '\n' + \
            'Agents: ' + str(self.agents) + '\n' + \
            'Boxes: ' + str(self.boxes) + '\n' + \
            'State: ' + str(self.state) + '\n' + \
            'Tasks: ' + ', '.join(map(str,self.tasks))


class Plan:
    def __init__(self, agent_plans: defaultdict[str, list[Task]] = None):
        self._plans: defaultdict = agent_plans or defaultdict(list)
    
    def __str__(self):
        tasks_to_line = lambda t: ', '.join(map(str,t))
        return '\n\t'.join(map(tasks_to_line, self._tasks))
    
    def __iter__(self):
        ''' Iterates over the timesteps, returning the tasks of all agents per 
            timestep. '''
        return iter([self.tasks_at(t) for t in range(0, self.length())])
    
    def tasks_at(self, time: int) -> tuple[Task]:
        ''' Returns the tasks of all agents at the given timestep. '''
        return (p[time] for p in self._plans.values() if time < len(p))
    
    def tasks_of(self, agent: str) -> list[Task]:
        ''' Returns the agentplan for the given agent. '''
        return self._plans[agent]
    
    def agent_plans(self) -> defaultdict[str, list[Task]]:
        ''' Returns the plans of all agents. '''
        return self._plans
    
    def length(self):
        ''' Returns the length of the longest agent plan '''
        return max([len(p) for p in self._plans.values()], default=0)


def load_problem(directory: str) -> Problem | None:
    ''' Loads a Problem from the given directory. Returns None if the directory 
        contains no problem files. '''
    graphpath = os.path.join(directory, 'graph.xml')
    problempath = os.path.join(directory, 'problem.yaml')
    if not os.path.exists(graphpath) or not os.path.exists(problempath):
        return None
    graph: DiGraph = networkx.read_graphml(graphpath)
    with open(problempath, 'r') as stream:
        content = yaml.safe_load(stream)
    name = os.path.basename(directory)
    agents = set(content['agents'])
    boxes = set(content['boxes'])
    state = dict(content['state'])
    tasks = [Task.from_list(c) for c in content['tasks']]
    return Problem(name, graph, agents, boxes, state, tasks)

def load_plan(directory: str) -> Plan:
    ''' Loads the plan from the given directory. Returns None if the directory 
        contains no plan file. '''
    planpath = os.path.join(directory, 'plan.yaml')
    if not os.path.exists(planpath):
        return Plan()
    with open(planpath, 'r') as stream:
        loaded_plan = yaml.safe_load(stream)
    plan = Plan()
    for a,p in loaded_plan.items():
        plan.tasks_of(a).extend([Task.from_list(t) for t in p])
    return plan

def save_plan(directory: str, plan: Plan):
    ''' Saves the plan to the given directory. '''
    def task_representer(dumper, task):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', task.to_list(), flow_style=True)
    yaml.add_representer(Task, task_representer)
    planpath = os.path.join(directory, 'plan.yaml')
    agent_plans = {a:[*p] for a,p in plan.agent_plans().items()}
    with open(planpath, 'w') as stream:
        yaml.dump(agent_plans, stream)


class Solution:
    ''' This class evaluates plans for conflicts, unsatisfied tasks, correctness
        and score. '''
    def __init__(self, problem: Problem, plan: Plan):
        self.problem = problem
        self.plan = plan
        self._states = [self.problem.state]
        state = self._states[0]
        for tasks in self.plan:
            state = apply_effects(state, tasks)
            self._states.append(state)
    
    def state(self, time) -> State:
        ''' Returns the State at the given point in time. state(0) returns the 
            initial state of the problem. '''
        time = clip(time, 0, len(self._states)-1)
        return self._states[time]
    
    def unknown_task_args(self):
        ''' Returns a (time, task, arg) tuple for each task that uses an unknown
            argument. '''
        result = []
        for time in range(0, self.plan.length()):
            tasks = self.plan.tasks_at(time)
            conflicts = [(time, t, t.agent) for t in tasks if t.agent not in self.problem.agents] \
                    + [(time, t, t.node) for t in tasks if t.node not in self.problem.nodes] \
                    + [(time, t, t.box) for t in tasks if t.box and t.box not in self.problem.boxes]
            result.extend(conflicts)
        return result
    
    def unsuitable_tasks(self):
        ''' Returns a (time, task, agent) tuple for each task that is given to 
            the wrong agent. '''
        result = []
        for agent in self.problem.agents:
            agent_plan = self.plan.tasks_of(agent)
            for (index, task) in enumerate(agent_plan):
                if task.agent != agent:
                    result.append((index, task, agent))
        return result
    
    def unsatisfied_tasks(self):
        ''' Returns a (time, task) tuple for each unsatisfied task in all 
            timesteps. '''
        result = []
        for time in range(0, self.plan.length()):
            tasks = self.plan.tasks_at(time)
            unsat = [(time, t) for t in tasks if not t.is_satisfied(self.state(time))]
            result.extend(unsat)
        return result 
    
    def task_conflicts(self):
        ''' Returns a (time, task, task) tuple for each pair of conflicting 
            tasks in all timesteps. '''
        have_conflict = lambda x,y: x.agent == y.agent or x.node == y.node
        result = []
        for time in range(0, self.plan.length()):
            tasks = self.plan.tasks_at(time)
            conflicts = [(time, t1, t2) for t1,t2 in combinations(tasks, 2) if have_conflict(t1, t2)]
            result.extend(conflicts)
        return result
    
    def missing_tasks(self) -> list[Task]:
        ''' Returns all tasks from the problem that are not present in the plan.
        '''
        result = []
        for agent in self.problem.agents:
            goaltasks = Counter(self.problem.tasks_of(agent))
            plantasks = Counter(self.plan.tasks_of(agent))
            missing = goaltasks - plantasks
            result.extend(missing)
        return result
    
    def illegal_tasks(self):
        ''' Returns a (time, task) tuple for each task that was not part of the 
            problem or was planned in a wrong order. '''
        result = []
        for agent in self.problem.agents:
            agent_tasks = self.problem.tasks_of(agent)
            agent_plan = self.plan.tasks_of(agent)
            current_index = 0
            current_task = agent_tasks[current_index] if agent_tasks else None
            for (index, task) in enumerate(agent_plan):
                if task is None:
                    continue  
                elif task == current_task:
                    current_index += 1
                    current_task = agent_tasks[current_index] if current_index < len(agent_tasks) else None
                elif task.type not in [TaskType.MOVE, TaskType.WAIT]:
                    result.append((index, task))
                
        return result
    
    def issues(self):
        ''' Checks whether there are any conflicts, unsatisfied tasks or other 
            issues with this the plan. '''
        return (self.unknown_task_args()
            or self.unsuitable_tasks()
            or self.unsatisfied_tasks() 
            or self.task_conflicts()
            or self.missing_tasks() 
            or self.illegal_tasks())
    
    def generateReport(self):
        ''' Generates a textual report about issues of the plan. '''
        text = ''
        issues = self.unknown_task_args()
        if issues:
            text += f'Found {len(issues)} unknown task arguments.\n' + \
                    f'  First at time {issues[0][0]}: {issues[0][1]}: arg "{issues[0][2]}"\n'
        issues = self.unsuitable_tasks()
        if issues:
            text += f'Found {len(issues)} unsuitable tasks.\n' + \
                    f'  First at time {issues[0][0]}: {issues[0][1]} for agent "{issues[0][2]}"\n'
        issues = self.task_conflicts()
        if issues:
            text += f'Found {len(issues)} task conflicts.\n' + \
                    f'  First at time {issues[0][0]}: {issues[0][1]} # {issues[0][2]}\n'
        issues = self.unsatisfied_tasks()
        if issues:
            text += f'Found {len(issues)} unsatisfied tasks.\n' + \
                    f'  First at time {issues[0][0]}: {issues[0][1]}\n'
        issues = self.missing_tasks()
        if issues:
            text += f'Found {len(issues)} missing tasks:\n' + \
                    f'  {issues}\n'
        issues = self.illegal_tasks()
        if issues:
            text += f'Found {len(issues)} illegal tasks:\n' + \
                    f'  First at time {issues[0][0]}: {issues[0][1]}\n'
        return text
    
    def score(self) -> float:
        ''' Calculates the plan score. Higher is better. '''
        if self.issues():
            return 0
        absolute_score = self._score(self.plan)
        if absolute_score == 0:
            return 1000
        base_score = self._score(plan_conflicting(self.problem))
        return (base_score/absolute_score) * 1000

    def _score(self, plan: Plan):
        return sum(self._completion_time(plan, a) + self._task_count(plan, a) for a in self.problem.agents)

    def _completion_time(self, plan: Plan, agent):
        agent_tasks = self.problem.tasks_of(agent)
        agent_plan = plan.tasks_of(agent)
        total = 0
        index = 0
        for task in agent_tasks:
            # Tasks can occur more than once. Be sure to get the one at the next index.
            index = agent_plan.index(task, index)
            total += index 
        return total

    def _task_count(self, plan: Plan, agent):
        return len([t for t in plan.tasks_of(agent) if t and t.type != TaskType.WAIT])


def plan_conflicting(problem: Problem) -> Plan:
    ''' Calculates a plan where each vehicle moves as if no other vehicles and boxes are present. '''
    agent_plans = defaultdict(list)
    for agent in problem.agents:
        current_node = problem.state[agent]
        for goaltask in problem.tasks_of(agent):
            new_tasks = plan_task_conflicting(problem.graph, current_node, goaltask)
            agent_plans[agent].extend(new_tasks)
            current_node = goaltask.node
    return Plan(agent_plans)


def plan_task_conflicting(graph: DiGraph, source: str, task: Task) -> list[Task] | None:
    ''' Plans a single task by moving to the task's node, ignoring conflicts with other agents and boxes. '''
    path = astar_path(graph, source, task.node, heuristic=euclidean_distance(graph))
    # path[0] is the source node
    plan = [Task.move(task.agent, p) for p in path[1:]]
    # For non-move tasks, we are not done by just moving to the target node, 
    # but have to perform the final task there 
    if task.type != TaskType.MOVE or task.type != TaskType.WAIT:
        plan.append(task)
    return plan


def euclidean_distance(graph: DiGraph):
    def distance(a, b):
        a = graph.nodes[a]
        b = graph.nodes[b]
        return norm((a['row']-b['row'], a['col']-b['col']))
    return distance

