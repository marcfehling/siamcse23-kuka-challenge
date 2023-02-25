import os
import sys
import time
from problem import Problem, Plan, load_problem, save_plan, Solution
from problem import plan_conflicting

def plan_problem(problem: Problem) -> Plan:
    ''' Calculates a plan for the given problem. '''
    print('TODO: Calculate a conflict-free plan that completes all problem tasks.')
    print('For demonstration purposes a conflicting plan is created.')
    return plan_conflicting(problem)

if __name__ == '__main__':
    def load_plan_save(problem_dir: str) -> float:
        problem = load_problem(problem_dir)
        if problem:
            print(f"-------- {os.path.basename(problem_dir)} --------")
            start = time.process_time()
            plan = plan_problem(problem) or Plan()
            duration = time.process_time() - start
            save_plan(problem_dir, plan)
            solution = Solution(problem, plan)
            print(solution.generateReport())
            print(f"Duration: {duration:.2f} s")
            print(f"Score: {solution.score()}")
            return solution.score()
        return 0
    
    if len(sys.argv) == 2:
        problem_dir = sys.argv[1]
        if not os.path.isdir(problem_dir):
            raise ValueError(f'Argument "{problem_dir}" is not a directory')
        load_plan_save(problem_dir)
    else:
        total = 0
        for problem_dir in [ f.path for f in os.scandir('problems') if f.is_dir() ]:
            total += load_plan_save(problem_dir)
        print(f"Total score: {total}")
