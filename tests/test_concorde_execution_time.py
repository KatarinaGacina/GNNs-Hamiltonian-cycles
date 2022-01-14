from tqdm import tqdm
import platform
import subprocess
import itertools
import pandas

from src.GraphGenerators import ErdosRenyiGenerator
from src.ExactSolvers import ConcordeHamiltonSolver


def test_concorde_execution_time():
    graph_sizes = [int((1.5)**k) for k in range(10, 25)]
    examples_per_size = 10
    ham_existence_prob = 0.8
    print(f"Checking concorde execution time on Erdos-Renyi graphs with {ham_existence_prob} asysmptotic probability of being Hamiltonian."
          f"Timing execution {examples_per_size} times per each graph_size in {graph_sizes}")
    if platform.system() == "Linux":
        processor_model_name = subprocess.check_output('lscpu | grep "Model name"', shell=True, text=True).split(":")[-1].strip()
    else:
        processor_model_name = "unknown"
    print(f"Processor model: {processor_model_name} (Concorde is a single-thread program)")
    concorde_solver = ConcordeHamiltonSolver(working_subdir="test_execution_time")
    
    _sizes_column = []
    _real_time_column = []
    _user_time_column = []
    _is_hamiltonian_column = []
    for size in tqdm(graph_sizes):
        generator = ErdosRenyiGenerator(size, ham_existence_prob)
        for graph in tqdm(itertools.islice(generator, examples_per_size), leave=False, total=examples_per_size):
            solution, real_time, user_time = concorde_solver.time_execution(graph)
            _sizes_column.append(size)
            _real_time_column.append(real_time)
            _user_time_column.append(user_time)
            _is_hamiltonian_column.append(len(solution) == (size + 1))
            
    df_times = pandas.DataFrame({"graph_size": _sizes_column, "real_execution_time": _real_time_column,
                                 "user_execution_time": _user_time_column, "is_hamiltonian": _is_hamiltonian_column})
    return df_times

    
if __name__ == "__main__":
    df_times = test_concorde_execution_time()
    df_times.to_csv("tests/concorde_execution_times.csv")
    print(df_times.groupby("graph_size").agg("mean"))
