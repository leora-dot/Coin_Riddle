import itertools
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from progress.bar import Bar

##QUESTION:
    #GIVEN AN INTEGER, N (2-99), AND AN ARRAY OF INTEGERS (1-99)
    #IN A BAD MONETARY SYSTEM WHERE COINS ONLY COME IN THE DENOMINATIONS OF THE VALUES IN THE ARRAY, HOW CAN YOU MAKE THE VALUE N USING THE LEAST NUMBER OF COINS POSSIBLE?

##LATEST EFFICIENCY STATS (n = 10,000):
    #Actual Combination Reduction: -50.1%

#PARITY FUNCTIONS:

def parity(integer):
    if integer % 2 == 0:
        return "even"
    else:
        return "odd"

def is_all_values_even(input_list):

    if not input_list:
        return None

    for val in input_list:
        if parity(val) == "odd":
            return False
    return True

def is_all_values_odd(input_list):

    if not input_list:
        return None

    for val in input_list:
        if parity(val) == "even":
            return False

    return True

def is_all_values_x(input_list, x):

    for val in input_list:
        if not val == x:
            return False

    return True

#COMBINATION FUNCTIONS

def factorial(val):
    return math.factorial(val)

def num_combinations_replacement(n, k):
    #arr = range(0 , n)
    #validation = len(list(itertools.combinations_with_replacement(arr, k)))
    num_combinations = factorial(n + k - 1) / (factorial(k) * factorial(n-1))

    return num_combinations

class Solver():

    def __init__(self, n, arr):

        self.n = n
        self.input_arr = sorted(arr)
        self.processed_arr = [val for val in self.input_arr] #CREATES A COPY

    def is_arr_empty(self):

        if self.processed_arr:
            is_empty = False
        else:
            is_empty = True

        return is_empty

    def trunc_n_in_arr(self):

        if self.is_arr_empty():
            return

        if self.n in self.processed_arr:
            self.processed_arr = [self.n]

    def trunc_arr_vals_over_n(self):

        if self.is_arr_empty():
            return

        self.processed_arr = arr = [val for val in self.processed_arr if val <= self.n]

    def trunc_arr_duplicates(self):

        if self.is_arr_empty():
            return

        new_arr = []

        for val in self.processed_arr:
            if val not in new_arr:
                new_arr.append(val)

        self.processed_arr = new_arr

    def trunc_arr_parity(self):

        if self.is_arr_empty():
            return

        if (parity(self.n) == "odd") and is_all_values_even(self.processed_arr):
            self.processed_arr = []

    def trunc_arr_max_n_factor(self):

        if self.is_arr_empty():
            return

        #IF THE MAX IS A FACTOR OF N, WE DON'T NEED ANY OTHER NUMBERS
        max_val = max(self.processed_arr)
        if self.n % max_val == 0:
            self.processed_arr = [max_val]

    def trunc_arr_max_n_factor_and_distance(self):

        if self.is_arr_empty():
            return

        while len(self.processed_arr) > 1:
            self.trunc_arr_max_n_factor()

            if len(self.processed_arr) == 1: #IF MAX_FACTOR WAS FOUND
                return

            max_val = max(self.processed_arr)
            min_val = min(self.processed_arr)

            dist_from_n = self.n - max_val

            if dist_from_n < min_val:
                self.processed_arr = self.processed_arr[:-1]

            else:
                break

    def trunc_arr_over_50(self):

        if self.is_arr_empty():
            return

        if (min(self.processed_arr) >= 50) and (self.n not in self.processed_arr):
            self.processed_arr = []

    def optimize_arr(self):

        #METHODS WHERE ORDER DOESN'T MATTER
        self.trunc_n_in_arr() #SINGLE VALUE
        self.trunc_arr_vals_over_n() #REDUCES ARRAY SIZE
        self.trunc_arr_over_50() #REJECTS UNSOLVABLE
        self.trunc_arr_duplicates() #REDUCES ARRAY SIZE

        #METHODS THAT WORK BEST AFTER OTHER TRUNCATIONS HAVE BEEN APPLIED
        self.trunc_arr_max_n_factor_and_distance() #REDUCES ARRAY SIZE OR SINGLE VALUE
        self.trunc_arr_parity() #REJECTS UNSOLVABLE

class Brute_Force_Solver(Solver):

    def solve_by_brute_force(self, reduce_force = False, track_stats = False):

        #SPECIFIC SET UP

        if reduce_force == False:

            arr = self.input_arr

            max_num_coins_per_combination = math.floor( self.n / min(arr))
            k_values = list(range(1, max_num_coins_per_combination + 1))

        if reduce_force == True:

            self.optimize_arr()

            if self.is_arr_empty():

                if track_stats:
                    self.num_combinations_tested = 0

                return None

            arr = self.processed_arr
            min_num_coins_per_combination = math.ceil( self.n / max(arr))
            max_num_coins_per_combination = math.ceil( self.n / min(arr))

            k_values = list(range(min_num_coins_per_combination, max_num_coins_per_combination +1))

            #THE PARITY FILTER:
                #IF YOU ONLY HAVE ODD NUMBER: ODD NUMBER OF ODDS IS ODD & EVEN NUMBER OF ODDS IS EVEN
                #THEREFORE, IF ALL VALUES IN ARR ARE ODD, YOU CAN FILTER OUT ANY K VALUES THAT DON'T MATCH N'S PARITY

            if k_values and is_all_values_odd(arr):

                n_parity = parity(self.n)
                parity_match_k_values = []

                for k in k_values:
                    if parity(k) == n_parity:
                        parity_match_k_values.append(k)

                k_values = parity_match_k_values

        #GENERIC BRUTE FORCE SOLUTION

        if track_stats:
            self.num_combinations_tested = 0
            self.arr_size = len(arr)
            self.k_values = k_values
            self.num_k_values = len(k_values)

        for k in k_values:

            combinations = itertools.combinations_with_replacement(arr, k)

            for combination in combinations:
                if sum(combination) == self.n:
                    combination = list(combination)
                    combination = sorted(combination)
                    return combination

            if track_stats:
                self.num_combinations_tested +=1

        return None

    def generate_efficiency_stats(self):

        if self.num_combinations_tested == 0:
            num_combinations_max = 0
            num_k_values = 0

        else:

            num_combinations_max = 0
            num_k_values = self.num_k_values

            for k in self.k_values:
                num_combinations = num_combinations_replacement(self.arr_size, k)
                num_combinations_max += num_combinations

        ##ADD VALUES TO DICITONARY

        efficiency_dict = {}
        efficiency_dict["num_combinations_actual"] = self.num_combinations_tested
        efficiency_dict["num_combinations_max"] = num_combinations_max
        efficiency_dict["array_size"] = self.arr_size
        efficiency_dict["num_k_values"] = num_k_values

        return efficiency_dict

class Recursion_Solver(Solver):

    def solve_by_recursion_balancer(self, solution = None, prev_solutions = None, unsolvable = False):

        if unsolvable:
            return None

        #ON FIRST CALL, POPULATE SOLUTION & PREV_SOLUTION
        if (not solution):
            solution = []
            prev_solutions = []
            first_call = True

        else:
            first_call = False

        #EVALUATE GIVEN SOLUTION

        solution_sum = sum(solution)
        solution_distance = self.n - solution_sum

        #BASE CASE, SOLUTION:
        if solution_distance == 0:
            solution = sorted(solution)
            return solution

        #BASE CASE, NO SOLUTION:
        if not first_call: #ON THE FIRST CALL, THEY'RE BOTH EMPTY LISTS, SO WE DON'T CHECK
            is_inf_loop = solution in prev_solutions
            if is_inf_loop:
                return self.solve_by_recursion_balancer(solution = None, prev_solutions = None, unsolvable = True)

        #BEFORE TAKING RECURSIVE STEP, LOG CURRENT POSITION
        prev_solutions.append([x for x in solution])

        if solution_distance >= min(self.processed_arr):
            possible_values = [x for x in self.processed_arr if x <= solution_distance]

            next_val = max(possible_values)
            solution.append(next_val)

        else:

            if solution[0] == min(self.processed_arr):
                self.processed_arr.pop()

                solution = []
                prev_solutions = []

                if not self.processed_arr:
                    unsolvable = True

            else:
                solution = solution[1:]

        #RECURSIVE CALL
        return self.solve_by_recursion_balancer(solution, prev_solutions, unsolvable)

tup_list = [ (88, [3, 9, 20, 23, 35, 56, 61, 74, 87, 95]), (59, [2, 5, 25, 34, 37, 56, 83, 88]), (61, [3, 11, 15, 56, 83, 87, 89]), (58, [2, 4, 20, 22, 28, 32, 40, 45, 53, 99]), (68, [6, 18, 25, 31, 42, 46, 48, 54, 59, 80]), (69, [9, 18, 23, 24, 31, 57, 79, 85, 92]), (83, [2, 14, 24, 57, 61, 73, 77, 88, 92]), (73, [4, 22, 29, 41, 44, 57, 66, 72, 75, 81]), (88, [3, 11, 20, 30, 36, 52, 58, 76]), (70, [3, 7, 15, 15, 56, 58, 88]), (53, [6, 12, 13, 17, 35, 43, 46, 59, 63, 79]), (42, [6, 16, 18, 27, 31, 35, 60, 69, 70, 99]), (52, [11, 20, 21, 24, 25, 26, 30, 57, 64, 75]), (82, [5, 16, 27, 31, 63, 97]), (96, [2, 7, 15]), (54, [2, 5, 29, 36, 43, 78, 87, 96]) ]

#for tup in tup_list:

    #brute = Brute_Force_Solver(tup[0], tup[1])
    #brute_solution = brute.solve_by_brute_force(reduce_force = True)

    #recursive = Recursion_Solver(tup[0], tup[1])
    #recursive_solution = recursive.solve_by_recursion_balancer()

    #print("Problem: {}".format(tup))
    #print("Correct Solution: {}".format(brute_solution))
    #print("Recursive Solution: {}".format(recursive_solution))

class Tester():

    def __init__(self):
        pass

    def generate_array(self, min_size = 3, max_size = 10):

        potential_sizes = list(range(min_size, max_size + 1))
        potential_values = list(range(2, 100))

        arr_size = random.choice(potential_sizes)
        arr = [random.choice(potential_values) for i in range(arr_size)]

        self.arr = sorted(arr)

        return self.arr

    def generate_n(self):
        self.n = random.choice(list(range(1, 100)))

        return self.n

    def test_method(self):

        print("This method should be overwritten.")

    def valid_method(self):

        print("This method should be overwritten")

    def run_tests(self, num_tests):

        print("Comparing Methods...")
        valid_results = []
        invalid_results = []

        bar = Bar("", max=num_tests)

        for i in range(num_tests):
            self.generate_array()
            self.generate_n()
            input_tup = (self.n, self.arr)

            valid_solution = self.valid_method()
            test_solution = self.test_method()

            if test_solution == valid_solution:
                valid_results.append(input_tup)
            else:
                invalid_results.append(input_tup)

            bar.next()

        bar.finish()

        #REPORTING RESULTS

        if not invalid_results:
            print("Test method is valid")
        else:
            print("Test method is invalid in the following {} cases: \n{}".format(str(len(invalid_results)),str(invalid_results)))

class Brute_Force_Tester(Tester):

    def test_method(self):

        solver = Brute_Force_Solver(self.n, self.arr)
        solution = solver.solve_by_brute_force(reduce_force = True)
        return solution

    def valid_method(self):

        solver = Brute_Force_Solver(self.n, self.arr)
        solution = solver.solve_by_brute_force(reduce_force = False)
        return solution

#tester = Brute_Force_Tester()
#tester.run_tests(5)

class Recursion_Tester(Tester):

    def test_method(self):

        solver = Recursion_Solver(self.n, self.arr)
        solution = solver.solve_by_recursion_balancer()
        return solution

    def valid_method(self):

        solver = Brute_Force_Solver(self.n, self.arr)
        solution = solver.solve_by_brute_force(reduce_force = True)
        return solution

#tester = Recursion_Tester()
#tester.run_tests(1000)

class Efficiency_Tester(Tester):

    def __init__(self):
        self.metric_list = ["num_combinations_actual", "num_combinations_max", "num_k_values", "array_size"]

    def is_in_metric_list(self, metric):
        if metric in self.metric_list:
            is_in_list = True
        else:
            is_in_list = False

        return is_in_list

    def generate_efficiency_comparison(self, num_tests, metric):

        if not self.is_in_metric_list(metric):
            print("Invalid metric requested")
            return

        print("Comparing {} efficiency...".format(metric))

        self.metric_unreduced_list = []
        self.metric_reduced_list = []

        ##RUN TESTS

        bar = Bar("", max=num_tests)

        for i in range(num_tests):
            self.generate_n()
            self.generate_array(2, 10)

            solver = Brute_Force_Solver(self.n, self.arr)
            solver.solve_by_brute_force(track_stats = True)
            efficiency_dict = solver.generate_efficiency_stats()
            test_metric = efficiency_dict.get(metric)
            self.metric_unreduced_list.append(test_metric)

            solver.solve_by_brute_force(reduce_force = True, track_stats = True)
            efficiency_dict = solver.generate_efficiency_stats()
            test_metric = efficiency_dict.get(metric)
            self.metric_reduced_list.append(test_metric)

        bar.finish()

        #SOME NUMBERS, I LIKE TO SEE PRINTED

        if metric == "num_k_values":
            self.print_results_as_num_k_values(num_tests)

        if metric == "num_combinations_actual":
            self.print_results_as_num_combinations_actual()

    def print_results_as_num_k_values(self, num_tests):

        unreduced_zero_count = self.metric_unreduced_list.count(0)
        unreduced_one_count = self.metric_unreduced_list.count(1)
        reduced_zero_count = self.metric_reduced_list.count(0)
        reduced_one_count = self.metric_reduced_list.count(1)

        unreduced_zero_perc = unreduced_zero_count / num_tests
        unreduced_one_perc = unreduced_one_count / num_tests
        reduced_zero_perc = reduced_zero_count / num_tests
        reduced_one_count = reduced_one_count / num_tests

        print("""

        Brute Force (Unreduced):
        No testing needed for {:.1%} of problems
        One k tested for {:.1%} of problems

        Brute Force (Reduced):
        No testing needed for {:.1%} of problems
        One k tested for {:.1%} of problems

        """.format(unreduced_zero_perc, unreduced_one_perc, reduced_zero_perc, reduced_one_count))

    def print_results_as_num_combinations_actual(self):

        unreduced_combo = sum(self.metric_unreduced_list)
        reduced_combo = sum(self.metric_reduced_list)
        perc_reduction = (reduced_combo - unreduced_combo) / unreduced_combo

        print("""

        Brute Force (Unreduced):
        {:,} combinations tested

        Brute Force (Reduced):
        {:,} combinations tested

        Overall Reduction: {:.1%}

        """.format(unreduced_combo, reduced_combo, perc_reduction))

    def show_efficiency_comparison(self, num_tests, metric):

        if not self.is_in_metric_list(metric):
            print("Invalid metric requested")
            return

        self.generate_efficiency_comparison(num_tests, metric)

        #FOR num_k_values, I WANT TO REMOVE BINNING SO THAT I CAN DIFFERENTIATE BETWEEN 0 & 1 VALUES
        if metric == "num_k_values":
            bin_count_reduced = max(self.metric_reduced_list) - min(self.metric_reduced_list) + 1
            bin_count_unreduced = max(self.metric_unreduced_list) - min(self.metric_unreduced_list) + 1

        else:
            bin_count_reduced = None
            bin_count_unreduced = None

        with sns.axes_style("white"):

            sns.distplot(self.metric_reduced_list, bins = bin_count_reduced, kde=True, label= "Reduced")
            sns.distplot(self.metric_unreduced_list, bins = bin_count_unreduced, color = "red", kde = True, label = "Unreduced")

            plt.legend()
            plt.xlim(0, None)
            plt.title("{} Distribution by Brute Force Method".format(metric))

        plt.show()

#efficiency_test = Efficiency_Tester()
#efficiency_test.show_efficiency_comparison(100, "num_combinations_actual")
#efficiency_test.show_efficiency_comparison(100, "num_combinations_max")
#efficiency_test.show_efficiency_comparison(10000, "num_k_values")
#efficiency_test.show_efficiency_comparison(100, "array_size")

class Exploratory_Tester(Tester):

    def __init__(self):
        pass

    def solve_array_for_all_n_vals(self, arr = None):

        #GENERATE ARRAY
        if arr:
            self.arr = arr
        else:
            self.generate_array()

        ##TESTING

        self.n_to_is_solution_dict = {}
        self.n_to_solution_dict = {}

        for n in range(2, 100):
            solver = Brute_Force_Solver(n, self.arr)
            solution = solver.solve_by_brute_force(reduce_force = True)

            if solution == None:
                self.n_to_is_solution_dict[n] = False

            else:
                self.n_to_is_solution_dict[n] = True
                self.n_to_solution_dict[n] = solution

    def show_array_solvability(self, arr = None):
        self.solve_array_for_all_n_vals(arr)

        ##GENERATE DF

        df = pd.DataFrame.from_dict(self.n_to_is_solution_dict, orient = "index", columns = ["is_solution"])
        df.reset_index(inplace = True, drop = False)
        df.rename(columns = {"index": "n"}, inplace = True)

        #VISUALIZE STRIPPLOT

        with sns.axes_style("white"):
            sns.stripplot(x="is_solution", y="n", data=df)

            plt.title("Solvability by n for arr: {}".format(str(self.arr)))

        plt.show()

    def generate_array_solutions_df(self, arr = None ):
        self.solve_array_for_all_n_vals(arr)

        df = pd.DataFrame(list(range(2,100)))
        df.rename(columns = {0: "n"}, inplace = True)

        for val in self.arr:
            df[val] = df["n"].map(self.n_to_solution_dict) # EACH COL HOLDS LIST
            df[val] = df[val].apply(lambda x : x.count(val) if type(x) == list else 0)

        self.arr_solutions_df = df

    def show_array_solutions(self, arr = None):
        self.generate_array_solutions_df(arr)

        #SET DF INDEX
        df = self.arr_solutions_df.copy()
        df.set_index("n", inplace = True)

        #DROP NON-SOLUTIONS

        non_zero_series = (df != 0).any(axis=1)
        df = df.loc[non_zero_series]

        print(df)

        #VISUALIZE

        plt.figure(figsize = (16,9.5))

        with sns.axes_style("white"):
            sns.heatmap(df, cmap="Blues")

            plt.title("Solutions Grid")
            plt.xlabel("Array Values")

        plt.show()

explorer = Exploratory_Tester()
#explorer.show_array_solvability()
#explorer.show_array_solutions()

print(95%78)
print(95%74)
print(95%48)
print(95%25)
print(95%14)
print(95%13)
print(95%12)
