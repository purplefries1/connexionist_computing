import numpy as np
from MLP import MLP
import math

def generate_examples():
    examples_set = []
    
    inputs = np.random.uniform(-1, 1, (500, 4))
    coeffs = np.array([1, -1, 1, -1])
    training_set = [(inputs[i], np.array([np.sin(inputs[i] @ coeffs)])) for i in range(500)]

    return training_set[:400], training_set[400:]

def sin_problem():

    training_set, testing_set = generate_examples()

    lst_hidden_units = [5, 6, 7, 10, 12]
    lst_epochs = [100, 1000, 3000, 10000]
    lst_learning_rate = [0.05, 0.1, 0.3, 0.5, 1]


    hidden_type = "T" # function for hidden units (T: tanh, S: sigmoid)
    output_type = "L" # function for output units (T: tanh, S: Sigmoid, L: Linear)

    global_results = []

    with open("logs/SIN_problem/SIN_init_no_sqrt.txt", "w") as f:

        def log(msg=""):
            f.write(msg + "\n")
            print(msg)


        for hu in lst_hidden_units:

            for lr in lst_learning_rate:

                for e in lst_epochs:

                    mlp = MLP(NI=4, NH=hu, NO=1, hidden_type=hidden_type, output_type=output_type)

                    log("---------------------------------------------------------\n")
                    log(f"Hidden units: {hu}")
                    log(f"Epochs: {e}")
                    log(f"Learning rate: {lr}\n")

                    log("Errors during training:")
                    mlp.training(max_epochs=e, training_set=training_set, learning_rate=lr, log=log, testing_set=testing_set)

                    test_error = tests(testing_set, mlp, log)
                    global_results.append({
                        "hidden_units": hu,
                        "learning_rate": lr,
                        "epochs": e,
                        "test_error": test_error
                    })



            log("=========================================================\n")


        log("\n==================== GLOBAL SUMMARY ====================\n")

        # sort from best to worst test_error
        global_results.sort(key=lambda d: d["test_error"])

        for res in global_results:
            log(
                f"HU={res['hidden_units']} | LR={res['learning_rate']} | "
                f"Epochs={res['epochs']} | Test Error={res['test_error']}"
            )


def tests(testing_set, mlp:MLP, log):

    test_error = 0 
    log("\n\nOutputs after training:\n")


    for x, t in testing_set:

        mlp.forward(x)
        out = float(mlp.o[0])
        target = float(t[0])
        test_error += 0.5 * (out - target)**2

        log(f"Input: {x} | Target: {target} | Output: {out}")


    test_error /= len(testing_set)

    print(f"\nTest Error: {test_error}\n")



    print("\n\n")

    return test_error





sin_problem()