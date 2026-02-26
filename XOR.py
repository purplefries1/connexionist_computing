import numpy as np
from MLP import MLP




def tests():



    training_set = [
        (np.array([0.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([1.0, 1.0]), np.array([0.0])),
    ]

    lst_hidden_units = [3, 4, 5, 7, 10]
    lst_epochs = [1, 10, 1000, 2000, 10000]
    lst_learning_rate = [0.001, 0.01, 0.1, 1, 2, 5]

    # list to store all results
    global_results = []

    with open("logs/FILE.txt", "w") as f:

        def log(msg=""):
            f.write(msg + "\n")
            print(msg)

        for hu in lst_hidden_units:

            for lr in lst_learning_rate:

                for e in lst_epochs:

                    mlp = MLP(NI=2, NH=hu, NO=1, hidden_type="S", output_type="S")

                    log("---------------------------------------------------------\n")
                    log(f"Hidden units: {hu}")
                    log(f"Epochs: {e}")
                    log(f"Learning rate: {lr}\n")

                    log("Errors during training:")
                    mlp.training(max_epochs=e, training_set=training_set, learning_rate=lr, log=log)

                    log("\nOutputs after training:")
                    correct = 0
                    total = len(training_set)

                    for x, t in training_set:  # final tests

                        mlp.forward(x)
                        out = float(mlp.o[0])
                        target = int(t[0])
                        pred = 1 if out >= 0.5 else 0

                        log(f"Input: {x} | Target: {target} | Output: {out} | Pred: {pred}")

                        if pred == target:
                            correct += 1

                    accuracy = correct / total
                    log(f"\nAccuracy: {accuracy}\n")


                    global_results.append({
                        "hidden_units": hu,
                        "learning_rate": lr,
                        "epochs": e,
                        "accuracy": accuracy
                    })

            log("=========================================================\n")


        log("\n==================== GLOBAL SUMMARY ====================\n")

        # sort from best to worst accuracy
        global_results.sort(key=lambda d: d["accuracy"], reverse=True)

        for res in global_results:
            log(
                f"HU={res['hidden_units']} | LR={res['learning_rate']} | "
                f"Epochs={res['epochs']} | Accuracy={res['accuracy']}"
            )






tests()