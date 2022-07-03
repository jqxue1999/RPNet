import matplotlib.pyplot as plt

targeted = {
    "queries": [100, 200, 300, 400, 500, 1000],
    "base": [56.2, 75, 85.9, 89.8, 91.4, 100],
    "exp1": {
        "describe": "add gaussian noise at output layer",
        "data": {
            "sigma=0.003": [24.2, 33.6, 39.1, 41.4, 42.2, 48.4],
            "sigma=0.009": [25, 30.5, 34.4, 35.9, 35.9, 38.3],
            "sigma=0.03": [15.6, 18, 21.9, 22.7, 24.2, 27.3],
            "sigma=0.09": [12.5, 14.1, 14.8, 14.8, 15.6, 17.2]
        }
    },
    "exp2": {
        "describe": "add gaussian noise at input layer",
        "data": {
            "sigma=0.003": [38.3, 56.2, 64.1, 74.2, 80.5, 100],
            "sigma=0.009": [48.4, 68.8, 82.8, 89.8, 92.2, 100],
            "sigma=0.03": [31.2, 43, 57.8, 65.6, 72.7, 84.4],
            "sigma=0.09": [37.5, 44.5, 47.7, 51.6, 53.1, 64.1]
        }
    },
    "exp3": {
        "describe": "add gaussian noise at input layer and use model trained with gaussian noise",
        "data": {
            "sigma=0.0": [3.1, 9.4, 24.2, 31.2, 38.3, 66.4],
            "sigma=0.003": [1.6, 7, 18, 28.9, 30.5, 59.4],
            "sigma=0.009": [3.1, 9.4, 20.3, 28.1, 34.4, 62.5],
            "sigma=0.03": [1.6, 5.5, 7, 10.2, 10.9, 18.8],
            "sigma=0.09": [7, 8.6, 10.2, 10.2, 10.2, 12.5]
        }
    }
}

untargeted = {
    "queries": [100, 200, 300, 400, 500, 1000],
    "base": [97.7, 100, 100, 100, 100, 100],
    "exp1": {
        "describe": "add gaussian noise at output layer",
        "data": {
            "sigma=0.003": [74.2, 77.3, 81.2, 82.8, 86.7, 100],
            "sigma=0.009": [70.3, 75, 76.6, 77.3, 80.5, 85.2],
            "sigma=0.03": [58.6, 64.8, 69.5, 69.5, 71.1, 78.1],
            "sigma=0.09": [50.8, 57, 60.2, 66.4, 67.2, 71.1]
        }
    },
    "exp2": {
        "describe": "add gaussian noise at input layer",
        "data": {
            "sigma=0.003": [97.7, 100, 100, 100, 100, 100],
            "sigma=0.009": [97.7, 100, 100, 100, 100, 100],
            "sigma=0.03": [100, 100, 100, 100, 100, 100],
            "sigma=0.09": [100, 100, 100, 100, 100, 100]
        }
    },
    "exp3": {
        "describe": "add gaussian noise at input layer and use model trained with gaussian noise",
        "data": {
            "sigma=0.0": [30.5, 40.6, 45.3, 50, 53.1, 60.2],
            "sigma=0.003": [30.5, 39.1, 42.2, 46.9, 50, 57.8],
            "sigma=0.009": [25.8, 39.1, 43.8, 46.9, 48.4, 58.6],
            "sigma=0.03": [23.4, 29.7, 34.4, 40.6, 41.4, 49.2],
            "sigma=0.09": [32, 35.2, 35.9, 38.3, 39.8, 46.1]
        }
    }
}

acc = {
    "sigmas": [0.0, 0.003, 0.009, 0.03, 0.09],
    "exp1": [74.7, 74.2, 74.1, 74.1, 74],
    "exp2": [74.73, 74.95, 75.15, 72.64, 33.62],
    "exp3": [75.38, 75.54, 75.41, 75.68, 73.7]
}

exp1_targeted = {
    "queries": [100, 200, 300, 400, 500, 1000],
    "data": {
        "0.0": [66.4, 93, 100, 100, 100, 100],
        "0.003": [24.2, 33.6, 39.1, 41.4, 42.2, 48.4],
        "0.009": [25, 30.5, 34.4, 35.9, 35.9, 38.3],
        "0.03": [15.6, 18, 21.9, 22.7, 24.2, 27.3],
        "0.09": [12.5, 14.1, 14.8, 14.8, 15.6, 17.2],
        "0.2": [14.1, 20.3, 22.7, 25, 28.1, 32.8]
    }
}

exp1_untargeted = {
    "queries": [100, 200, 300, 400, 500, 1000],
    "data": {
        "0.0": [66.4, 93, 100, 100, 100, 100],
        "0.003": [74.2, 77.3, 81.2, 82.8, 86.7, 100],
        "0.009": [70.3, 75, 76.6, 77.3, 80.5, 85.2],
        "0.03": [58.6, 64.8, 69.5, 69.5, 71.1, 78.1],
        "0.09": [50.8, 57, 60.2, 66.4, 67.2, 71.1],
        "0.16": [42.2, 48.4, 50, 53.9, 56.2, 63.3],
        "0.19": [44.5, 50, 57.8, 61.7, 63.3, 80.5],
        "0.2": [50.8, 65.6, 75.8, 83.6, 89.1, 100],
        "0.25": [100, 100, 100, 100, 100, 100]
    }
}
dev_name = ["exp1", "exp2", "exp3"]


def show_dev_sccessful_rate(success_rate, accuracy, savedir):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 5), dpi=120)

    for idx, exp in enumerate(dev_name):
        dev_data = success_rate.get(exp)
        ax[idx].plot(success_rate.get("queries"), success_rate.get("base"), "k--", label="without defends")
        for sigma in dev_data.get("data").keys():
            ax[idx].plot(success_rate.get("queries"), dev_data.get("data").get(sigma), label=sigma, marker=".")
        ax[idx].legend()
        ax[idx].set_xticks(success_rate.get("queries"))
        ax[idx].set_xlabel("queries")
        ax[idx].set_ylim(0, 110)
        ax[idx].set_ylabel("attack successful rate")
        ax[idx].grid(True)
        ax[idx].set_title(exp)

    for exp in dev_name:
        y = accuracy.get(exp)
        ax[3].plot(accuracy.get("sigmas"), y, label=exp, marker=".")
    ax[3].legend()
    ax[3].grid(True)
    ax[3].set_title("accuracy on clean images")
    ax[3].set_xlabel("sigma")
    ax[3].set_ylabel("accuracy")

    fig.savefig(savedir)


def show_exp1(exp, savedir):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=120)

    for sigma in exp.get("data").keys():
        ax[0].plot(exp.get("queries"), exp.get("data").get(sigma), label="sigma={}".format(sigma), marker=".")
    ax[0].legend()
    ax[0].set_xticks(exp.get("queries"))
    ax[0].grid(True)
    ax[0].set_title("attack successful rate")
    ax[0].set_xlabel("queries")
    ax[0].set_ylabel("attack successful rate")

    for idx, query_num in enumerate(exp.get("queries")):
        ax[1].plot(exp.get("data").keys(), [x[idx] for x in exp.get("data").values()],
                   label="queries={}".format(query_num), marker=".")

    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title("attack successful rate with different sigma (queries=500)")
    ax[1].set_xlabel("sigma")
    ax[1].set_ylabel("attack successful rate")
    fig.savefig(savedir)


if __name__ == "__main__":
    # show_dev_sccessful_rate(targeted, acc, "./image/targeted.png")
    # show_dev_sccessful_rate(untargeted, acc, "./image/untargeted.png")
    # show_exp1(exp1_targeted, "./image/exp1_targeted.png")
    show_exp1(exp1_untargeted, "./image/exp1_untargeted.png")
