import matplotlib.pyplot as plt

dev_name = ["exp1", "exp2", "exp3", "exp4", "exp5"]
queries = [100, 200, 300, 400, 500, 1000]

data = {
    "exp1": {
        "sigma=0.0": {
            "targeted": {
                "epsilon=0.5": [7.8, 13.3, 15.6, 22.7, 27.3, 46.9],
                "epsilon=0.8": [29.7, 50.0, 57.8, 67.2, 73.4, 90.6],
                "epsilon=1.0": [58.6, 76.6, 84.4, 88.3, 92.2, 100],
                "epsilon=1.3": [80.5, 92.2, 97.7, 100, 100, 100],
                "epsilon=1.5": [83.6, 94.5, 96.9, 100, 100, 100],
                "epsilon=1.8": [91.4, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [26.6, 43.8, 60.2, 68.0, 75.8, 85.2],
                "epsilon=0.8": [90.6, 93.8, 94.5, 95.3, 100, 100],
                "epsilon=1.0": [97.7, 100, 100, 100, 100, 100],
                "epsilon=1.3": [100, 100, 100, 100, 100, 100],
                "epsilon=1.5": [100, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        },
        "sigma=0.009": {
            "targeted": {
                "epsilon=0.5": [7.8, 10.2, 11.7, 13.3, 13.3, 16.4],
                "epsilon=0.8": [15.6, 21.9, 25, 27.3, 28.1, 34.4],
                "epsilon=1.0": [27.3, 31.2, 33.6, 39.1, 45.3, 53.9],
                "epsilon=1.3": [45.3, 53.1, 60.9, 68.8, 72.7, 81.2],
                "epsilon=1.5": [55.5, 68, 75, 80.5, 86.7, 100],
                "epsilon=1.8": [77.3, 88.3, 91.4, 93, 93.8, 100],
                "epsilon=2.0": [77.3, 85.9, 93, 95.3, 95.3, 100]
            },
            "untargeted": {
                "epsilon=0.5": [29.7, 32.8, 38.3, 43, 46.9, 54.7],
                "epsilon=0.8": [54.7, 62.5, 64.1, 64.8, 68.8, 72.7],
                "epsilon=1.0": [74.2, 82.8, 85.2, 86.7, 87.5, 100],
                "epsilon=1.3": [86.7, 92.2, 97.7, 100, 100, 100],
                "epsilon=1.5": [97.7, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        }
    },
    "exp2": {
        "sigma=0.0": {
            "targeted": {
                "epsilon=0.5": [7.8, 13.3, 15.6, 22.7, 27.3, 46.9],
                "epsilon=0.8": [29.7, 50.0, 57.8, 67.2, 73.4, 90.6],
                "epsilon=1.0": [58.6, 76.6, 84.4, 88.3, 92.2, 100],
                "epsilon=1.3": [80.5, 92.2, 97.7, 100, 100, 100],
                "epsilon=1.5": [83.6, 94.5, 96.9, 100, 100, 100],
                "epsilon=1.8": [91.4, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [26.6, 43.8, 60.2, 68.0, 75.8, 85.2],
                "epsilon=0.8": [90.6, 93.8, 94.5, 95.3, 100, 100],
                "epsilon=1.0": [97.7, 100, 100, 100, 100, 100],
                "epsilon=1.3": [100, 100, 100, 100, 100, 100],
                "epsilon=1.5": [100, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        },
        "sigma=0.009": {
            "targeted": {
                "epsilon=0.5": [2.3, 3.9, 10.2, 12.5, 18.8, 29.7],
                "epsilon=0.8": [20.3, 36.7, 50.8, 68, 74.2, 78.9],
                "epsilon=1.0": [41.4, 64.8, 72.7, 81.2, 85.9, 90.6],
                "epsilon=1.3": [73.4, 85.9, 92.2, 97.7, 100, 100],
                "epsilon=1.5": [86.7, 96.1, 96.9, 100, 100, 100],
                "epsilon=1.8": [88.3, 97.7, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [32.8, 47.7, 57, 64.8, 68.8, 85.2],
                "epsilon=0.8": [89.8, 95.3, 95.3, 100, 100, 100],
                "epsilon=1.0": [97.7, 100, 100, 100, 100, 100],
                "epsilon=1.3": [100, 100, 100, 100, 100, 100],
                "epsilon=1.5": [100, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        }
    },
    "exp3": {
        "sigma=0.0": {
            "targeted": {
                "epsilon=0.5": [1.6, 1.6, 1.6, 2.3, 2.3, 2.3],
                "epsilon=0.8": [2.3, 2.3, 3.1, 5.5, 6.3, 18.8],
                "epsilon=1.0": [3.1, 9.5, 24.2, 31.2, 38.3, 66.4],
                "epsilon=1.3": [13.3, 43, 69.5, 82.8, 92.2, 100],
                "epsilon=1.5": [30.4, 74.2, 93.8, 98.4, 100, 100],
                "epsilon=1.8": [74.2, 97, 98.4, 100, 100, 100],
                "epsilon=2.0": [95.3, 96.9, 100, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [17.2, 18.8, 19.5, 20.3, 20.3, 21.1],
                "epsilon=0.8": [21.9, 26.6, 30.5, 32.8, 35.2, 42.2],
                "epsilon=1.0": [28.1, 37.5, 43, 47.7, 49.2, 54.6],
                "epsilon=1.3": [46.1, 59.4, 62.5, 63.3, 65.6, 66.4],
                "epsilon=1.5": [55.5, 63.3, 63.3, 64.8, 66.4, 70.3],
                "epsilon=1.8": [64.8, 71.1, 71.9, 72.7, 72.7, 75],
                "epsilon=2.0": [70.3, 73.4, 76.6, 77.3, 77.3, 80.5]
            }
        },
        "sigma=0.009": {
            "targeted": {
                "epsilon=0.5": [2.3, 2.3, 2.3, 2.3, 2.3, 3.1],
                "epsilon=0.8": [2.3, 4.7, 8.6, 10.9, 13.3, 31.2],
                "epsilon=1.0": [3.9, 13.3, 28.1, 39.1, 46.1, 84.4],
                "epsilon=1.3": [23.4, 57, 77.3, 95.3, 96.1, 100],
                "epsilon=1.5": [43.8, 89.1, 96.1, 96.1, 100, 100],
                "epsilon=1.8": [82, 96.9, 96.9, 100, 100, 100],
                "epsilon=2.0": [96.9, 96.9, 100, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [14.8, 15.6, 18.8, 19.5, 19.5, 20.3],
                "epsilon=0.8": [17.2, 24.2, 28.9, 32, 33.6, 43],
                "epsilon=1.0": [26.6, 39.8, 43, 47.7, 49.2, 62.5],
                "epsilon=1.3": [49.2, 60.2, 69.5, 71.1, 71.1, 72.7],
                "epsilon=1.5": [61.7, 68.8, 71.1, 72.7, 73.4, 75],
                "epsilon=1.8": [71.9, 77.3, 77.3, 77.3, 77.3, 78.9],
                "epsilon=2.0": [77.3, 78.9, 80.5, 81.2, 81.2, 85.2]
            }
        }
    },
    "exp4": {
        "sigma=0.0": {
            "targeted": {
                "epsilon=0.5": [2.3, 2.3, 2.3, 2.3, 3.1, 3.1],
                "epsilon=0.8": [4.7, 4.7, 6.2, 6.2, 6.2, 6.2],
                "epsilon=1.0": [5.5, 6.2, 7, 7, 7, 7],
                "epsilon=1.3": [8.6, 8.6, 10.2, 10.2, 10.2, 10.2],
                "epsilon=1.5": [7, 7.8, 8.6, 9.4, 10.2, 11.7],
                "epsilon=1.8": [10.2, 12.5, 12.5, 13.3, 13.3, 14.1],
                "epsilon=2.0": [10.2, 13.3, 14.1, 14.8, 14.8, 18.8]
            },
            "untargeted": {
                "epsilon=0.5": [15.6, 17.2, 17.2, 18, 18, 21.1],
                "epsilon=0.8": [19.5, 25.8, 26.6, 26.6, 26.6, 28.9],
                "epsilon=1.0": [28.1, 30.5, 32, 32.8, 33.6, 34.4],
                "epsilon=1.3": [32.8, 36.7, 36.7, 36.7, 36.7, 39.1],
                "epsilon=1.5": [36.7, 40.6, 40.6, 41.4, 41.4, 44.5],
                "epsilon=1.8": [41.4, 44.5, 46.9, 46.9, 48.4, 50],
                "epsilon=2.0": [46.9, 54.7, 55.5, 55.5, 57.8, 58.6]
            }
        },
        "sigma=0.009": {
            "targeted": {
                "epsilon=0.5": [1.6, 1.6, 1.6, 1.6, 1.6, 2.3],
                "epsilon=0.8": [1.6, 1.6, 2.3, 2.3, 2.3, 3.1],
                "epsilon=1.0": [3.1, 3.9, 3.9, 3.9, 3.9, 4.7],
                "epsilon=1.3": [3.9, 6.2, 7.8, 7.8, 8.6, 8.6],
                "epsilon=1.5": [7, 7.8, 9.4, 9.4, 9.4, 9.4],
                "epsilon=1.8": [8.6, 10.2, 10.9, 11.7, 11.7, 13.3],
                "epsilon=2.0": [10.2, 12.5, 15.6, 15.6, 15.6, 17.2]
            },
            "untargeted": {
                "epsilon=0.5": [13.3, 14.8, 14.8, 14.8, 15.6, 17.2],
                "epsilon=0.8": [17.2, 20.3, 23.4, 23.4, 24.2, 27.3],
                "epsilon=1.0": [22.7, 28.9, 31.2, 33.6, 33.6, 39.1],
                "epsilon=1.3": [32, 35.2, 35.2, 35.9, 35.9, 39.1],
                "epsilon=1.5": [36.7, 38.3, 39.1, 39.1, 39.1, 41.4],
                "epsilon=1.8": [42.2, 45.3, 46.1, 47.7, 47.7, 50.8],
                "epsilon=2.0": [46.1, 48.4, 50.8, 50.8, 51.6, 56.2]
            }
        }
    },
    "exp5": {
        "sigma=0.0": {
            "targeted": {
                "epsilon=0.5": [5.5, 7, 8.6, 8.6, 9.4, 17.2],
                "epsilon=0.8": [16.4, 21.9, 31.2, 33.6, 39.1, 50],
                "epsilon=1.0": [24.2, 37.5, 45.3, 53.9, 56.2, 73.4],
                "epsilon=1.3": [47.7, 70.3, 74.2, 78.9, 84.4, 88.3],
                "epsilon=1.5": [67.2, 80.5, 85.9, 89.1, 90.6, 100],
                "epsilon=1.8": [74.2, 87.5, 92.2, 93.8, 94.5, 100],
                "epsilon=2.0": [86.7, 91.4, 94.5, 96.9, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [38.3, 47.7, 56.2, 67.2, 73.4, 85.2],
                "epsilon=0.8": [92.2, 97.7, 100, 100, 100, 100],
                "epsilon=1.0": [99.2, 100, 100, 100, 100, 100],
                "epsilon=1.3": [100, 100, 100, 100, 100, 100],
                "epsilon=1.5": [100, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        },
        "sigma=0.009": {
            "targeted": {
                "epsilon=0.5": [4.7, 6.2, 6.2, 6.2, 6.2, 12.5],
                "epsilon=0.8": [13.3, 19.5, 28.9, 34.4, 42.2, 57],
                "epsilon=1.0": [24.2, 41.4, 47.7, 53.1, 57.8, 70.3],
                "epsilon=1.3": [49.2, 66.4, 79.7, 86.7, 87.5, 100],
                "epsilon=1.5": [54.7, 78.9, 92.2, 94.5, 96.9, 100],
                "epsilon=1.8": [76.6, 89.1, 96.1, 96.9, 100, 100],
                "epsilon=2.0": [87.5, 94.5, 96.9, 100, 100, 100]
            },
            "untargeted": {
                "epsilon=0.5": [34.4, 43, 52.3, 58.6, 63.3, 81.2],
                "epsilon=0.8": [90.6, 96.9, 100, 100, 100, 100],
                "epsilon=1.0": [100, 100, 100, 100, 100, 100],
                "epsilon=1.3": [100, 100, 100, 100, 100, 100],
                "epsilon=1.5": [100, 100, 100, 100, 100, 100],
                "epsilon=1.8": [100, 100, 100, 100, 100, 100],
                "epsilon=2.0": [100, 100, 100, 100, 100, 100]
            }
        }
    }
}


def show_dev_sccessful_rate(success_rate, savedir, targeted):
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(22, 5), dpi=120)

    for idx, exp in enumerate(data.keys()):
        dev_dict = success_rate.get(exp)
        # ax[idx].plot(queries, success_rate.get("sigma=0.009").get("targeted"), "k--", label="without defends")
        for epsilon in dev_dict.get("sigma=0.009").get(targeted):
            ax[idx].plot(queries, dev_dict.get("sigma=0.009").get(targeted).get(epsilon), label=epsilon, marker=".")
        ax[idx].legend()
        ax[idx].set_xticks(queries)
        ax[idx].set_xlabel("queries")
        ax[0].set_ylabel("attack successful rate")
        ax[idx].grid(True)
        ax[idx].set_title(exp)

    fig.savefig(savedir)


def show_exp(success_rate, exp, savedir, targeted):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=120)

    dev_dict = success_rate.get(exp)
    for idx, sigma in enumerate(dev_dict.keys()):
        for epsilon in dev_dict.get(sigma).get(targeted).keys():
            ax[idx].plot(queries, dev_dict.get(sigma).get(targeted).get(epsilon), label=epsilon, marker=".")
        ax[idx].legend()
        ax[idx].set_xticks(queries)
        ax[idx].grid(True)
        ax[idx].set_title("attack model with different epsilon, {}".format(sigma))
        ax[idx].set_xlabel("queries")
    ax[0].set_ylabel("attack successful rate")

    fig.savefig(savedir)


if __name__ == "__main__":
    show_dev_sccessful_rate(data, "./image/fix sigma/all_untargeted.png", "untargeted")
    show_dev_sccessful_rate(data, "./image/fix sigma/all_targeted.png", "targeted")
    # show_exp(data, "exp1", "./image/fix sigma/exp1_targeted", "targeted")
    # show_exp(data, "exp2", "./image/fix sigma/exp2_targeted", "targeted")
    # show_exp(data, "exp3", "./image/fix sigma/exp3_targeted", "targeted")
    # show_exp(data, "exp4", "./image/fix sigma/exp4_targeted", "targeted")
    # show_exp(data, "exp5", "./image/fix sigma/exp5_targeted", "targeted")

    # show_exp(data, "exp1", "./image/fix sigma/exp1_untargeted", "untargeted")
    # show_exp(data, "exp2", "./image/fix sigma/exp2_untargeted", "untargeted")
    # show_exp(data, "exp3", "./image/fix sigma/exp3_untargeted", "untargeted")
    # show_exp(data, "exp4", "./image/fix sigma/exp4_untargeted", "untargeted")
    # show_exp(data, "exp5", "./image/fix sigma/exp5_untargeted", "untargeted")
