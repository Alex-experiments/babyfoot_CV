import json
import os

import matplotlib.pyplot as plt

SAVE_FOLDER = os.path.join("stats")


def scale(data, factor):
    return [x * factor if x is not None else None for x in data]


def show_stats(file_path: str) -> None:
    # Recover data
    with open(file_path, "r") as openfile:
        dictionary = json.load(openfile)

    fig, ax = plt.subplots()

    # Print global statistics
    print("########   Global statistics   ########\n")
    print(f"Score red: {dictionary['score_red']}")
    print(f"Score blue: {dictionary['score_blue']}")
    print(f"Duration: {dictionary['duration']:.0f} s")
    print(f"Ball maximum speed: {dictionary['ball_max_speed']:.0f} cm/s")
    print(f"Ball total travelled distance: {dictionary['ball_total_distance']:.0f} cm")
    print(f"Possession red: {round(100*dictionary['possession_red'])} %")
    print(f"Possession blue: {round(100*dictionary['possession_blue'])} %")

    # Plot curves
    plt.plot(dictionary["time"], dictionary["ball_speed"], label="Ball Speed (in cm/s)")
    plt.plot(
        dictionary["time"],
        scale(dictionary["ball_acceleration"], 0.1),
        label="Ball Acceleration (in 10x cm/s^2)",
    )
    plt.plot(
        dictionary["time"],
        dictionary["ball_angle"],
        label="Ball Deviation Angle (in °)",
    )

    # Modify background for possession
    ax.fill_between(
        dictionary["time"],
        0,
        1,
        where=[x == "red" for x in dictionary["possession"]],
        color="red",
        alpha=0.2,
        transform=ax.get_xaxis_transform(),
    )
    ax.fill_between(
        dictionary["time"],
        0,
        1,
        where=[x == "blue" for x in dictionary["possession"]],
        color="blue",
        alpha=0.2,
        transform=ax.get_xaxis_transform(),
    )

    # Plot Shots
    red_shots = [
        x["time"]
        for x in dictionary["events"]
        if x["name"] == "Shot" and x["player"]["team"] == "red"
    ]
    plt.scatter(
        red_shots,
        [0] * len(red_shots),
        c="red",
        marker="*",
        edgecolors="none",
        s=250,
        label="Red shots",
    )
    blue_shots = [
        x["time"]
        for x in dictionary["events"]
        if x["name"] == "Shot" and x["player"]["team"] == "blue"
    ]
    plt.scatter(
        blue_shots,
        [0] * len(blue_shots),
        c="blue",
        marker="*",
        edgecolors="none",
        s=250,
        label="Blue shots",
    )

    # Plot Goals
    red_goals = [
        x["time"]
        for x in dictionary["events"]
        if x["name"] == "Goal" and x["team"] == "red"
    ]
    plt.scatter(
        red_goals,
        [0] * len(red_goals),
        c="red",
        marker="D",
        edgecolors="none",
        s=100,
        label="Red goals",
    )
    blue_goals = [
        x["time"]
        for x in dictionary["events"]
        if x["name"] == "Goal" and x["team"] == "blue"
    ]
    plt.scatter(
        blue_goals,
        [0] * len(blue_goals),
        c="blue",
        marker="D",
        edgecolors="none",
        s=100,
        label="Blue goals",
    )

    # Provide plot description
    plt.xlabel("Time (in s)")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Game statistics")

    plt.savefig(os.path.join(SAVE_FOLDER, "stats.jpg"))
    plt.show()


if __name__ == "__main__":
    import os

    file_path = os.path.join("stats", "experiment.json")
    show_stats(file_path)
