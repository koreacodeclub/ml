import stadium
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np





def graph_results(track_test_results, width=100):
    """produces the graphs in the paper"""
    for result in track_test_results:
        graph_speed_test(result, width)


def graph_speed_test(test_files, width=100, size=10000):
    """produces individual graph for each experiment"""
    i = 0
    for test in test_files:
        u = 0
        w = width
        df = pd.read_csv(test, index_col=0)
        name = test.split(".")
        name = name[0]
        # title.append(name)
        x = []
        y = []
        while w < size:
            this_y = np.sum(df[u:w]) // 100
            this_x = w
            y.append(this_y)
            x.append(this_x)
            u = w
            w += width
        if i == 0:
            marker = "o"
            c = "blue"
            i += 1
        else:
            marker = "x"
            c = "pink"
        plt.plot(x, y, label=name, color=c, marker=marker)
        plt.ylabel("actions to complete")
        plt.xlabel("laps completed")
    plt.legend()
    plt.show()


def test_drive_world():
    """initial testing of the functionality of the program output is graphical"""
    world = stadium.World("q")
    for track in world.tracks:
        print(track.table.df)
        print(track.car.learner)
        track.car.seek_starting_line()
        window, canvas, racer = track.render()
        track.car.drive_course(window, canvas)


def init_env():
    """initializes the world, returns the q and sarsa envs"""
    q_env = stadium.World("q")
    sarsa_env = stadium.World("sarsa")
    return q_env, sarsa_env


def learn_course(track):
    """completes one iteration of learning the specified track"""
    track.car.seek_starting_line()
    window, canvas, racer = track.render()
    track.car.drive_course(window, canvas)


def learn_course_until(track, total_laps, animate=False, difficult=False, new_driver=False):
    """continues to learn a course until it completes a number of total_laps
    optional ags to animate, change crash difficulty, and randomize starting space(new_driver)"""
    if difficult:
        track.car.difficult = True  # hard crash setting
    if new_driver:
        track.car.new_driver = True  # random starting space
    else:
        track.car.new_driver = False
    if animate:  # output graphical display using tkinter
        window, canvas, racer = track.render()
    else:
        window = None
        canvas = None
    lap_speeds = []
    lap = 0
    while lap < total_laps:
        if not difficult:
            if lap == total_laps // 4:  # changes epsilon value as course is learned
                track.car.learner.epsilon -= 0.15
            elif lap == 2 * total_laps // 4:
                track.car.learner.epsilon -= 0.15
            elif lap == 3 * total_laps // 4:
                track.car.learner.epsilon -= 0.15
        else:
            track.car.learner.epsilon = 0.01  # to try to stop exploration of a learned track on difficult
        track.car.finished = False  # ready
        track.car.seek_starting_line()  # set
        speed = track.car.drive_course(window, canvas, animate=animate)  # go
        lap_speeds.append(speed)  # record
        lap += 1
        print("finished lap: %s in %s iterations" % (lap, speed))
    filename = "%s-learner-%s-%s.csv" % (track.car.learner.mode, track.name, total_laps)
    speeds_file = "%s-lap-speeds.csv" % filename
    speed_df = pd.DataFrame(data=lap_speeds)
    track.table.df.to_csv(filename)  # save q-table
    speed_df.to_csv(speeds_file)  # save results
    return lap_speeds


def learn_world(iterations=10000, complete=False, animate=False):
    """program to produce the full set of experiments in the paper
    set complete to True to run the difficult crash setting test"""
    q, s = init_env()
    dq, ds = init_env()
    learn_course_until(q.tracks[0], iterations, animate=animate)
    learn_course_until(s.tracks[0], iterations, animate=animate)
    learn_course_until(q.tracks[1], iterations, animate=animate)
    learn_course_until(s.tracks[1], iterations, animate=animate)
    learn_course_until(q.tracks[2], iterations, animate=animate)
    learn_course_until(s.tracks[2], iterations, animate=animate)
    if complete:
        learn_course_until(dq.tracks[2], 10000, animate=False, difficult=True)
        learn_course_until(ds.tracks[2], 10000, animate=False, difficult=True)


learn_world(iterations=100, complete=False, animate=True)  # execute
