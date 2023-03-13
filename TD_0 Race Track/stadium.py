import os
import numpy as np
import tkinter
import reinforcement as re


class World:
    """class to construct the tracks from the datafiles"""

    def __init__(self, learning_mode):
        self.learning_mode = learning_mode
        self.tracks = []
        file_path = os.path.realpath(__file__)  # get the path for where the program is being executed
        self.root = os.path.dirname(file_path)  # get the directory for the application
        self.data = self.root + "\\tracks"
        self.build_tracks(learning_mode=learning_mode)  # "q" or "sarsa"

    def __str__(self):
        string = "%s world" % self.learning_mode
        return string

    def build_tracks(self, learning_mode="sarsa"):
        """used to build one track from each datafile"""
        track_files = os.listdir(self.data)
        for track in track_files:
            track_path = self.data + "\\" + track
            this_track = Track(track_path, track, learning_mode=learning_mode)
            self.tracks.append(this_track)


class Track:
    """the track that the car drives on"""

    def __init__(self, source, name, learning_mode="q", goal=100):
        self.name = name.strip(".txt")
        self.source = source
        self.matrix = []
        self.height = None
        self.width = None
        self.goal = goal
        self.starts = []
        self.finish = []
        try:
            with open(self.source, 'r') as data:  # read text file
                for y, line in enumerate(data):
                    if y > 0:
                        line = line.strip("\n")
                        # print(line)
                        row = []
                        for x, char in enumerate(line):
                            tile = self.lay_tile(char, x, self.height - y)
                            row.append(tile)
                        self.matrix.append(row)
                    else:
                        line = line.split(",")
                        self.height = int(line[0])
                        self.width = int(line[1])
        except IOError:  # if there is no .txt file
            print("Unable to import data.")
        data.close()
        self.table = re.Table(self)  # constructs the q-table
        self.car = Car(self, learning_mode) # adds a car

    def __str__(self):
        """prints the track in the console"""
        track = ""
        for row in self.matrix:
            line = ""
            for tile in row:
                line = line + tile.char
            line = line + "\n"
            track += line
        return track

    def lay_tile(self, char, x, y):
        """converts the read character to a tile"""
        if char == "#":
            tile = Out(x, y)
        elif char == ".":
            tile = In(x, y)
        elif char == "S":
            tile = Start(x, y)
            self.starts.append(tile)
        elif char == "F":
            tile = Finish(x, y, self.goal)
            self.finish.append(tile)
        return tile

    def get_tile(self, x, y):
        """returns the tile at x, y"""
        tile = self.matrix[self.height - 1 - y][x]
        return tile

    def get_tile_by_id(self, tile_id):
        """selects a tile by its id"""
        selected = None
        for line in self.matrix:
            for tile in line:
                if tile.id == tile_id:
                    selected = tile
        return selected

    def render(self, mult=20):
        """used to display the track / car as it completes laps"""
        window = tkinter.Tk()
        window.title(self.name)
        mult = mult
        height = self.height * mult
        width = self.width * mult
        canvas = tkinter.Canvas(window, bg="white", height=height, width=width)
        y = height
        for row in self.matrix:
            x = 0
            for tile in row:
                color = tile.get_color()
                canvas.create_rectangle(x, y, x + mult, y - mult, fill=color)
                x += mult
            y = y - mult
        try:
            racer = canvas.create_rectangle(self.car.position.x * mult, self.car.position.y * mult,
                                            (self.car.position.x * mult) + mult, (self.car.position.y * mult) + mult,
                                            fill="yellow")
        except AttributeError:
            self.car.seek_starting_line()
            racer = canvas.create_rectangle(self.car.position.x * mult, self.car.position.y * mult,
                                            (self.car.position.x * mult) + mult, (self.car.position.y * mult) + mult,
                                            fill="yellow")
        canvas.pack()
        return window, canvas, racer


class Tile:

    """class to construct tiles"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = None
        self.reward = None
        self.char = None
        self.course = None
        self.color = None
        self.id = None

    def __str__(self):
        return self.char

    def get_color(self):
        return self.color


class In(Tile):

    """regular course tiles reward = -1"""

    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "in"
        self.course = True
        self.char = "."
        self.color = "white"
        self.id = self.char + "x" + str(self.x) + "y" + str(self.y)
        self.reward = -1


class Out(Tile):

    """a wall tile, produces crashes and reward -10"""

    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "out"
        self.course = False
        self.char = "#"
        self.color = "black"
        self.id = self.char + "x" + str(self.x) + "y" + str(self.y)
        self.reward = -10


class Start(Tile):

    """used as a starting position on a track, counts as in course"""

    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "start"
        self.course = True
        self.char = "S"
        self.color = "green"
        self.id = self.char + "x" + str(self.x) + "y" + str(self.y)
        self.reward = -1


class Finish(Tile):

    """the goal tiles, finishes upon reaching, reward = 100"""

    def __init__(self, x, y, reward=100):
        super().__init__(x, y)
        self.type = "finish"
        self.course = False
        self.reward = reward
        self.char = "F"
        self.color = "purple"
        self.id = self.char + "x" + str(self.x) + "y" + str(self.y)


class Car:

    """car to handle the driving / physics of getting around the course"""

    def __init__(self, track, mode):
        self.difficult = False  # True = hard crash mode
        self.new_driver = False  # True = Random starting position
        self.position = None
        self.x_speed = 0
        self.y_speed = 0
        self.x_acc = 0
        self.y_acc = 0
        self.track = track  # links the car to the track
        self.finished = False
        if mode == "q":
            self.learner = re.Q(self)  # q learner
        elif mode == "sarsa":
            self.learner = re.SARSA(self)  # sarsa learner
        self.state = None
        self.action = None
        self.crashed = False

    def seek_starting_line(self):
        """resets the car to the staring line"""
        if self.difficult and self.new_driver:  # algorithm for random starts
            x_dim = len(self.track.matrix[0])
            y_dim = len(self.track.matrix)
            start_set = False
            while not start_set:
                x = np.random.randint(0, x_dim)
                y = np.random.randint(0, y_dim)
                tile = self.track.get_tile(x, y)
                if tile.course:
                    start = tile
                    start_set = True
        else:  # normal function
            random_start = np.random.randint(0, len(self.track.starts))
            start = self.track.starts[random_start]
        self.position = start
        # print("car at starting position x: %s y: %s on track: %s" % (self.position.x, self.position.y,
        # self.track.name))

    def accelerate(self, action):
        """accelerates 80% of the time based on the input action, speed bound by -5,5"""
        acc = action.split(",")
        failure = 0
        chance = np.random.randint(0, 5)
        self.x_acc = int(acc[0])
        self.y_acc = int(acc[1])
        if chance != failure:
            try_x = self.x_speed + self.x_acc
            try_y = self.y_speed + self.y_acc
            if np.abs(try_x) <= 5:  # speed limit
                self.x_speed = try_x
            else:
                pass
                # print("attempting to exceed x speed limit")
            if np.abs(try_y) <= 5:
                self.y_speed = try_y
            else:
                pass
                # print("attempting to exceed y speed limit")
        else:
            # print("acceleration failed")
            pass

    def set_state_action_pair(self):
        """sets the state and action of the car based on position and acc"""
        self.state = self.learner.table.select_state(self.position.x, self.position.y, self.x_speed, self.y_speed)
        self.action = self.learner.table.select_action(self.x_acc, self.y_acc)

    def update_action(self, action):
        """updates the action passed"""
        self.action = action

    def update_state(self):
        """updates the state by position"""
        self.state = self.learner.table.select_state(self.position.x, self.position.y, self.x_speed, self.y_speed)

    def get_current_state_action_pair_q(self):
        """returns the q value of current state action pair"""
        state_action_q = self.learner.table.fetch(self.state, self.action)
        return state_action_q

    def get_current_speed(self):
        print("speed: x = %s y = %s" % (self.x_speed, self.y_speed))

    def set_position(self, tile):
        """sets the position of the car based on tile beneath it"""
        if tile.course:
            self.position = tile  # stay
            # print("current position: x = %s, y = %s" % (self.position.x, self.position.y))
        else:
            if tile.type != "finish":
                # print("car crashed")
                self.crash()
            else:
                print("reached finish line")
                self.position = tile  # you finished
                self.finished = True
        return self.position

    def crash(self):
        self.x_speed = 0
        self.y_speed = 0
        if self.difficult:
            self.seek_starting_line()
        self.crashed = True

    def move(self):
        """algorithm to move based on current x,y speed one tile at a time"""
        moved_x = 0
        moved_y = 0
        new_x = self.position.x
        new_y = self.position.y
        old_position = self.position
        # print("starting at: x = %s, y = %s" % (self.position.x, self.position.y))
        # print("moving x = %s, y = %s" % (self.x_speed, self.y_speed))
        # print("to position x = %s y = %s" % (self.position.x + self.x_speed, self.position.y + self.y_speed))
        if self.x_speed == 0 or self.y_speed == 0:
            if self.x_speed != 0:  # move only x
                while np.abs(moved_x) < self.x_speed and not self.finished:
                    new_x = new_x + (np.abs(self.x_speed) / self.x_speed)
                    moved_x = moved_x + (np.abs(self.x_speed) / self.x_speed)
                    new_tile = self.track.get_tile(int(new_x), int(new_y))
                    self.set_position(new_tile)
            elif self.y_speed != 0:  # move only y
                while np.abs(moved_y) < self.y_speed and not self.finished:
                    new_y = new_y + (np.abs(self.y_speed) / self.y_speed)
                    moved_y = moved_y + (np.abs(self.y_speed) / self.y_speed)
                    new_tile = self.track.get_tile(int(new_x), int(new_y))
                    self.set_position(new_tile)
        else:  # move both x and y
            vector = self.y_speed / self.x_speed
            while (np.abs(moved_y) < np.abs(self.y_speed) or np.abs(moved_x) < np.abs(
                    self.x_speed)) and not self.finished:
                if np.abs(vector) < 1:
                    new_x = new_x + (np.abs(self.x_speed) / self.x_speed)
                    moved_x = moved_x + (np.abs(self.x_speed) / self.x_speed)
                    new_y = new_y + vector
                    moved_y = moved_y + vector
                elif np.abs(vector) > 1 and not self.finished:
                    vector_transformed = self.x_speed / self.y_speed
                    new_y = new_y + (np.abs(self.y_speed) / self.y_speed)
                    moved_y = moved_y + (np.abs(self.y_speed) / self.y_speed)
                    new_x = new_x + vector_transformed
                    moved_x = moved_x + vector_transformed
                elif np.abs(vector) == 1 and not self.finished:
                    new_x = new_x + (np.abs(self.x_speed) / self.x_speed)
                    moved_x = moved_x + (np.abs(self.x_speed) / self.x_speed)
                    new_y = new_y + (np.abs(self.y_speed) / self.y_speed)
                    moved_y = moved_y + (np.abs(self.y_speed) / self.y_speed)
                new_tile = self.track.get_tile(int(new_x), int(new_y))
                self.set_position(new_tile)
        # print("moved to: x = %s, y = %s" % (self.position.x, self.position.y))
        return self.position, old_position

    def drive_course(self, window, canvas, mult=20, animate=True):
        """drives the course one time"""
        fo_action = None  # for sarsa learners
        iterations = 0
        while not self.finished:
            self.crashed = False
            self.learner.reward = 0
            self.update_state()
            if self.learner.predetermined:  # sarsa
                initial_action = fo_action  # do the action you already selected
                self.learner.predetermined = False
            else:
                initial_action = self.learner.determine_action(self)  # find the best action based on epsilon
            self.update_action(initial_action)  # take the action
            self.learner.q1 = self.get_current_state_action_pair_q()  # get the q value of current
            self.learner.store_state_action(self, initial=True)
            self.accelerate(initial_action)  # try to accelerate
            current, old = self.move()
            if animate:
                canvas.create_rectangle(old.x * mult, old.y * mult,
                                        (old.x * mult) + mult, (old.y * mult) + mult, fill="blue")
                canvas.create_rectangle(self.position.x * mult, self.position.y * mult,
                                        (self.position.x * mult) + mult, (self.position.y * mult) + mult, fill="yellow")
            self.learner.reward = self.learner.get_reward(self)  # get immediate reward
            self.update_state()  # observe new state
            if not self.finished:
                if self.difficult and self.new_driver and self.crashed:  # no benefit from crashing on random start
                    fo_q = -1
                    self.learner.predetermined = False
                else:  # determine follow-on action
                    fo_q, fo_state, fo_action = self.learner.determine_follow_on_action()
                    self.learner.follow_on = [fo_state, fo_action]
                    self.learner.q2 = fo_q
            else:
                self.learner.q2 = 0  # terminal state
            self.learner.update_q()  # update q of first state-action pair
            if animate:
                canvas.pack()
                window.update()
            iterations += 1
        return iterations
