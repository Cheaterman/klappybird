from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.widget import Widget
from toy_nn import NeuralNetwork
import datetime
import json
import os
import random


POOL_SIZE = 100
PIPE_SPAWN_INITIAL_SPEED = 150


class KlappyBirds(App):
    birds = ListProperty()
    dead_birds = ListProperty()
    pipes = ListProperty()
    score = NumericProperty()
    generations = NumericProperty()
    highscore = NumericProperty()
    update_speed = NumericProperty(1)
    update_limit = NumericProperty()
    limit_speed = BooleanProperty()
    start_time = ObjectProperty()
    time_since_start = StringProperty()
    best_brain = ObjectProperty()
    mode = StringProperty('train')

    def build(self):
        self.game_area = self.root.ids.game_area
        Clock.schedule_interval(self.update_quick, 0)
        self.start_time = datetime.datetime.now()
        self.pipe_spawn_speed = 0
        self.reset()

    def reset(self):
        for pipe in self.pipes[:]:
            self.pipes.remove(pipe)
            self.game_area.remove_widget(pipe)

        for bird in self.birds[:]:
            self.birds.remove(bird)
            self.game_area.remove_widget(bird)

        if self.mode == 'best' and self.best_brain:
            bird = Bird(self.best_brain)
            self.birds.append(bird)
            self.game_area.add_widget(bird)
        elif not self.dead_birds:
            if self.best_brain:  # Add at least one best bird
                bird = Bird(self.best_brain)
                self.birds.append(bird)
                self.game_area.add_widget(bird)
            for _ in range(POOL_SIZE - len(self.birds)):
                bird = Bird()
                self.birds.append(bird)
                self.game_area.add_widget(bird)
        else:
            self.new_generation()

        self.score = 0
        self.frame_count = 0
        self.pipe_spawn_time = 0
        if self.pipe_spawn_speed:
            self.pipe_spawn_speed /= 2

    def calculate_fitness(self):
        scores = [bird.score for bird in self.dead_birds]
        max_score = max(scores)
        if max_score >= self.highscore:
            self.best_brain = self.dead_birds[scores.index(max_score)].brain
        total_score = sum([score ** 2 for score in scores])
        for bird in self.dead_birds:
            bird.fitness = bird.score ** 2 / total_score

    def new_generation(self):
        self.calculate_fitness()

        dead_birds = self.dead_birds
        self.dead_birds = []
        brains = []

        for _ in range(POOL_SIZE):
            selected = -1
            selector = random.random()
            while selector > 0:
                selector -= dead_birds[selected].fitness
                selected += 1
            brains.append(dead_birds[selected].brain)

        for brain in brains:
            bird = Bird(brain)
            bird.brain.mutate(.01)
            self.birds.append(bird)
            self.game_area.add_widget(bird)

        self.generations += 1

    def update_quick(self, *args):
        for speed in range(int(self.update_speed)):
            self.update()
            if self.limit_speed and Clock.time() - Clock.get_time() >= 1 / 15.:
                self.update_limit = speed
                break
        else:
            self.update_limit = 0

        timedelta = datetime.datetime.now() - self.start_time
        hours, rest = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        self.time_since_start = '{}{:02d}:{:02d}:{:02d}.{:02d}'.format(
            '' if not timedelta.days else '{}d '.format(timedelta.days),
            hours, minutes, seconds, timedelta.microseconds // 10 ** 4,
        )

    def update(self, *args):
        birds_x = self.birds[0].x  # All birds have same x
        closest_pipe = None

        for pipe in self.pipes[:]:
            pipe.update()

            if pipe.right < 0:
                self.pipes.remove(pipe)
                self.game_area.remove_widget(pipe)
                continue

        for pipe in self.pipes[:]:
            if pipe.x + pipe.width < birds_x:
                continue
            else:
                closest_pipe = pipe
                break

        for bird in self.birds[:]:
            if bird.center_y < 0:
                self.kill(bird)
                continue

            for pipe in self.pipes:
                if pipe.collide_widget(bird):
                    self.kill(bird)
                    dead = True
                    break
            else:
                dead = False
            if dead:
                continue

            bird.update()

            if closest_pipe is not None:
                bird.think(closest_pipe)

        if not self.birds:
            self.reset()

        self.score += .1

        if self.score > self.highscore:
            self.highscore = self.score

        if int(self.score * 10) % 1000 == 0:
            self.pipe_spawn_speed += 5

        self.pipe_spawn_time -= 1

        if self.pipe_spawn_time <= 0:
            pipe = Pipe(x=self.game_area.right)
            self.pipes.append(pipe)
            self.game_area.add_widget(pipe)
            self.pipe_spawn_time = (
                PIPE_SPAWN_INITIAL_SPEED - self.pipe_spawn_speed
            )

        self.frame_count += 1

    def kill(self, bird):
        self.birds.remove(bird)
        self.dead_birds.append(bird)
        self.game_area.remove_widget(bird)
        bird.score = self.score

    def serialize_best(self):
        if not self.best_brain:
            return
        with open('best_bird.json', 'w') as json_file:
            json_file.write(self.best_brain.serialize())

    def deserialize_best(self):
        if not os.path.exists('best_bird.json'):
            return
        with open('best_bird.json') as json_file:
            self.best_brain = NeuralNetwork.deserialize(json.load(json_file))


class Bird(Widget):
    gravity = NumericProperty(.98)
    velocity = NumericProperty(0)
    lift = NumericProperty(25)
    drag = NumericProperty(.05)

    def __init__(self, brain=None, **kwargs):
        super(Bird, self).__init__(**kwargs)
        if brain is None:
            self.brain = NeuralNetwork(5, 5, 1)
        else:
            self.brain = brain.copy()
        self.fitness = 0

    def think(self, closest_pipe):
        parent = self.parent
        inputs = [
            (self.y / parent.height) - .5,
            self.velocity / self.lift / 10,
            (closest_pipe.x / parent.width),
            (closest_pipe.top_pipe_height / parent.height) * 2 - 1,
            (closest_pipe.bottom_pipe_height / parent.height) * 2 - 1,
        ]
        output = self.brain.predict(inputs)[0]
        if output > 0:
            self.up()

    def up(self):
        self.velocity += self.lift

    def update(self, *args):
        self.velocity -= self.gravity
        self.velocity *= 1 - self.drag
        self.y += self.velocity


class Pipe(Widget):
    top_pipe_height = NumericProperty(1)
    bottom_pipe_height = NumericProperty(0)
    spacing_ratio = NumericProperty(.25)
    spacing = NumericProperty(1)
    speed = NumericProperty(5)

    def on_parent(self, pipe, parent):
        if not parent:
            return
        height = self.parent.height
        self.spacing = spacing = height * self.spacing_ratio
        self.top_pipe_height = (
            height - random.random() * (height - spacing)
        )
        self.bottom_pipe_height = self.top_pipe_height - spacing

    def update(self, *args):
        self.x -= self.speed

    def collide_widget(self, widget):
        if(
            widget.right < self.x or
            widget.x > self.right or (
                widget.y > self.bottom_pipe_height and
                widget.top < self.top_pipe_height
            )
        ):
            return False
        return True


app = KlappyBirds()


if __name__ == '__main__':
    app.run()
