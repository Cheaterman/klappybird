from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
)
from kivy.uix.widget import Widget
from toy_nn import NeuralNetwork
import random


POOL_SIZE = 100


class KlappyBirds(App):
    birds = ListProperty()
    dead_birds = ListProperty()
    pipes = ListProperty()
    score = NumericProperty()
    highscore = NumericProperty()

    def build(self):
        self.reset()
        Clock.schedule_interval(self.update, 0)

    def reset(self):
        for pipe in self.pipes[:]:
            self.pipes.remove(pipe)
            self.root.remove_widget(pipe)

        for bird in self.birds[:]:
            self.birds.remove(bird)
            self.root.remove_widget(bird)

        if not self.dead_birds:
            for _ in range(POOL_SIZE):
                bird = Bird()
                self.birds.append(bird)
                self.root.add_widget(bird)
        else:
            self.new_generation()

        self.score = 0
        self.frame_count = 0

    def calculate_fitness(self):
        total_score = sum([bird.score for bird in self.dead_birds])
        for bird in self.dead_birds:
            bird.fitness = bird.score / total_score

    def new_generation(self):
        self.calculate_fitness()

        dead_birds = self.dead_birds
        self.dead_birds = []
        brains = []

        for i in range(POOL_SIZE):
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
            self.root.add_widget(bird)

    def update(self, *args):
        if not self.birds:
            self.reset()

        birds_x = self.birds[0].x  # All birds have same x
        closest_pipe = None

        for pipe in self.pipes[:]:
            if pipe.right < 0:
                self.pipes.remove(pipe)
                self.root.remove_widget(pipe)
                continue

            if pipe.x + pipe.width < birds_x:
                continue
            else:
                closest_pipe = pipe
                break

        for bird in self.birds[:]:
            bird.update()

            if closest_pipe is not None:
                bird.think(closest_pipe)

            for pipe in self.pipes:
                if pipe.collide_widget(bird):
                    Clock.unschedule(bird.update)
                    self.birds.remove(bird)
                    if(
                        self.dead_birds and
                        bird.score > self.dead_birds[-1].score
                    ):
                        if not self.birds:
                            bird.score *= 10
                        elif len(self.birds) < 2:
                            bird.score *= 5
                    self.dead_birds.append(bird)
                    self.root.remove_widget(bird)
                    break
            else:
                if bird.score > self.score:
                    self.score = bird.score
                if bird.score > self.highscore:
                    self.highscore = bird.score

        if self.frame_count % 100 == 0:
            pipe = Pipe(x=self.root.width)
            self.pipes.append(pipe)
            self.root.add_widget(pipe)

        self.frame_count += 1


class Bird(Widget):
    gravity = NumericProperty(.98)
    velocity = NumericProperty(0)
    lift = NumericProperty(25)
    drag = NumericProperty(.05)
    score = NumericProperty()

    def __init__(self, brain=None, **kwargs):
        super(Bird, self).__init__(**kwargs)
        if brain is None:
            self.brain = NeuralNetwork(5, 5, 1)
        else:
            self.brain = brain.copy()
        self.fitness = 0

    def think(self, closest_pipe):
        inputs = [
            self.y / self.parent.height,
            self.velocity / self.parent.height,
            closest_pipe.x / self.parent.width,
            closest_pipe.top_pipe_height / self.parent.height,
            closest_pipe.bottom_pipe_height / self.parent.height,
        ]
        output = self.brain.predict(inputs)[0]
        if output > 0:
            self.up()

    def up(self):
        self.velocity += self.lift

    def update(self, *args):
        self.score += .1

        self.velocity -= self.gravity
        self.velocity *= 1 - self.drag
        self.y += self.velocity

        if self.center_y < 0:
            self.center_y = 0
            self.velocity = 0

        if self.center_y > self.parent.height:
            self.center_y = self.parent.height


class Pipe(Widget):
    top_pipe_height = NumericProperty(1)
    bottom_pipe_height = NumericProperty(0)
    spacing_ratio = NumericProperty(.25)
    spacing = NumericProperty(1)
    speed = NumericProperty(5)

    def __init__(self, **kwargs):
        super(Pipe, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 0)

    def on_size(self, pipe, size):
        height = size[1]
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
