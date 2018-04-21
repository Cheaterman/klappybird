from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
)
from kivy.uix.widget import Widget
import random


class KlappyBirds(App):
    bird = ObjectProperty(rebind=True)
    pipes = ListProperty()
    highscore = NumericProperty()

    def build(self):
        self.bird = Bird()
        self.root.add_widget(self.bird)

        self.pipes.append(Pipe(x=Window.width))
        self.root.add_widget(self.pipes[-1])

        self.frame_count = 0
        Clock.schedule_interval(self.update, 0)

    def update(self, *args):
        self.frame_count += 1

        if self.bird.score > self.highscore:
            self.highscore = self.bird.score

        if self.frame_count % 100 == 0:
            self.pipes.append(Pipe(x=Window.width))
            self.root.add_widget(self.pipes[-1])

        for pipe in self.pipes[:]:
            if pipe.right < 0:
                self.pipes.remove(pipe)
            elif pipe.collide_widget(self.bird):
                pipe.highlight = True
                self.bird.score = 0
            else:
                pipe.highlight = False


class Bird(Widget):
    gravity = NumericProperty(.98)
    velocity = NumericProperty(0)
    lift = NumericProperty(25)
    drag = NumericProperty(.05)
    score = NumericProperty()

    def __init__(self, **kwargs):
        super(Bird, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 0)

    def on_touch_down(self, touch):
        self.up()

    def up(self):
        self.velocity += self.lift

    def update(self, *args):
        self.score += .1

        self.velocity -= self.gravity
        self.velocity *= 1 - self.drag
        self.y += self.velocity

        if self.center_y <= 0:
            self.center_y = 0
            self.velocity = 0


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
