import os
import sys

import numpy as np
import random as rnd
import time

ball_symbol = '0'
basket_symbol = '#'
empty_symbol = '-'

class Ball:
    def __init__(self, grid, s=ball_symbol):
        self.x = rnd.randint(0, grid.size_x - 1)
        self.y = 0
        self.s = s

    def fall(self):
        self.y += 1


class Basket:
    def __init__(self, grid, s=basket_symbol):
        self.y = grid.size_x - 1
        self.x = int(grid.size_x / 2)
        self.s = s

    def move(self, direction):
        self.x += direction


class Game:
    def __init__(self, size_x=10, size_y=10):
        self.size_x = size_x
        self.size_y = size_y
        self.img_window = 15
        self.grid = self.create_grid(size_y, size_x)
        self.basket = Basket(self)
        self.balls = []

    def create_grid(self, n, m):
        grid = [[empty_symbol for i in range(n)] for i in range(m)]
        return grid

    def create_ball(self):
        ball = Ball(self)
        self.balls.append(ball)

    def move_balls(self):
        for ball in self.balls:
            if ball.y < self.size_y - 1:
                ball.fall()

    def move_basket(self, direction):
        self.basket.x += direction

        if self.basket.x <= 0:
            self.basket.x = 0

        if self.basket.x >= self.size_x - 1:
            self.basket.x = self.size_x - 1

    def show_grid(self):

        for row in self.grid:
            line = ''
            for i in range(0, len(row)):
                if i == len(row) - 1:
                    line += row[i]
                else:
                    line += row[i] + ' '
            print('|', line, '|')

    def tick(self, basket_dir, is_show):
        self.move_balls()
        self.move_basket(direction=basket_dir)

        self.grid = self.create_grid(self.size_y, self.size_x)

        for ball in self.balls:
            if ball.x < len(self.grid[0]) and ball.y < len(self.grid):
                self.grid[ball.y][ball.x] = ball.s

        self.grid[self.basket.y][self.basket.x] = self.basket.s

        if is_show:
            self.show_grid()
        return self.balls, self.basket

    def remove_falled_balls(self):
        for ball in self.balls:
            if ball.y >= self.size_y - 1:
                self.balls.remove(ball)

    def get_matrix(self):
        matrix = np.zeros((self.size_x, self.size_y))
        for i in range(0, self.size_x):
            for j in range(0, self.size_y):
                if self.grid[j][i] == empty_symbol:
                    matrix[j, i] = 0
                elif self.grid[j][i] == ball_symbol:
                    matrix[j, i] = 128
                elif self.grid[j][i] == basket_symbol:
                    matrix[j, i] = 256
        return matrix
    
    def get_image(self):
        matrix = np.zeros((self.size_x*self.img_window, self.size_y*self.img_window))
        for i in range(0, self.size_x):
            for j in range(0, self.size_y):
                if self.grid[j][i] == empty_symbol:
                    self._put_pixels(matrix, j, i, 0)
                elif self.grid[j][i] == ball_symbol:
                    self._put_pixels(matrix, j, i, 128)
                elif self.grid[j][i] == basket_symbol:
                    self._put_pixels(matrix, j, i, 256)
        return matrix
    
    def _put_pixels(self, matrix, x, y, value):
        for i in np.arange(0, self.img_window):
            for j in np.arange(0, self.img_window):
                matrix[x*self.img_window + i, y*self.img_window + j] = value

    
