from Agent import Agent
from Game_Balls import Game

import os
import sys
import argparse

import numpy as np
import random as rnd
import time
from PIL import Image
import cv2

def main(game, agent, show):
    game.create_ball()
    clear = lambda: os.system('cls')
    score = 0
    steps = 1
    epoch = 1
    all_balls = 0
    balls, basket = game.tick(0, 0)
    state = np.squeeze(game.get_matrix())

    while(1):
        if len(balls) == 0 or rnd.random()*3 < balls[0].y/game.size_y:
            game.create_ball()
            all_balls += 1
            
        if agent.model_name == 'lstm':
            state = balls[0].x, balls[0].y, basket.x
            state = np.reshape(state, [1, 3])
        if agent.model_name == 'conv':
            state = game.get_matrix()
            state = np.reshape(state, [1, game.size_x, game.size_y, 1])
        if agent.model_name == 'dense':        
            state = np.squeeze(game.get_matrix())
            state = np.reshape(state, [1, game.size_y * game.size_x])
        
        if show:
            agent.eps = 0.0
        action = agent.do_action(state=state)
        reward = 0
        
        act = action - 1
        balls, basket = game.tick(act, show)
        
        if agent.model_name == 'conv':
            next_state = game.get_matrix()
            next_state = np.reshape(next_state, [1, game.size_x, game.size_y, 1])
        if agent.model_name == 'lstm':
            next_state = balls[0].x, balls[0].y, basket.x
            next_state = np.reshape(next_state, [1, 3])
        if agent.model_name == 'dense': 
            next_state = np.squeeze(game.get_matrix())
            next_state = np.reshape(next_state, [1, game.size_y * game.size_x])

        for ball in balls:
            if (ball.x == basket.x) and (ball.y == game.size_y - 1):
                score += 1
                reward = 1
            elif (ball.x != basket.x) and (ball.y == game.size_y - 1):
                reward = -1

        game.remove_falled_balls()
        agent.remember(state, action, reward, next_state)

        steps += 1
        if steps % 100 == 0:
            if not show:
                agent.save_model()
        
            print('\nEpoch ', epoch, ' Game ', int(steps/(game.size_y - 1)), 'All balls:', all_balls,  ' Scores: ', score, ' Eps ', agent.eps)
        

        if show:
            print('All balls:', all_balls, 'Scores: ', score, 'Game', int(steps/(game.size_y - 1)), 'eps', agent.eps)
            print('_'*25)
            time.sleep(0.05)
            if int(steps/game.size_y) == 20:
                break
            frame = game.get_image()

            cv2.imshow('frame', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            agent.replay()
            
        if agent.eps <= agent.eps_min and not show:
            agent.eps = rnd.random()
            score = 0
            steps = 1
            epoch += 1
            all_balls = 0
        clear()

    #out.release()   
    cv2.destroyAllWindows()

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-n', '--net', choices=['conv', 'dense', 'lstm'], default='dense')
    return parser
    
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    
    h, w = 12, 12
    agent = Agent()
    agent.model_name = namespace.net 
    #agent.model = agent.create_conv_model(h, w)
    #agent.model = agent.create_dence_model(h*w)
    #agent.model = agent.create_lstm_model(3)

    agent.load_model(agent.model_name + '_model.json', agent.model_name + '_model.h5')
    game = Game(h, w)
    main(game, agent, 1)
    

