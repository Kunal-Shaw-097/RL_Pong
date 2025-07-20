import pygame
import numpy as np
import random

class PongEnv:
    def __init__(self, render_mode=False):
        pygame.init()
        self.WIDTH, self.HEIGHT = 640, 480
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 100
        self.BALL_SIZE = 10
        self.PADDLE_SPEED = 5
        self.BALL_SPEED = 5
        self.FPS = 60
        self.render_mode = render_mode

        if self.render_mode:
            self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pong RL Environment")
        else:
            self.win = None

        self.clock = pygame.time.Clock()
        self.action_space = [0, 1, 2]  # 0 = stay, 1 = up, 2 = down
        self.reset()

    def reset(self):
        self.player_y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.ai_y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.HEIGHT // 2

        # Use random float velocities with speed range and angle diversity
        angle = random.uniform(-np.pi / 3, np.pi / 3)
        speed = self.BALL_SPEED
        self.ball_vel_x = -speed  # always starts toward player
        self.ball_vel_y = speed * np.tan(angle)

        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_vel_x / self.BALL_SPEED,
            self.ball_vel_y / self.BALL_SPEED,
            self.player_y / self.HEIGHT,
        ], dtype=np.float32)


    def step(self, action):
        # === Player action ===
        if action == 1 and self.player_y > 0:
            self.player_y -= self.PADDLE_SPEED
        elif action == 2 and self.player_y + self.PADDLE_HEIGHT < self.HEIGHT:
            self.player_y += self.PADDLE_SPEED

        # === AI movement (basic tracking) ===
        if self.ai_y + self.PADDLE_HEIGHT // 2 < self.ball_y:
            self.ai_y += self.PADDLE_SPEED
        elif self.ai_y + self.PADDLE_HEIGHT // 2 > self.ball_y:
            self.ai_y -= self.PADDLE_SPEED

        # === Update ball position ===
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # === Ball bounce off top/bottom walls ===
        if self.ball_y <= 0 or self.ball_y >= self.HEIGHT - self.BALL_SIZE:
            self.ball_vel_y *= -1
            self.ball_y = max(0, min(self.ball_y, self.HEIGHT - self.BALL_SIZE))

        # === Check scoring ===
        if self.ball_x <= 0:
            self.done = True
            return self.get_state(), -1.0, self.done
        elif self.ball_x >= self.WIDTH:
            self.done = True
            return self.get_state(), 5.0, self.done

        # === Collision Rectangles ===
        player_rect = pygame.Rect(10, self.player_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ai_rect = pygame.Rect(self.WIDTH - 20, self.ai_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        reward = 0.001  # small survival reward
        player_hit = False

        # === Player Paddle Collision ===
        if player_rect.colliderect(ball_rect) and self.ball_vel_x < 0:
            self.ball_x = player_rect.right + 1  # push ball outside paddle
            self.ball_vel_x *= -1
            player_hit = True

        # === AI Paddle Collision ===
        if ai_rect.colliderect(ball_rect) and self.ball_vel_x > 0:
            self.ball_x = ai_rect.left - self.BALL_SIZE - 1
            self.ball_vel_x *= -1

        # === Clamp ball velocity to avoid stickiness ===
        if abs(self.ball_vel_x) < 1.0:
            self.ball_vel_x = np.sign(self.ball_vel_x) * 1.0

        # === Reward logic ===
        if player_hit:
            reward += 2.0

        # Distance-based penalty
        dist = abs((self.player_y + self.PADDLE_HEIGHT // 2) - self.ball_y)
        reward -= 0.005 * (dist / self.HEIGHT)

        # Standby center reward (if ball is far from player)
        if self.ball_x > self.WIDTH // 2:
            reward -= 0.002 * abs(self.player_y - (self.HEIGHT // 2)) / self.HEIGHT

        # Clamp reward to avoid explosion
        reward = max(min(reward, 10.0), -2.0)

        return self.get_state(), reward, self.done






    def render(self):
        if not self.render_mode:
            return

        self.clock.tick(self.FPS)
        self.win.fill((0, 0, 0))

        WHITE = (255, 255, 255)
        pygame.draw.rect(self.win, WHITE, (10, self.player_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.win, WHITE, (self.WIDTH - 20, self.ai_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.win, WHITE, (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        pygame.display.update()

    def close(self):
        pygame.quit()
