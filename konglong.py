import pygame
import random
import numpy as np
import os

# 初始化 Pygame
pygame.init()

# 游戏常量设置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60

# 强化学习参数
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # 探索率（有10%的概率随机乱跳，用来尝试新可能）

class Dino:
    def __init__(self):
        self.x = 50
        self.y = 300
        self.width = 40
        self.height = 60
        self.is_jumping = False
        self.jump_velocity = 0
        self.gravity = 0.8

    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = -14

    def update(self):
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            if self.y >= 300:
                self.y = 300
                self.is_jumping = False

    def draw(self, screen):
        # 绘制小恐龙（绿色矩形代替）
        pygame.draw.rect(screen, (46, 139, 87), (self.x, self.y, self.width, self.height))

class Cactus:
    def __init__(self, speed):
        self.x = SCREEN_WIDTH
        self.y = 310
        self.width = 30
        self.height = 50
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        # 绘制仙人掌（红色矩形代替）
        pygame.draw.rect(screen, (178, 34, 34), (self.x, self.y, self.width, self.height))

class QLearningAgent:
    def __init__(self):
        # Q表：状态由（距离区间，速度区间）组成，动作有2个（0: 不动, 1: 跳跃）
        # 为了让原生Q表能存下，我们将连续的距离离散化为20个区间，速度离散化为5个区间
        self.q_table = np.zeros((25, 5, 2))

    def get_state_indices(self, distance, speed):
        # 将复杂的连续数值映射到有限的矩阵索引中
        dist_idx = min(int(distance / 40), 24)
        speed_idx = min(int((speed - 5) / 2), 4)
        return dist_idx, speed_idx

    def choose_action(self, distance, speed):
        # ϵ-贪婪策略
        if random.random() < EPSILON:
            return random.choice([0, 1])  # 随机探索
        
        dist_idx, speed_idx = self.get_state_indices(distance, speed)
        return np.argmax(self.q_table[dist_idx, speed_idx])  # 选择当前最优动作

    def learn(self, distance, speed, action, reward, next_distance, next_speed):
        # 更新Q表公式 (Q-learning)
        d_idx, s_idx = self.get_state_indices(distance, speed)
        next_d_idx, next_s_idx = self.get_state_indices(next_distance, next_speed)
        
        old_value = self.q_table[d_idx, s_idx, action]
        next_max = np.max(self.q_table[next_d_idx, next_s_idx])
        
        # 核心强化学习公式：新值 = 旧值 + 学习率 * (回报 + 折扣因子 * 未来最大期望 - 旧值)
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        self.q_table[d_idx, s_idx, action] = new_value

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("电脑自学小恐龙游戏 (强化学习)")
    clock = pygame.time.Clock()

    agent = QLearningAgent()
    generation = 1
    high_score = 0

    while True:
        # 初始化单局游戏环境
        dino = Dino()
        game_speed = 7
        cacti = [Cactus(game_speed)]
        score = 0
        running = True
        
        # 记录上一次的状态用于学习
        last_distance = SCREEN_WIDTH
        last_speed = game_speed
        last_action = 0

        while running:
            screen.fill((247, 247, 247)) # 浅灰色背景
            
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 获取最近的仙人掌
            active_cactus = None
            for c in cacti:
                if c.x + c.width > dino.x:
                    active_cactus = c
                    break
            
            if not active_cactus:
                active_cactus = cacti[0]

            # 1. 获取当前状态
            current_distance = active_cactus.x - dino.x
            current_speed = game_speed

            # 2. AI 做出决策
            action = agent.choose_action(current_distance, current_speed)
            if action == 1:
                dino.jump()

            # 3. 游戏物理引擎更新
            dino.update()
            
            # 仙人掌生成与移动
            if cacti[-1].x < SCREEN_WIDTH - random.randint(300, 500):
                cacti.append(Cactus(game_speed))
            
            for c in cacti:
                c.update()
                if c.x < -c.width:
                    cacti.remove(c)
                    score += 1
                    # 每过障碍物速度稍微加快一点，增加难度
                    if score % 5 == 0:
                        game_speed += 0.5

            # 4. 判定碰撞（计算奖励）
            reward = 1  # 每活过一帧，给一个小奖励
            collided = False
            
            for c in cacti:
                # 经典的碰撞箱检测
                if (dino.x < c.x + c.width and dino.x + dino.width > c.x and
                    dino.y < c.y + c.height and dino.y + dino.height > c.y):
                    reward = -100  # 撞死了，严重惩罚！
                    collided = True
                    running = False
            
            # 如果成功越过了这个仙人掌（仙人掌刚好到了恐龙后面）
            if active_cactus.x + active_cactus.width < dino.x and not collided:
                reward = 100  # 跳过障碍，巨额奖励！

            # 5. 让 AI 进行学习更新
            agent.learn(last_distance, last_speed, last_action, reward, current_distance, current_speed)

            # 保存当前状态供下一帧更新使用
            last_distance = current_distance
            last_speed = current_speed
            last_action = action

            # 绘制画面
            dino.draw(screen)
            for c in cacti:
                c.draw(screen)
            
            # 绘制地面横线
            pygame.draw.line(screen, (83, 83, 83), (0, 360), (SCREEN_WIDTH, 360), 2)

            # 显示统计文本数据
            if score > high_score:
                high_score = score
            font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20) # 使用黑体支持中文
            text_gen = font.render(f"世代 (Generation): {generation}", True, (50, 50, 50))
            text_score = font.render(f"当前得分 (Score): {score}", True, (50, 50, 50))
            text_high = font.render(f"历史最高 (High Score): {high_score}", True, (50, 50, 50))
            
            screen.blit(text_gen, (20, 20))
            screen.blit(text_score, (20, 45))
            screen.blit(text_high, (20, 70))

            pygame.display.flip()
            clock.tick(FPS)

        # 游戏结束，世代加 1，重新循环开始新的一局
        generation += 1

if __name__ == "__main__":
    main()