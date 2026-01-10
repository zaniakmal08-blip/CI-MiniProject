import pygame
import numpy as np
import random

# --- 1. Simulation Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 120  # High speed for faster learning
ROBOT_RADIUS = 8
GOAL_RADIUS = 15
WHITE, BLACK, RED, GREEN, BLUE = (255, 255, 255), (0, 0, 0), (220, 0, 0), (0, 200, 0), (0, 120, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# --- 2. The FIXED Maze Map ---
maze_walls = [
    pygame.Rect(0, 150, 650, 20),  # Top barrier
    pygame.Rect(150, 320, 650, 20),  # Middle barrier
    pygame.Rect(0, 480, 600, 20)  # Bottom barrier
]
extra_obstacles = [np.array([120, 240]), np.array([680, 240]), np.array([400, 400])]


# --- 3. Controller (The "Brain") ---
def get_move(pos, goal, walls, circles, dna):
    det_range, push_str, speed = dna
    to_goal = goal - pos
    dist_to_goal = np.linalg.norm(to_goal)
    if dist_to_goal < 10: return np.array([0.0, 0.0])

    move_dir = to_goal / dist_to_goal  # Pull toward goal

    # Obstacle Avoidance logic [cite: 13, 15]
    for wall in walls:
        cx, cy = max(wall.left, min(pos[0], wall.right)), max(wall.top, min(pos[1], wall.bottom))
        diff = pos - np.array([cx, cy])
        dist = np.linalg.norm(diff)
        if 0 < dist < det_range:
            move_dir += (diff / dist) * ((det_range - dist) / push_str)

    for obs in circles:
        diff = pos - obs
        dist = np.linalg.norm(diff)
        if 0 < dist < det_range:
            move_dir += (diff / dist) * ((det_range - dist) / push_str)

    mag = np.linalg.norm(move_dir)
    return (move_dir / mag) * speed if mag > 0 else np.array([0.0, 0.0])


# --- 4. GA Learning Logic ---
POP_SIZE = 12
GENS = 5
population = [[random.uniform(50, 150), random.uniform(5, 15), 5.0] for _ in range(POP_SIZE)]
best_ever_dna = population[0]
best_ever_score = -1

# Simulation Loop
for gen in range(GENS):
    gen_scores = []
    for ind_idx, dna in enumerate(population):
        robot_pos = np.array([40.0, 40.0])
        goal_pos = np.array([750.0, 550.0])
        steps, min_dist, path = 0, np.linalg.norm(robot_pos - goal_pos), []

        # INCREASED STEP LIMIT (from 400 to 1200) to ensure it reaches the end
        while steps < 1200:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); exit()

            move = get_move(robot_pos, goal_pos, maze_walls, extra_obstacles, dna)
            robot_pos += move
            path.append(tuple(robot_pos.astype(int)))
            steps += 1

            d = np.linalg.norm(robot_pos - goal_pos)
            if d < min_dist: min_dist = d
            if d < 15: break  # GOAL REACHED!

            # Visualization
            screen.fill(WHITE)
            for wall in maze_walls: pygame.draw.rect(screen, (40, 40, 40), wall)
            for obs in extra_obstacles: pygame.draw.circle(screen, RED, obs.astype(int), 20)
            pygame.draw.circle(screen, GREEN, goal_pos.astype(int), GOAL_RADIUS)
            if len(path) > 2: pygame.draw.lines(screen, (220, 220, 220), False, path, 1)
            pygame.draw.circle(screen, BLUE, robot_pos.astype(int), ROBOT_RADIUS)

            screen.blit(
                font.render(f"GEN: {gen + 1} | IND: {ind_idx + 1} | SCORE: {int(1000 - min_dist)}", True, BLACK),
                (20, 20))
            pygame.display.flip()
            clock.tick(FPS)

        # Fitness: Massive bonus for finishing the maze
        score = (1000 - min_dist) + (500 if min_dist < 15 else 0)
        gen_scores.append(score)
        if score > best_ever_score:
            best_ever_score, best_ever_dna = score, dna

    # Natural Selection (Mutation) [cite: 11, 30]
    population = [best_ever_dna + np.random.uniform(-10, 10, 3) for _ in range(POP_SIZE)]

print("Maze Training Complete.")
pygame.quit()