import pygame
import numpy as np

# --- 1. Simulation Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60
ROBOT_RADIUS = 10
GOAL_RADIUS = 15
STEP_SIZE = 3

# --- 2. Colors (FIXED: Added BLACK) ---
WHITE = (255, 255, 255)
RED = (200, 0, 0)  # Obstacles [cite: 25]
GREEN = (0, 200, 0)  # Goal [cite: 25]
BLUE = (0, 100, 255)  # Robot [cite: 27]
BLACK = (0, 0, 0)  # Text color
GRAY = (180, 180, 180)  # Path trace color

# --- 3. Setup Environment ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MCTA 3371: Intelligent Navigation GUI")
font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# Positions
robot_pos = np.array([50.0, 50.0])
goal_pos = np.array([750.0, 550.0])
path_history = []  # To store the trace line

# Map Setup: Obstacles [cite: 26]
obstacles = [
    np.array([400, 300]), np.array([200, 150]),
    np.array([600, 450]), np.array([400, 100]),
    np.array([150, 400]), np.array([500, 200])
]


# --- 4. Navigation Controller (Soft Computing Logic) [cite: 13, 15] ---
def get_move(pos, goal, obs_list):
    to_goal = goal - pos
    dist_to_goal = np.linalg.norm(to_goal)

    if dist_to_goal < 5:
        return np.array([0.0, 0.0])

    # Base move toward goal [cite: 16]
    move_dir = to_goal / dist_to_goal

    # Fuzzy-style Avoidance [cite: 11, 30]
    for obs in obs_list:
        to_obs = pos - obs
        dist = np.linalg.norm(to_obs)
        if dist < 80:  # Detection range [cite: 28]
            push_strength = (80 - dist) / 15
            move_dir += (to_obs / dist) * push_strength

    move_dir = move_dir / np.linalg.norm(move_dir)
    return move_dir * STEP_SIZE


# --- 5. Main Loop ---
running = True
while running:
    import pygame
    import numpy as np

    # --- 1. Simulation Constants ---
    WIDTH, HEIGHT = 800, 600
    FPS = 60
    ROBOT_RADIUS = 8
    GOAL_RADIUS = 12
    STEP_SIZE = 2.5

    # --- 2. Colors ---
    WHITE = (255, 255, 255)
    WALL_COLOR = (50, 50, 50)  # Dark Gray for Maze
    RED = (200, 0, 0)  # Small obstacles
    GREEN = (0, 200, 0)  # Goal
    BLUE = (0, 100, 255)  # Robot
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)

    # --- 3. Setup Environment ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MCTA 3371: Maze Navigation & Obstacle Avoidance")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    # Positions
    robot_pos = np.array([40.0, 40.0])
    goal_pos = np.array([750.0, 550.0])
    path_history = []

    # --- 4. Maze & Obstacle Definitions ---
    # Walls are defined as Pygame Rects: (x, y, width, height)
    maze_walls = [
        pygame.Rect(0, 100, 600, 20),  # Horizontal Wall 1
        pygame.Rect(200, 250, 600, 20),  # Horizontal Wall 2
        pygame.Rect(0, 400, 500, 20),  # Horizontal Wall 3
        pygame.Rect(500, 400, 20, 200),  # Vertical Wall 1
    ]

    # Additional small circular obstacles for complexity [cite: 25]
    extra_obstacles = [
        np.array([100, 200]), np.array([700, 150]),
        np.array([300, 350]), np.array([650, 500])
    ]


    # --- 5. Navigation Controller (Hybrid Avoidance) ---
    def get_move(pos, goal, walls, circles):
        to_goal = goal - pos
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 5:
            return np.array([0.0, 0.0])

        # Primary "Force": Toward Goal [cite: 16]
        move_dir = to_goal / dist_to_goal

        # Avoid Maze Walls
        for wall in walls:
            # Find closest point on rectangle to robot
            closest_x = max(wall.left, min(pos[0], wall.right))
            closest_y = max(wall.top, min(pos[1], wall.bottom))
            closest_pt = np.array([closest_x, closest_y])

            diff = pos - closest_pt
            dist = np.linalg.norm(diff)

            if dist < 40:  # Detection threshold for walls
                push = (40 - dist) / 10
                # If robot is inside or too close, push away
                if dist == 0: diff = np.array([1.0, 1.0])  # Prevent division by zero
                move_dir += (diff / (dist + 0.1)) * push

        # Avoid Small Obstacles
        for obs in circles:
            diff = pos - obs
            dist = np.linalg.norm(diff)
            if dist < 60:
                push = (60 - dist) / 12
                move_dir += (diff / dist) * push

        # Normalize final direction
        mag = np.linalg.norm(move_dir)
        if mag > 0:
            move_dir = move_dir / mag
        return move_dir * STEP_SIZE


    # --- 6. Main Loop ---
    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                goal_pos = np.array([float(event.pos[0]), float(event.pos[1])])
                path_history = []

        # Calculate Movement
        move = get_move(robot_pos, goal_pos, maze_walls, extra_obstacles)

        # Collision check: Don't move if it would put us inside a wall
        next_pos = robot_pos + move
        robot_rect = pygame.Rect(next_pos[0] - ROBOT_RADIUS, next_pos[1] - ROBOT_RADIUS, ROBOT_RADIUS * 2,
                                 ROBOT_RADIUS * 2)

        collision = any(robot_rect.colliderect(w) for w in maze_walls)
        if not collision:
            robot_pos = next_pos

        path_history.append(tuple(robot_pos.astype(int)))

        # --- Drawing ---
        if len(path_history) > 2:
            pygame.draw.lines(screen, GRAY, False, path_history, 2)

        # Draw Maze Walls
        for wall in maze_walls:
            pygame.draw.rect(screen, WALL_COLOR, wall)

        # Draw Extra Obstacles
        for obs in extra_obstacles:
            pygame.draw.circle(screen, RED, obs.astype(int), 20)

        pygame.draw.circle(screen, GREEN, goal_pos.astype(int), GOAL_RADIUS)
        pygame.draw.circle(screen, BLUE, robot_pos.astype(int), ROBOT_RADIUS)

        # GUI Info
        screen.blit(font.render("COMPLEX MAP: Maze + Obstacles", True, BLACK), (10, 10))
        screen.blit(font.render("Click to move Goal - Robot will try to find a path", True, BLACK), (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
pygame.quit()