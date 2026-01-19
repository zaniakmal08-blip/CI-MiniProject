import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import random

# --- SETTINGS ---
POP_SIZE = 40 
GENS = 5 
MAX_STEPS = 150 
GOAL = np.array([14.0, 14.0])
START = np.array([1.0, 1.0])
MAP_SIZE = 16

# --- OBSTACLES ---
CIRCLES = [[4, 4, 1.2], [12, 4, 0.8], [3, 11, 1.0], [14, 8, 0.6], [8, 7, 0.9], [6, 8, 0.5]]
SQUARES = [[1, 8, 1.2], [10, 10, 1.0], [6, 2, 1.1], [13, 12, 0.9]]
TRIANGLES = [np.array([[5, 10], [8, 11], [6, 13]]), np.array([[9, 3], [11, 5], [8, 5]]), np.array([[2, 5], [4, 7], [2, 7]])]

def is_inside_tri(p, tri):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, tri[0], tri[1]), sign(p, tri[1], tri[2]), sign(p, tri[2], tri[0])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))

class HybridRobot:
    def __init__(self, dna=None, is_explorer=False):
        self.pos = START.astype(float)
        self.angle = np.pi/4 
        self.dna = dna if dna is not None else np.random.uniform(-1, 1, 6)
        self.is_explorer = is_explorer
        self.path = [self.pos.copy()]
        self.alive, self.done = True, False
        self.sensor_angles = np.radians([-90, -45, 0, 45, 90])

    def check_collision(self, p):
        if not (0 <= p[0] <= MAP_SIZE and 0 <= p[1] <= MAP_SIZE): return True
        if any(np.linalg.norm(p - [c[0],c[1]]) < c[2] + 0.15 for c in CIRCLES): return True
        if any(s[0]-0.05 <= p[0] <= s[0]+s[2]+0.05 and s[1]-0.05 <= p[1] <= s[1]+s[2]+0.05 for s in SQUARES): return True
        if any(is_inside_tri(p, t) for t in TRIANGLES): return True
        return False

    def calculate_step(self, is_final=False):
        if not self.alive or self.done: return
        
        # 1. Sensors
        readings = np.full(5, 4.5)
        for i, s_angle in enumerate(self.sensor_angles):
            total_angle = self.angle + s_angle
            s_vec = np.array([np.cos(total_angle), np.sin(total_angle)])
            for dist in np.linspace(0.1, 4.5, 12):
                if self.check_collision(self.pos + s_vec * dist):
                    readings[i] = dist; break
        
        min_dist = np.min(readings)
        
        # 2. Dynamic Speed
        speed = 0.4
        if is_final:
            speed = 0.15 if min_dist < 1.8 else 0.8 # Slow near wall, Sprint in clear
            
        # 3. Steering Logic: Priority = TARGET (STAR)
        # Goal Seeking (Primary Force)
        to_goal = GOAL - self.pos
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        goal_diff = np.arctan2(np.sin(angle_to_goal - self.angle), np.cos(angle_to_goal - self.angle))
        
        # Fuzzy Obstacle Avoidance (Secondary/Reflexive Force)
        fuzzy_steering = 0
        for i in range(5):
            danger = max(0, (3.2 - readings[i])/3.2)**2 
            fuzzy_steering += danger * self.dna[i] * 3.0 # Strong reaction to avoid "langgar"

        # Combine: Goal is always trying to pull, Fuzzy only overrides when danger is high
        steering = (goal_diff * 1.5) + fuzzy_steering 
        
        self.angle += np.clip(steering, -0.8, 0.8)
        new_pos = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * speed
        
        if self.check_collision(new_pos): self.alive = False
        else:
            self.pos = new_pos
            self.path.append(self.pos.copy())
            if np.linalg.norm(GOAL - self.pos) < 0.8: self.done = True

# --- SIMULATION ---
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
pop = [HybridRobot(is_explorer=(i > POP_SIZE//2)) for i in range(POP_SIZE)]
top_3_paths = []

for gen in range(GENS):
    for r in pop: r.pos, r.path, r.alive, r.done, r.angle = START.astype(float), [START.copy()], True, False, np.pi/4
    for step in range(MAX_STEPS):
        ax.clear()
        ax.set_title(f"Target: STAR | Gen {gen+1}/{GENS} | Grey = History Only")
        ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
        for cx, cy, r_ in CIRCLES: ax.add_patch(patches.Circle((cx, cy), r_, color='black', alpha=0.3))
        for sx, sy, sz in SQUARES: ax.add_patch(patches.Rectangle((sx, sy), sz, sz, color='navy', alpha=0.3))
        for tri in TRIANGLES: ax.add_patch(patches.Polygon(tri, color='darkgreen', alpha=0.3))
        ax.plot(GOAL[0], GOAL[1], 'r*', markersize=15, zorder=10) # THE STAR
        for pth in top_3_paths: ax.plot(pth[:,0], pth[:,1], color='gray', alpha=0.1, linewidth=2, zorder=1)

        active = 0
        for r in pop:
            r.calculate_step()
            if r.alive and not r.done: active += 1
            clr = 'orange' if r.is_explorer else 'dodgerblue'
            if r.done: clr = 'lime'
            elif not r.alive: clr = 'red'
            ax.plot(np.array(r.path)[:,0], np.array(r.path)[:,1], color=clr, alpha=0.3)
        plt.pause(0.001)
        if active == 0: break

    pop.sort(key=lambda x: (1000/(np.linalg.norm(GOAL - x.pos)+1) + (10000 - len(x.path)*10 if x.done else 0) - (5000 if not x.alive else 0)), reverse=True)
    top_3_paths = [np.array(pop[i].path) for i in range(min(3, len(pop)))]
    best_dna = copy.deepcopy(pop[0].dna)
    pop = [HybridRobot(best_dna)] + [HybridRobot(best_dna + np.random.normal(0,0.06,6)) for _ in range(POP_SIZE//2)] + [HybridRobot(None, is_explorer=True) for _ in range(POP_SIZE//2 - 1)]

# PHASE 2: Victory Lap
ax.clear()
champion = HybridRobot(best_dna)
print("Starting Final Path...")

while champion.alive and not champion.done and len(champion.path) < 300:
    ax.clear()
    ax.set_title("OPTIMAL STAR-TARGETED PATH", fontsize=14, color='green', fontweight='bold')
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
    for cx, cy, r_ in CIRCLES: ax.add_patch(patches.Circle((cx, cy), r_, color='black'))
    for sx, sy, sz in SQUARES: ax.add_patch(patches.Rectangle((sx, sy), sz, sz, color='navy'))
    for tri in TRIANGLES: ax.add_patch(patches.Polygon(tri, color='darkgreen'))
    ax.plot(GOAL[0], GOAL[1], 'r*', markersize=25, zorder=10)
    
    champion.calculate_step(is_final=True)
    ax.plot(np.array(champion.path)[:,0], np.array(champion.path)[:,1], color='lime', linewidth=6, zorder=5)
    ax.plot(champion.pos[0], champion.pos[1], 'yo', markersize=14, markeredgecolor='k', zorder=11)
    plt.pause(0.04)

plt.ioff()
plt.show()