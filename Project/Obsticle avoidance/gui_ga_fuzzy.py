import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import copy

# --- INITIAL SETTINGS ---
DEFAULT_POP_SIZE = 40 
DEFAULT_GENS = 10 
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

    def calculate_step(self, speed=0.4):
        if not self.alive or self.done: return
        readings = np.full(5, 4.5)
        for i, s_angle in enumerate(self.sensor_angles):
            total_angle = self.angle + s_angle
            s_vec = np.array([np.cos(total_angle), np.sin(total_angle)])
            for dist in np.linspace(0.1, 4.5, 12):
                if self.check_collision(self.pos + s_vec * dist):
                    readings[i] = dist; break
        
        to_goal = GOAL - self.pos
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        goal_diff = np.arctan2(np.sin(angle_to_goal - self.angle), np.cos(angle_to_goal - self.angle))
        
        fuzzy_steering = 0
        for i in range(5):
            danger = max(0, (3.2 - readings[i])/3.2)**2 
            fuzzy_steering += danger * self.dna[i] * 3.0

        steering = (goal_diff * 1.5) + fuzzy_steering 
        self.angle += np.clip(steering, -0.8, 0.8)
        new_pos = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * speed
        
        if self.check_collision(new_pos): self.alive = False
        else:
            self.pos = new_pos
            self.path.append(self.pos.copy())
            if np.linalg.norm(GOAL - self.pos) < 0.8: self.done = True

# --- GUI STATE ---
class SimulationState:
    def __init__(self):
        self.is_playing = False
        self.reset_needed = False
        self.show_best_trigger = False
        self.best_dna = None

state = SimulationState()

# --- CALLBACKS ---
def toggle_play(event):
    state.is_playing = not state.is_playing
    btn_play.label.set_text('PAUSE' if state.is_playing else 'PLAY')

def reset_sim(event):
    state.reset_needed = True

def trigger_victory_lap(event):
    if state.best_dna is not None:
        state.show_best_trigger = True

# --- SETUP PLOT ---
fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(bottom=0.25)

# UI Elements
ax_pop = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_gen = plt.axes([0.15, 0.10, 0.3, 0.03])
ax_play = plt.axes([0.55, 0.10, 0.1, 0.08])
ax_reset = plt.axes([0.67, 0.10, 0.1, 0.08])
ax_best = plt.axes([0.79, 0.10, 0.15, 0.08])

s_pop = Slider(ax_pop, 'Populations', 10, 100, valinit=DEFAULT_POP_SIZE, valstep=1)
s_gen = Slider(ax_gen, 'Gens', 1, 50, valinit=DEFAULT_GENS, valstep=1)
btn_play = Button(ax_play, 'PLAY')
btn_reset = Button(ax_reset, 'RESET')
btn_best = Button(ax_best, 'SHOW BEST')

btn_play.on_clicked(toggle_play)
btn_reset.on_clicked(reset_sim)
btn_best.on_clicked(trigger_victory_lap)

def draw_map(title):
    ax.clear()
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
    for cx, cy, r_ in CIRCLES: ax.add_patch(patches.Circle((cx, cy), r_, color='black', alpha=0.3))
    for sx, sy, sz in SQUARES: ax.add_patch(patches.Rectangle((sx, sy), sz, sz, color='navy', alpha=0.3))
    for tri in TRIANGLES: ax.add_patch(patches.Polygon(tri, color='darkgreen', alpha=0.3))
    ax.plot(GOAL[0], GOAL[1], 'r*', markersize=15, zorder=10)

# --- MAIN LOOP ---
while True:
    pop_size = int(s_pop.val)
    max_gens = int(s_gen.val)
    pop = [HybridRobot(is_explorer=(i > pop_size//2)) for i in range(pop_size)]
    top_3_paths = []
    state.reset_needed = False
    state.best_dna = pop[0].dna # Initial random best

    for gen in range(max_gens):
        if state.reset_needed: break
        
        for r in pop: 
            r.pos, r.path, r.alive, r.done, r.angle = START.astype(float), [START.copy()], True, False, np.pi/4
        
        for step in range(MAX_STEPS):
            if state.reset_needed: break
            
            # SHOW BEST / VICTORY LAP LOGIC
            if state.show_best_trigger:
                state.is_playing = False # Pause main sim
                champ = HybridRobot(state.best_dna)
                while champ.alive and not champ.done and len(champ.path) < 300:
                    draw_map("FINAL OPTIMAL PATH RUN")
                    champ.calculate_step(speed=0.2) # Slower for visibility
                    ax.plot(np.array(champ.path)[:,0], np.array(champ.path)[:,1], color='lime', linewidth=4, zorder=5)
                    ax.plot(champ.pos[0], champ.pos[1], 'yo', markersize=10, markeredgecolor='k', zorder=11)
                    plt.pause(0.01)
                state.show_best_trigger = False
                btn_play.label.set_text('PLAY')

            while not state.is_playing and not state.reset_needed and not state.show_best_trigger:
                plt.pause(0.1)
            
            draw_map(f"Population: pop_size | Gen {gen+1}/{max_gens} | Step {step}")
            for pth in top_3_paths: ax.plot(pth[:,0], pth[:,1], color='gray', alpha=0.1)

            active = 0
            for r in pop:
                r.calculate_step()
                if r.alive and not r.done: active += 1
                clr = 'orange' if r.is_explorer else 'dodgerblue'
                if r.done: clr = 'lime'
                elif not r.alive: clr = 'red'
                ax.plot(np.array(r.path)[:,0], np.array(r.path)[:,1], color=clr, alpha=0.2)
            
            plt.pause(0.001)
            if active == 0: break

        # Evolution
        pop.sort(key=lambda x: (1000/(np.linalg.norm(GOAL - x.pos)+1) + (10000 if x.done else 0)), reverse=True)
        state.best_dna = copy.deepcopy(pop[0].dna)
        top_3_paths = [np.array(pop[i].path) for i in range(min(3, len(pop)))]
        pop = [HybridRobot(state.best_dna)] + \
              [HybridRobot(state.best_dna + np.random.normal(0,0.06,6)) for _ in range(pop_size//2)] + \
              [HybridRobot(None, is_explorer=True) for _ in range(pop_size//2 - 1)]

    if not state.reset_needed:
        state.is_playing = False
        btn_play.label.set_text('PLAY')
        print("Generations Complete. Click SHOW BEST for the victory lap or RESET to restart.")
        while not state.reset_needed:
            if state.show_best_trigger:
                # Same victory lap code for when simulation is finished
                champ = HybridRobot(state.best_dna)
                while champ.alive and not champ.done and len(champ.path) < 300:
                    draw_map("OPTIMAL STAR-TARGETED PATH")
                    champ.calculate_step(speed=0.2)
                    ax.plot(np.array(champ.path)[:,0], np.array(champ.path)[:,1], color='lime', linewidth=6, zorder=5)
                    ax.plot(champ.pos[0], champ.pos[1], 'yo', markersize=12, markeredgecolor='k', zorder=11)
                    plt.pause(0.02)
                state.show_best_trigger = False

            plt.pause(0.1)
