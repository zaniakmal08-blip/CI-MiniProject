import tkinter as tk
import math
import random
import json

# ==========================================
# 1. MAP & CONFIGURATION
# ==========================================
GOAL = (200, 40)
START_POSE = (360, 460, 3.14)

OBS_COMPLEX = [
    (0, 0, 400, 10), (0, 490, 400, 500), (0, 0, 10, 500), (390, 0, 400, 500), # Borders
    (80, 400, 400, 410), (0, 300, 320, 310), (80, 200, 400, 210),
    (0, 120, 150, 130), (250, 120, 400, 130),
    # Red Obstacles
    (180, 410, 190, 440), (240, 250, 250, 300), (100, 150, 120, 180),
    (200, 210, 210, 240), (80, 210, 90, 240), (150, 310, 160, 350),(300, 340, 310, 400)
]

# GA SETTINGS
POP_SIZE = 20          
GENERATIONS = 100      
MUTATION_RATE = 0.15   
MAX_STEPS = 600        

# ==========================================
# 2. PARAMETERIZED BRAIN
# ==========================================
class DynamicFuzzyBrain:
    def __init__(self, genes):
        # genes = [close_max, med_min, med_max, far_min]
        c_max, m_min, m_max, f_min = genes
        
        # Enforce logic: Min cannot be > Max
        if m_min >= m_max: m_min = m_max - 1
        
        self.d_close = [0, 0, c_max]
        self.d_med   = [m_min, 40, m_max]
        self.d_far   = [f_min, 100, 1000]

        self.a_right    = [-3.14, -1.0, -0.1]
        self.a_straight = [-0.3, 0.0, 0.3]
        self.a_left     = [0.1, 1.0, 3.14]

        self.turn_out = {"Hard_Right": -0.8, "Soft_Right": -0.3, "Straight": 0.0, "Soft_Left": 0.3, "Hard_Left": 0.8}
        self.speed_out = {"Stop": 0.0, "Slow": 2.0, "Medium": 4.0, "Fast": 7.0}

    def trimf(self, x, params):
        a, b, c = params
        if x <= a or x >= c: return 0.0
        if a < x <= b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def compute(self, sensors, goal_angle):
        def get_mfs(val):
            return {"C": self.trimf(val, self.d_close), "M": self.trimf(val, self.d_med), "F": self.trimf(val, self.d_far)}
        
        s_mfs = [get_mfs(d) for d in sensors]
        g_Right = self.trimf(goal_angle, self.a_right)
        g_Str   = self.trimf(goal_angle, self.a_straight)
        g_Left  = self.trimf(goal_angle, self.a_left)

        turn_rules = {k: 0.0 for k in self.turn_out}
        speed_rules = {k: 0.0 for k in self.speed_out}
        
        def fire(strength, turn_a, speed_a):
            turn_rules[turn_a] = max(turn_rules[turn_a], strength)
            speed_rules[speed_a] = max(speed_rules[speed_a], strength)

        # --- RULES ---
        # 1. Panic
        if sensors[1] < sensors[2]: fire(s_mfs[0]["C"], "Hard_Right", "Slow")
        else: fire(s_mfs[0]["C"], "Hard_Left", "Slow")

        # 2. Navigate
        fire(min(s_mfs[3]["C"], s_mfs[4]["C"]), "Straight", "Medium") 
        fire(min(s_mfs[3]["C"], s_mfs[4]["F"]), "Soft_Right", "Medium")
        fire(min(s_mfs[4]["C"], s_mfs[3]["F"]), "Soft_Left", "Medium")
        fire(s_mfs[1]["C"], "Soft_Right", "Slow")
        fire(s_mfs[2]["C"], "Soft_Left", "Slow")

        # 3. Wall Hugging
        fire(min(s_mfs[4]["C"], s_mfs[2]["C"], s_mfs[0]["M"]), "Soft_Left", "Slow")
        fire(min(s_mfs[3]["C"], s_mfs[1]["C"], s_mfs[0]["M"]), "Soft_Right", "Slow")

        # 4. Goal
        safe = min(max(s_mfs[0]["F"], s_mfs[0]["M"]), max(s_mfs[1]["F"], s_mfs[1]["M"]), max(s_mfs[2]["F"], s_mfs[2]["M"]))
        spd = "Fast" if min(s_mfs[0]["F"], s_mfs[1]["F"]) > 0.5 else "Medium"
        fire(min(safe, g_Left), "Soft_Left", spd)
        fire(min(safe, g_Right), "Soft_Right", spd)
        fire(min(safe, g_Str), "Straight", spd)

        t_num = sum(v * self.turn_out[k] for k, v in turn_rules.items())
        t_den = sum(turn_rules.values())
        turn = t_num / t_den if t_den != 0 else 0.0

        s_num = sum(v * self.speed_out[k] for k, v in speed_rules.items())
        s_den = sum(speed_rules.values())
        speed = 2.0 if s_den == 0 else s_num / s_den
        return speed, turn

# ==========================================
# 3. VISUAL TRAINER APP
# ==========================================
class GAVisualTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("GA Visual Trainer")
        self.root.geometry("800x600")

        # -- UI --
        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.info_panel = tk.Frame(root)
        self.info_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.lbl_gen = tk.Label(self.info_panel, text="Generation: 0", font=("Arial", 14, "bold"))
        self.lbl_gen.pack(pady=5)
        self.lbl_ind = tk.Label(self.info_panel, text="Robot: 0/0", font=("Arial", 12))
        self.lbl_ind.pack(pady=5)
        self.lbl_fit = tk.Label(self.info_panel, text="Best Fitness: 0", font=("Arial", 12, "bold"), fg="green")
        self.lbl_fit.pack(pady=20)
        self.lbl_params = tk.Label(self.info_panel, text="Current Genes:\n-", font=("Consolas", 10), justify=tk.LEFT)
        self.lbl_params.pack(pady=20)

        tk.Button(self.info_panel, text="SAVE & STOP", command=self.save_and_exit, bg="red", fg="white", height=2).pack(side=tk.BOTTOM, pady=20, fill=tk.X)

        # -- GA STATE --
        self.population = [self.create_random_genes() for _ in range(POP_SIZE)]
        self.scored_population = []
        self.gen_count = 1
        self.ind_index = 0
        self.best_global_fitness = 0.0
        self.best_global_genes = []

        # -- ROBOT STATE --
        self.poly = self.canvas.create_polygon(0, 0, 0, 0, fill="blue")
        self.sensor_lines = [self.canvas.create_line(0,0,0,0, fill="red") for _ in range(5)]
        self.path_lines = []
        self.visited = set()
        self.steps = 0
        self.active = False
        
        # Init Map
        self.canvas.create_oval(GOAL[0]-10, GOAL[1]-10, GOAL[0]+10, GOAL[1]+10, fill="green")
        for i, o in enumerate(OBS_COMPLEX):
            col = "black" if i < 4 else ("gray" if i < 9 else "red")
            self.canvas.create_rectangle(o, fill=col)

        self.start_individual()
        # NOTE: run_loop is NOT called here anymore, it is called inside start_individual

    def create_random_genes(self):
        return [
            random.uniform(20, 60),
            random.uniform(5, 30),
            random.uniform(30, 80),
            random.uniform(30, 70)
        ]

    def start_individual(self):
        # Setup next robot
        self.current_genes = self.population[self.ind_index]
        self.brain = DynamicFuzzyBrain(self.current_genes)
        
        self.state = {"x": START_POSE[0], "y": START_POSE[1], "t": START_POSE[2]}
        self.visited = set()
        self.steps = 0
        self.active = True
        self.start_dist = math.hypot(GOAL[0] - self.state["x"], GOAL[1] - self.state["y"])

        # Update UI
        self.lbl_gen.config(text=f"Generation: {self.gen_count}")
        self.lbl_ind.config(text=f"Robot: {self.ind_index + 1} / {POP_SIZE}")
        genes_str = f"Close_Max: {self.current_genes[0]:.1f}\nMed_Min:   {self.current_genes[1]:.1f}\nMed_Max:   {self.current_genes[2]:.1f}\nFar_Min:   {self.current_genes[3]:.1f}"
        self.lbl_params.config(text=genes_str)
        
        # Clear previous path
        for line in self.path_lines: self.canvas.delete(line)
        self.path_lines = []
        
        # --- FIX: RESTART THE LOOP HERE ---
        self.root.after(10, self.run_loop)

    def get_sensors(self, x, y, t):
        angles = [0, 0.785, -0.785, 1.57, -1.57]
        readings = []
        for i, offset in enumerate(angles):
            ray_t = t + offset
            vx, vy = math.cos(ray_t), math.sin(ray_t)
            closest = 150.0
            
            for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
                if abs(vx) > 0.001:
                    t1, t2 = (ox1 - x)/vx, (ox2 - x)/vx
                    if 0 < t1 < closest and oy1 <= y + t1*vy <= oy2: closest = t1
                    if 0 < t2 < closest and oy1 <= y + t2*vy <= oy2: closest = t2
                if abs(vy) > 0.001:
                    t3, t4 = (oy1 - y)/vy, (oy2 - y)/vy
                    if 0 < t3 < closest and ox1 <= x + t3*vx <= ox2: closest = t3
                    if 0 < t4 < closest and ox1 <= x + t4*vx <= ox2: closest = t4
            
            readings.append(closest)
            self.canvas.coords(self.sensor_lines[i], x, y, x+closest*vx, y+closest*vy)
        return readings

    def calculate_fitness(self, status):
        final_dist = math.hypot(GOAL[0] - self.state["x"], GOAL[1] - self.state["y"])
        fitness = (self.start_dist - final_dist) * 2.0 
        
        if status == "GOAL":
            fitness += 5000.0 + (MAX_STEPS - self.steps) * 2
        elif status == "COLLISION":
            fitness -= 200.0
        
        if self.steps > 50:
            ratio = len(self.visited) / self.steps
            if ratio < 0.15: fitness -= 1000.0 

        return max(0.0, fitness)

    def end_individual(self, status):
        # 1. Save score
        fitness = self.calculate_fitness(status)
        self.scored_population.append((fitness, self.current_genes))
        
        if fitness > self.best_global_fitness:
            self.best_global_fitness = fitness
            self.best_global_genes = self.current_genes
            self.lbl_fit.config(text=f"Best Fitness: {fitness:.1f}")

        # 2. Advance index
        self.ind_index += 1
        
        # 3. Decision: Next Robot OR Next Generation
        if self.ind_index < POP_SIZE:
            # Loop restarts automatically inside start_individual via root.after
            self.start_individual()
        else:
            self.evolve_population()

    def evolve_population(self):
        self.scored_population.sort(key=lambda x: x[0], reverse=True)
        parents = [x[1] for x in self.scored_population[:4]] 
        
        next_gen = list(parents)
        while len(next_gen) < POP_SIZE:
            p1, p2 = random.sample(parents, 2)
            pt = random.randint(1, 3)
            child = p1[:pt] + p2[pt:]
            if random.random() < MUTATION_RATE:
                idx = random.randint(0, 3)
                child[idx] += random.uniform(-5, 5)
            next_gen.append(child)
        
        self.population = next_gen
        self.scored_population = []
        self.gen_count += 1
        self.ind_index = 0
        
        if self.gen_count <= GENERATIONS:
            self.start_individual()
        else:
            self.save_and_exit()

    def run_loop(self):
        # We don't check self.active for scheduling anymore, 
        # because the loop is now explicitly restarted by start_individual
        
        # 1. Physics
        x, y, t = self.state["x"], self.state["y"], self.state["t"]
        sensors = self.get_sensors(x, y, t)
        
        dx, dy = GOAL[0] - x, GOAL[1] - y
        goal_heading = math.atan2(dy, dx)
        angle_err = (goal_heading - t + math.pi) % (2 * math.pi) - math.pi
        
        speed, turn = self.brain.compute(sensors, angle_err)
        speed *= 2.0 
        
        new_t = t + turn
        new_x = x + math.cos(new_t) * speed
        new_y = y + math.sin(new_t) * speed
        
        # 2. Collision
        hit = False
        for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
            if ox1 < new_x < ox2 and oy1 < new_y < oy2:
                hit = True; break
        
        # 3. Draw
        r = 10
        self.canvas.coords(self.poly, 
            new_x + r*math.cos(new_t), new_y + r*math.sin(new_t),
            new_x + r*math.cos(new_t+2.5), new_y + r*math.sin(new_t+2.5),
            new_x + r*math.cos(new_t-2.5), new_y + r*math.sin(new_t-2.5)
        )
        if self.steps % 5 == 0:
            line = self.canvas.create_oval(new_x, new_y, new_x+2, new_y+2, fill="blue", outline="")
            self.path_lines.append(line)

        # 4. Check End (These call end_individual, which eventually calls start_individual, which restarts loop)
        if hit:
            self.end_individual("COLLISION")
            return
        elif math.hypot(dx, dy) < 15:
            self.end_individual("GOAL")
            return
        elif self.steps >= MAX_STEPS:
            self.end_individual("TIMEOUT")
            return

        # 5. Continue
        self.state = {"x": new_x, "y": new_y, "t": new_t}
        self.visited.add((int(new_x//10), int(new_y//10)))
        self.steps += 1
        
        self.root.after(1, self.run_loop)

    def save_and_exit(self):
        if not self.best_global_genes:
            print("No training done yet.")
            self.root.destroy()
            return
            
        print(f"Saving Best Genes: {self.best_global_genes}")
        with open("best_params.json", "w") as f:
            json.dump(self.best_global_genes, f)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GAVisualTrainer(root)
    root.mainloop()