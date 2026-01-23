import tkinter as tk
import math
import time  # <--- NEW IMPORT

# ==========================================
# 1. CONFIGURATION & MAP DATA
# ==========================================
GOAL = (200, 40)
START_POSE = (360, 460, 3.14)

OBS_COMPLEX = [
    (0, 0, 400, 10), (0, 490, 400, 500), (0, 0, 10, 500), (390, 0, 400, 500),
    (80, 400, 400, 410), (0, 300, 320, 310), (80, 200, 400, 210),
    (0, 120, 150, 130), (250, 120, 400, 130),
    (180, 410, 190, 440), (240, 250, 250, 300), (100, 150, 120, 180),
    (200, 210, 210, 240), (80, 210, 90, 240), (150, 310, 160, 350)
]

# ==========================================
# 2. FUZZY LOGIC BRAIN
# ==========================================
class FuzzyLogicBrain:
    def __init__(self):
        # --- INPUTS ---
        self.d_close = [0, 0, 40]      
        self.d_med   = [10, 40, 50]     
        self.d_far   = [40, 100, 500] 

        self.a_right    = [-3.14, -1.0, -0.1]
        self.a_straight = [-0.3, 0.0, 0.3]
        self.a_left     = [0.1, 1.0, 3.14]

        # --- OUTPUTS ---
        self.turn_out = {"Hard_Right": -0.8, "Soft_Right": -0.3, "Straight": 0.0, "Soft_Left": 0.3, "Hard_Left": 0.8}
        self.speed_out = {"Stop": 0.0, "Slow": 2.0, "Medium": 4.0, "Fast": 7.0}

    def trimf(self, x, params):
        a, b, c = params
        if x <= a or x >= c: return 0.0
        if a < x <= b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def compute(self, sensors, goal_angle):
        # --- 1. FUZZIFICATION ---
        debug_data = {} 
        
        def get_dist_mfs(val):
            return {
                "C": self.trimf(val, self.d_close),
                "M": self.trimf(val, self.d_med),
                "F": self.trimf(val, self.d_far)
            }
        
        s_mfs = []
        for i, d in enumerate(sensors):
            mfs = get_dist_mfs(d)
            s_mfs.append(mfs)
            debug_data[f"S{i}"] = {"val": d, "mfs": mfs, "type": "dist"}

        g_Right = self.trimf(goal_angle, self.a_right)
        g_Str   = self.trimf(goal_angle, self.a_straight)
        g_Left  = self.trimf(goal_angle, self.a_left)
        
        debug_data["Angle"] = {
            "val": goal_angle, 
            "mfs": {"R": g_Right, "S": g_Str, "L": g_Left},
            "type": "angle"
        }

        # --- 2. RULE EVALUATION ---
        turn_rules  = {k: 0.0 for k in self.turn_out}
        speed_rules = {k: 0.0 for k in self.speed_out}

        def fire(strength, turn_action, speed_action):
            turn_rules[turn_action]   = max(turn_rules[turn_action], strength)
            speed_rules[speed_action] = max(speed_rules[speed_action], strength)

        # RULES
        front_danger = s_mfs[0]["C"]
        if sensors[1] < sensors[2]: fire(front_danger, "Hard_Right", "Slow")
        else: fire(front_danger, "Hard_Left", "Slow")

        in_corridor = min(s_mfs[3]["C"], s_mfs[4]["C"])
        fire(in_corridor, "Straight", "Medium")

        fire(min(s_mfs[3]["C"], s_mfs[4]["F"]), "Soft_Right", "Medium") 
        fire(min(s_mfs[4]["C"], s_mfs[3]["F"]), "Soft_Left", "Medium")

        fire(s_mfs[1]["C"], "Soft_Right", "Slow") 
        fire(s_mfs[2]["C"], "Soft_Left",  "Slow")

        safe_front = max(s_mfs[0]["F"], s_mfs[0]["M"])
        safe_left  = max(s_mfs[1]["F"], s_mfs[1]["M"])
        safe_right = max(s_mfs[2]["F"], s_mfs[2]["M"])
        is_safe = min(safe_front, safe_left, safe_right)
        speed_level = "Fast" if min(s_mfs[0]["F"], s_mfs[1]["F"]) > 0.5 else "Medium"
        fire(min(is_safe, g_Left),  "Soft_Left",  speed_level)
        fire(min(is_safe, g_Right), "Soft_Right", speed_level)
        fire(min(is_safe, g_Str),   "Straight",   speed_level)

        fire(min(s_mfs[4]["C"], s_mfs[2]["C"], s_mfs[0]["M"]), "Soft_Left", "Slow")
        fire(min(s_mfs[3]["C"], s_mfs[1]["C"], s_mfs[0]["M"]), "Soft_Right", "Slow")

        # --- 3. DEFUZZIFICATION ---
        t_num, t_den = 0.0, 0.0
        for action, strength in turn_rules.items():
            t_num += strength * self.turn_out[action]
            t_den += strength
        final_turn = t_num / t_den if t_den != 0 else 0.0

        s_num, s_den = 0.0, 0.0
        for action, strength in speed_rules.items():
            s_num += strength * self.speed_out[action]
            s_den += strength

        final_speed = 2.0 if s_den == 0 else s_num / s_den
        return final_speed, final_turn, debug_data

# ==========================================
# 3. GUI GRAPH HELPER
# ==========================================
class FuzzyGraph:
    def __init__(self, parent, title, w=200, h=60, type="dist"):
        self.w, self.h = w, h
        self.type = type # 'dist' or 'angle'
        
        self.frame = tk.Frame(parent, bg="#f0f0f0", bd=1, relief=tk.RAISED)
        self.frame.pack(pady=2, fill=tk.X)
        
        tk.Label(self.frame, text=title, font=("Arial", 8, "bold"), bg="#f0f0f0").pack(side=tk.TOP, anchor="w", padx=5)
        
        self.canvas = tk.Canvas(self.frame, width=w, height=h, bg="white")
        self.canvas.pack(padx=5, pady=2)
        
        self.lbl_val = tk.Label(self.frame, text="Val: 0.0", font=("Consolas", 8), bg="#f0f0f0")
        self.lbl_val.pack(side=tk.BOTTOM, anchor="e", padx=5)

    def draw_bg(self, params):
        self.canvas.delete("all")
        self.canvas.create_line(0, self.h-10, self.w, self.h-10, fill="gray")
        
        if self.type == "dist":
            def to_x(v): return (v / 150.0) * self.w
            p_close, p_med, p_far = params
            self.tri([0, 0, p_close[2]], "red", "C")
            self.tri(p_med, "green", "M")
            self.tri([p_far[0], 100, 150], "blue", "F")
        else:
            def to_x(v): return ((v + 3.14) / 6.28) * self.w
            p_right, p_str, p_left = params
            self.tri(p_right, "blue", "R", to_x)
            self.tri(p_str, "green", "S", to_x)
            self.tri(p_left, "red", "L", to_x)

    def tri(self, pts, col, tag, mapper=None):
        if mapper is None: mapper = lambda v: (v / 150.0) * self.w
        x1, x2, x3 = mapper(pts[0]), mapper(pts[1]), mapper(pts[2])
        base_y = self.h - 10
        top_y = 10
        self.canvas.create_polygon(x1, base_y, x2, top_y, x3, base_y, 
                                   fill=col, outline=col, stipple="gray25", tags=tag)
        self.canvas.create_line(x1, base_y, x2, top_y, x3, base_y, fill=col)

    def update(self, val, mfs):
        self.canvas.delete("needle")
        
        if self.type == "dist": pos_x = (val / 150.0) * self.w
        else: pos_x = ((val + 3.14) / 6.28) * self.w
            
        self.canvas.create_line(pos_x, 0, pos_x, self.h, fill="black", width=2, tags="needle")
        
        txt = f"In: {val:.1f} | "
        for k, v in mfs.items(): txt += f"{k}:{v:.2f} "
        self.lbl_val.config(text=txt)
        
        self.canvas.itemconfig("C", stipple="gray25"); self.canvas.itemconfig("M", stipple="gray25")
        self.canvas.itemconfig("F", stipple="gray25"); self.canvas.itemconfig("R", stipple="gray25")
        self.canvas.itemconfig("S", stipple="gray25"); self.canvas.itemconfig("L", stipple="gray25")

        for k, v in mfs.items():
            if v > 0.01: self.canvas.itemconfig(k, stipple="")

# ==========================================
# 4. SIMULATION APP
# ==========================================
class FuzzySimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Robot - Real Time Dashboard")
        self.root.geometry("1100x700")

        # --- LEFT: SIMULATION ---
        sim_frame = tk.Frame(root)
        sim_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(sim_frame, width=500, height=500, bg="white")
        self.canvas.pack()
        
        btn_frame = tk.Frame(sim_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="RESET", command=self.reset_robot, bg="red", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        self.btn_pause = tk.Button(btn_frame, text="STOP", command=self.toggle_pause, bg="orange", fg="black", width=10)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        # --- RIGHT: FUZZY DASHBOARD (Split Columns) ---
        dash_frame = tk.Frame(root, bg="gray90", bd=2, relief=tk.SUNKEN)
        dash_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Column 1: SENSORS (Distances)
        col1 = tk.Frame(dash_frame, bg="gray90")
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(col1, text="SENSOR INPUTS", bg="gray90", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Column 2: LOGIC & OUTPUT (Angle + Speed/Turn)
        col2 = tk.Frame(dash_frame, bg="gray90")
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(col2, text="LOGIC & OUTPUT", bg="gray90", font=("Arial", 10, "bold")).pack(pady=5)

        # --- TIMER LABEL (NEW) ---
        self.lbl_timer = tk.Label(col2, text="Time: 0.00s", font=("Consolas", 14, "bold"), bg="black", fg="#00FF00")
        self.lbl_timer.pack(pady=(0, 10), fill=tk.X)

        self.graphs = []
        labels = ["Front", "Front-Left", "Front-Right", "Left", "Right"]
        
        self.brain = FuzzyLogicBrain()
        
        # Create 5 Distance Graphs in Column 1
        for i in range(5):
            g = FuzzyGraph(col1, labels[i], w=220, h=60, type="dist")
            g.draw_bg([self.brain.d_close, self.brain.d_med, self.brain.d_far])
            self.graphs.append(g)

        # Create Angle Graph in Column 2 (Top)
        angle_g = FuzzyGraph(col2, "Angle Error", w=220, h=60, type="angle")
        angle_g.draw_bg([self.brain.a_right, self.brain.a_straight, self.brain.a_left])
        self.graphs.append(angle_g) # Index 5 is Angle

        # Create Output Bars in Column 2 (Bottom)
        self.lbl_speed = tk.Label(col2, text="Speed: 0.0", font=("Consolas", 10, "bold"), bg="gray90")
        self.lbl_speed.pack(pady=(15, 0))
        self.bar_speed = tk.Canvas(col2, width=150, height=200, bg="white") # Vertical Bar
        self.bar_speed.pack(pady=5)
        
        self.lbl_turn = tk.Label(col2, text="Turn: 0.0", font=("Consolas", 10, "bold"), bg="gray90")
        self.lbl_turn.pack(pady=(15, 0))
        self.bar_turn = tk.Canvas(col2, width=200, height=40, bg="white") # Horizontal Bar
        self.bar_turn.pack(pady=5)

        # Init Sim Items
        self.canvas.create_oval(GOAL[0]-10, GOAL[1]-10, GOAL[0]+10, GOAL[1]+10, fill="green")
        for i, o in enumerate(OBS_COMPLEX):
            col = "black" if i < 4 else ("gray" if i < 9 else "red")
            self.canvas.create_rectangle(o, fill=col)

        self.poly = self.canvas.create_polygon(0, 0, 0, 0, fill="blue")
        self.ray_lines = [self.canvas.create_line(0, 0, 0, 0, fill="red", width=1) for _ in range(5)]

        self.paused = False
        self.reset_robot()
        self.run_loop()

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="RESUME" if self.paused else "STOP", bg="green" if self.paused else "orange")

    def reset_robot(self):
        self.state = {"x": START_POSE[0], "y": START_POSE[1], "t": START_POSE[2], "active": True}
        self.start_time = time.time()  # <--- START TIMER
        self.lbl_timer.config(text="Time: 0.00s")
        self.canvas.itemconfig(self.poly, fill="blue")

    def get_sensors(self, x, y, t):
        angles = [0, 0.785, -0.785, 1.57, -1.57]
        readings = []
        MAX_RANGE = 150.0
        for i, offset in enumerate(angles):
            ray_t = t + offset
            vx, vy = math.cos(ray_t), math.sin(ray_t)
            closest_dist = MAX_RANGE
            for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
                if abs(vx) > 0.0001:
                    t1, t2 = (ox1 - x)/vx, (ox2 - x)/vx
                    if 0 < t1 < closest_dist and oy1 <= y + t1*vy <= oy2: closest_dist = t1
                    if 0 < t2 < closest_dist and oy1 <= y + t2*vy <= oy2: closest_dist = t2
                if abs(vy) > 0.0001:
                    t3, t4 = (oy1 - y)/vy, (oy2 - y)/vy
                    if 0 < t3 < closest_dist and ox1 <= x + t3*vx <= ox2: closest_dist = t3
                    if 0 < t4 < closest_dist and ox1 <= x + t4*vx <= ox2: closest_dist = t4
            readings.append(closest_dist)
            self.canvas.coords(self.ray_lines[i], x, y, x+closest_dist*vx, y+closest_dist*vy)
        return readings

    def update_outputs(self, speed, turn):
        # Vertical Speed Bar
        self.lbl_speed.config(text=f"Speed: {speed:.1f}")
        self.bar_speed.delete("all")
        h = 200
        h_fill = (speed / 7.0) * h
        self.bar_speed.create_rectangle(0, h-h_fill, 150, h, fill="blue")
        
        # Horizontal Turn Bar
        self.lbl_turn.config(text=f"Turn: {turn:.2f}")
        self.bar_turn.delete("all")
        center = 100
        w_turn = turn * 100
        self.bar_turn.create_rectangle(center, 0, center + w_turn, 40, fill="purple")
        self.bar_turn.create_line(center, 0, center, 40, fill="black", width=2)

    def run_loop(self):
        if self.paused:
            self.root.after(100, self.run_loop)
            return
        if not self.state["active"]:
            self.root.after(100, self.run_loop)
            return

        # UPDATE TIMER
        elapsed = time.time() - self.start_time
        self.lbl_timer.config(text=f"Time: {elapsed:.2f}s")

        x, y, t = self.state["x"], self.state["y"], self.state["t"]
        sensors = self.get_sensors(x, y, t)
        dx, dy = GOAL[0] - x, GOAL[1] - y
        goal_heading = math.atan2(dy, dx)
        angle_err = (goal_heading - t + math.pi) % (2 * math.pi) - math.pi

        speed, turn, debug_info = self.brain.compute(sensors, angle_err)
        
        for i in range(5): self.graphs[i].update(debug_info[f"S{i}"]["val"], debug_info[f"S{i}"]["mfs"])
        self.graphs[5].update(debug_info["Angle"]["val"], debug_info["Angle"]["mfs"])
        self.update_outputs(speed, turn)

        new_t = t + turn
        new_x, new_y = x + math.cos(new_t) * speed, y + math.sin(new_t) * speed

        hit = False
        for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
            if ox1 < new_x < ox2 and oy1 < new_y < oy2: hit = True; break

        if hit:
            self.state["active"] = False
            self.canvas.itemconfig(self.poly, fill="red")
        elif math.hypot(dx, dy) < 15:
            self.state["active"] = False
            self.canvas.itemconfig(self.poly, fill="green")
        else:
            self.state["x"], self.state["y"], self.state["t"] = new_x, new_y, new_t

        r = 12
        pts = [
            new_x + r * math.cos(new_t), new_y + r * math.sin(new_t),
            new_x + r * math.cos(new_t + 2.5), new_y + r * math.sin(new_t + 2.5),
            new_x + r * math.cos(new_t - 2.5), new_y + r * math.sin(new_t - 2.5)
        ]
        self.canvas.coords(self.poly, *pts)
        self.root.after(30, self.run_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzySimApp(root)
    root.mainloop()