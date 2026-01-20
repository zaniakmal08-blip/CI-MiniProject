import tkinter as tk
import math
import json
import os
import time

# ==========================================
# 1. CONFIGURATION & MAP
# ==========================================
GOAL = (200, 40)
START_POSE = (360, 460, 3.14)

OBS_COMPLEX = [
    # Borders
    (0, 0, 400, 10), 
    (0, 490, 400, 500), 
    (0, 0, 10, 500), 
    (390, 0, 400, 500),

    # One simple obstacle (horizontal bar)
    (100, 250, 300, 270)
]


MANUAL_PARAMS = [40, 10, 50, 40]

# ==========================================
# 2. SHARED FUZZY LOGIC CLASS (Enhanced)
# ==========================================
class FuzzyBrain:
    def __init__(self, params):
        self.params = params
        c_max, m_min, m_max, f_min = params
        
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
        
        # Calculate MFs for all sensors
        s_mfs = [get_mfs(d) for d in sensors]
        
        # Capture Front Sensor Data for GUI Display (Index 0)
        debug_front = s_mfs[0] 
        
        g_Right = self.trimf(goal_angle, self.a_right)
        g_Str   = self.trimf(goal_angle, self.a_straight)
        g_Left  = self.trimf(goal_angle, self.a_left)

        turn_rules = {k: 0.0 for k in self.turn_out}
        speed_rules = {k: 0.0 for k in self.speed_out}
        
        def fire(strength, turn_a, speed_a):
            turn_rules[turn_a] = max(turn_rules[turn_a], strength)
            speed_rules[speed_a] = max(speed_rules[speed_a], strength)

        # --- RULES ---
        if sensors[1] < sensors[2]: fire(s_mfs[0]["C"], "Hard_Right", "Slow")
        else: fire(s_mfs[0]["C"], "Hard_Left", "Slow")

        fire(min(s_mfs[3]["C"], s_mfs[4]["C"]), "Straight", "Medium") 
        fire(min(s_mfs[3]["C"], s_mfs[4]["F"]), "Soft_Right", "Medium")
        fire(min(s_mfs[4]["C"], s_mfs[3]["F"]), "Soft_Left", "Medium")
        fire(s_mfs[1]["C"], "Soft_Right", "Slow")
        fire(s_mfs[2]["C"], "Soft_Left", "Slow")

        # Wall Hugging
        fire(min(s_mfs[4]["C"], s_mfs[2]["C"], s_mfs[0]["M"]), "Soft_Left", "Slow")
        fire(min(s_mfs[3]["C"], s_mfs[1]["C"], s_mfs[0]["M"]), "Soft_Right", "Slow")

        # Goal Seek
        safe = min(max(s_mfs[0]["F"], s_mfs[0]["M"]), max(s_mfs[1]["F"], s_mfs[1]["M"]), max(s_mfs[2]["F"], s_mfs[2]["M"]))
        spd = "Fast" if min(s_mfs[0]["F"], s_mfs[1]["F"]) > 0.5 else "Medium"
        fire(min(safe, g_Left), "Soft_Left", spd)
        fire(min(safe, g_Right), "Soft_Right", spd)
        fire(min(safe, g_Str), "Straight", spd)

        # Defuzzify
        t_num = sum(v * self.turn_out[k] for k, v in turn_rules.items())
        t_den = sum(turn_rules.values())
        turn = t_num / t_den if t_den != 0 else 0.0

        s_num = sum(v * self.speed_out[k] for k, v in speed_rules.items())
        s_den = sum(speed_rules.values())
        speed = 2.0 if s_den == 0 else s_num / s_den
        
        return speed, turn, debug_front

# ==========================================
# 3. COMPARISON APP
# ==========================================
class ComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Standard vs Optimized Fuzzy Logic (Analysis Mode)")
        self.root.geometry("1200x800")

        # Load Params
        self.opt_params = MANUAL_PARAMS
        if os.path.exists("best_params.json"):
            try:
                with open("best_params.json", "r") as f: self.opt_params = json.load(f)
                print("Loaded optimized parameters.")
            except: pass

        # --- LAYOUT SETUP ---
        # 1. Left (Standard)
        f_std = tk.Frame(root, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        f_std.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.panel_std = self.create_panel(f_std, "STANDARD", MANUAL_PARAMS, "#f0f0f0", "blue")

        # 2. Right (Optimized)
        f_opt = tk.Frame(root, bg="#e0ffe0", bd=2, relief=tk.GROOVE)
        f_opt.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.panel_opt = self.create_panel(f_opt, "OPTIMIZED GA", self.opt_params, "#e0ffe0", "green")

        tk.Button(root, text="RESTART SIMULATION", command=self.reset_sim, bg="orange", font=("Arial", 10, "bold")).place(relx=0.5, rely=0.02, anchor=tk.N)

        self.setup_sim()
        self.run_loop()

    def create_panel(self, parent, title, params, bg_col, ray_col):
        # Header
        tk.Label(parent, text=title, font=("Arial", 16, "bold"), bg=bg_col).pack(pady=5)
        
        # Canvas
        cv = tk.Canvas(parent, width=500, height=450, bg="white")
        cv.pack()

        # Timer & Status
        lbl_time = tk.Label(parent, text="Time: 0.0s", font=("Consolas", 14, "bold"), bg=bg_col)
        lbl_time.pack(pady=5)
        
        # Fuzzy Graph Canvas
        lbl_graph = tk.Label(parent, text="Membership Functions (Distance)", font=("Arial", 10, "bold"), bg=bg_col)
        lbl_graph.pack()
        cv_graph = tk.Canvas(parent, width=500, height=100, bg="white")
        cv_graph.pack(pady=5)
        self.draw_fuzzy_graph(cv_graph, params)
        
        # Fuzzification Live Data
        lbl_data = tk.Label(parent, text="Waiting...", font=("Consolas", 9), bg=bg_col, justify=tk.LEFT, relief=tk.SUNKEN, bd=1, padx=5, pady=5)
        lbl_data.pack(fill=tk.X, padx=10, pady=5)

        return {"cv": cv, "time": lbl_time, "data": lbl_data, "graph": cv_graph, "col": ray_col, "bg": bg_col}

    def draw_fuzzy_graph(self, cv, p):
        # p = [c_max, m_min, m_max, f_min]
        w, h = 500, 100
        max_dist = 150.0
        scale_x = w / max_dist
        scale_y = h - 10 # slightly padded

        def to_xy(x, y_val): # y_val 0.0 to 1.0
            return x * scale_x, h - (y_val * (h-20)) - 10

        # Draw Axes
        cv.create_line(0, h-10, w, h-10, fill="black") # X axis
        
        # Draw Close (Red) - (0,0, c_max)
        pts_c = [to_xy(0, 1), to_xy(0, 1), to_xy(p[0], 0)]
        cv.create_polygon(0, h-10, *pts_c, fill="", outline="red", width=2)
        cv.create_text(10, 10, text="Close", fill="red", anchor="w")

        # Draw Medium (Green) - (m_min, 40, m_max)
        pts_m = [to_xy(p[1], 0), to_xy(40, 1), to_xy(p[2], 0)]
        cv.create_polygon(*pts_m, fill="", outline="green", width=2)
        cv.create_text(w/2, 10, text="Medium", fill="green", anchor="c")

        # Draw Far (Blue) - (f_min, 100, max)
        pts_f = [to_xy(p[3], 0), to_xy(100, 1), to_xy(max_dist, 1), to_xy(max_dist, 0)]
        cv.create_polygon(*pts_f, fill="", outline="blue", width=2)
        cv.create_text(w-10, 10, text="Far", fill="blue", anchor="e")

    def setup_sim(self):
        # Create Robot State Objects
        self.bot_std = self.create_bot_state(MANUAL_PARAMS, self.panel_std)
        self.bot_opt = self.create_bot_state(self.opt_params, self.panel_opt)
        
        self.draw_map(self.panel_std["cv"])
        self.draw_map(self.panel_opt["cv"])

    def create_bot_state(self, params, panel):
        cv = panel["cv"]
        poly = cv.create_polygon(0, 0, 0, 0, fill=panel["col"])
        rays = [cv.create_line(0,0,0,0, fill="red") for _ in range(5)]
        return {
            "x": START_POSE[0], "y": START_POSE[1], "t": START_POSE[2],
            "active": True, "steps": 0, "brain": FuzzyBrain(params),
            "canvas": cv, "poly": poly, "rays": rays, 
            "panel": panel, "start_time": time.time(), "final_time": 0.0
        }

    def draw_map(self, cv):
        cv.delete("obs")
        cv.create_oval(GOAL[0]-10, GOAL[1]-10, GOAL[0]+10, GOAL[1]+10, fill="green", tags="obs")
        for i, o in enumerate(OBS_COMPLEX):
            col = "black" if i < 4 else ("gray" if i < 9 else "red")
            cv.create_rectangle(o, fill=col, tags="obs")

    def reset_sim(self):
        self.bot_std["active"] = False
        self.bot_opt["active"] = False
        self.root.after(100, self.setup_sim)

    def get_sensors(self, x, y, t, cv, rays):
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
            if rays: cv.coords(rays[i], x, y, x+closest*vx, y+closest*vy)
        return readings

    def update_bot(self, bot):
        if not bot["active"]: return

        # 1. Update Timer
        elapsed = time.time() - bot["start_time"]
        bot["panel"]["time"].config(text=f"Time: {elapsed:.2f}s")

        x, y, t = bot["x"], bot["y"], bot["t"]
        sensors = self.get_sensors(x, y, t, bot["canvas"], bot["rays"])
        
        dx, dy = GOAL[0] - x, GOAL[1] - y
        goal_heading = math.atan2(dy, dx)
        angle_err = (goal_heading - t + math.pi) % (2 * math.pi) - math.pi
        
        # 2. Compute (Get Debug Data) 
        speed, turn, dbg_front = bot["brain"].compute(sensors, angle_err)
        
        # 3. Update Fuzzification Display
        f_txt = (f"FRONT SENSOR ({sensors[0]:.1f}px):\n"
                 f"  Close: {dbg_front['C']:.2f} | Med: {dbg_front['M']:.2f} | Far: {dbg_front['F']:.2f}\n"
                 f"OUTPUT:\n"
                 f"  Speed: {speed:.2f} | Turn: {turn:.2f}")
        bot["panel"]["data"].config(text=f_txt)

        new_t = t + turn
        new_x = x + math.cos(new_t) * speed
        new_y = y + math.sin(new_t) * speed
        
        # 4. Check Collision/Goal
        hit = False
        for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
            if ox1 < new_x < ox2 and oy1 < new_y < oy2: hit = True; break
        
        if hit:
            bot["active"] = False
            bot["panel"]["time"].config(fg="red", text=f"CRASH: {elapsed:.2f}s")
            bot["canvas"].itemconfig(bot["poly"], fill="red")
        elif math.hypot(dx, dy) < 15:
            bot["active"] = False
            bot["panel"]["time"].config(fg="green", text=f"GOAL: {elapsed:.2f}s")
            bot["canvas"].itemconfig(bot["poly"], fill="gold")
        else:
            bot["x"], bot["y"], bot["t"] = new_x, new_y, new_t
            bot["steps"] += 1
            
            # Draw Robot
            r = 12
            bot["canvas"].coords(bot["poly"], 
                new_x + r*math.cos(new_t), new_y + r*math.sin(new_t),
                new_x + r*math.cos(new_t+2.5), new_y + r*math.sin(new_t+2.5),
                new_x + r*math.cos(new_t-2.5), new_y + r*math.sin(new_t-2.5)
            )

    def run_loop(self):
        self.update_bot(self.bot_std)
        self.update_bot(self.bot_opt)
        self.root.after(30, self.run_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ComparisonApp(root)
    root.mainloop()