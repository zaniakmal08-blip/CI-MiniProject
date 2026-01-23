import tkinter as tk
import math
import json
import os
import time
import random

# ==========================================
# 1. CONFIGURATION & MAPS
# ==========================================
GOAL = (200, 40)
DEFAULT_START = (360, 460, 3.14)

# --- Map 1: Complex (The Original Maze) ---
OBS_COMPLEX = [
    (0, 0, 400, 10), (0, 490, 400, 500), (0, 0, 10, 500), (390, 0, 400, 500),
    (80, 400, 400, 410), (0, 300, 320, 310), (80, 200, 400, 210),
    (0, 120, 150, 130), (250, 120, 400, 130),
    (180, 410, 190, 440), (240, 250, 250, 300), (100, 150, 120, 180),
    (200, 210, 210, 240), (80, 210, 90, 240), (150, 310, 160, 350), (300, 340, 310, 400)
]

# --- Map 2: Simple (From your new code) ---
OBS_SIMPLE = [
    (0, 0, 400, 10), (0, 490, 400, 500), (0, 0, 10, 500), (390, 0, 400, 500), # Borders
    (50, 180, 280, 200), # Horizontal Bar 1
    (200, 350, 400, 330), # Horizontal Bar 2
]

MANUAL_PARAMS = [40, 10, 50, 40]

# ==========================================
# 2. SHARED FUZZY LOGIC CLASS
# ==========================================
class FuzzyBrain:
    def __init__(self, params):
        self.params = params
        c_max, m_min, m_max, f_min = params
        
        self.d_close = [-10, 0, c_max]
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
        debug_front = s_mfs[0]
        
        g_Right = self.trimf(goal_angle, self.a_right)
        g_Str   = self.trimf(goal_angle, self.a_straight)
        g_Left  = self.trimf(goal_angle, self.a_left)

        turn_rules = {k: 0.0 for k in self.turn_out}
        speed_rules = {k: 0.0 for k in self.speed_out}
        
        def fire(strength, turn_a, speed_a):
            turn_rules[turn_a] = max(turn_rules[turn_a], strength)
            speed_rules[speed_a] = max(speed_rules[speed_a], strength)

        # RULES
        if sensors[1] < sensors[2]: fire(s_mfs[0]["C"], "Hard_Right", "Slow")
        else: fire(s_mfs[0]["C"], "Hard_Left", "Slow")

        fire(min(s_mfs[3]["C"], s_mfs[4]["C"]), "Straight", "Medium") 
        fire(min(s_mfs[3]["C"], s_mfs[4]["F"]), "Soft_Right", "Medium")
        fire(min(s_mfs[4]["C"], s_mfs[3]["F"]), "Soft_Left", "Medium")
        fire(s_mfs[1]["C"], "Soft_Right", "Slow")
        fire(s_mfs[2]["C"], "Soft_Left", "Slow")

        fire(min(s_mfs[4]["C"], s_mfs[2]["C"], s_mfs[0]["M"]), "Soft_Left", "Slow")
        fire(min(s_mfs[3]["C"], s_mfs[1]["C"], s_mfs[0]["M"]), "Soft_Right", "Slow")

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
        return speed, turn, debug_front

# ==========================================
# 3. COMPARISON APP
# ==========================================
class ComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Standard vs Optimized Fuzzy Logic (Analysis Mode)")
        self.root.geometry("1200x1000")

        # Stats Storage
        self.stats_std = {"wins": 0, "total": 1, "status": "running", "time": 0.0}
        self.stats_opt = {"wins": 0, "total": 1, "status": "running", "time": 0.0}
        self.race_score = {"std": 0, "opt": 0}
        self.attempt_count = 1 
        
        # --- Map State Management ---
        self.current_fixed_map = OBS_COMPLEX # Default to complex
        self.random_obstacles = []           # Dynamic additions

        # Load Params
        self.opt_params = MANUAL_PARAMS
        if os.path.exists("best_params.json"):
            try:
                with open("best_params.json", "r") as f: 
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.opt_params = [data["close_max"], data["med_min"], data["med_max"], data["far_min"]]
                    else:
                        self.opt_params = data
                print(f"Loaded optimized parameters: {self.opt_params}")
            except Exception as e: 
                print(f"Error loading params: {e}")

        # --- LAYOUT SETUP ---
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # BUTTONS
        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side=tk.TOP, pady=5)

        tk.Button(btn_frame, text="RERUN (SIMPLE MAP)", command=self.reset_simple, 
                  bg="#90EE90", font=("Arial", 10, "bold"), width=22).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="RERUN (COMPLEX MAP)", command=self.reset_complex, 
                  bg="#ADD8E6", font=("Arial", 10, "bold"), width=22).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="RERUN (RANDOM OBSTACLES)", command=self.reset_random, 
                  bg="orange", font=("Arial", 10, "bold"), width=25).pack(side=tk.LEFT, padx=5)

        self.lbl_race = tk.Label(top_frame, text="RACE SCORE: Std [ 0 ] - [ 0 ] Opt", font=("Arial", 14, "bold"), fg="purple")
        self.lbl_race.pack(side=tk.BOTTOM, pady=2)

        # Log Panel
        log_frame = tk.LabelFrame(root, text="Simulation History Log", font=("Arial", 10, "bold"))
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=4, font=("Consolas", 9), state=tk.DISABLED, bg="#f4f4f4")
        scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Simulation Panels
        f_std = tk.Frame(root, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        f_std.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.panel_std = self.create_panel(f_std, "STANDARD", MANUAL_PARAMS, "#f0f0f0", "blue")

        f_opt = tk.Frame(root, bg="#e0ffe0", bd=2, relief=tk.GROOVE)
        f_opt.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.panel_opt = self.create_panel(f_opt, "OPTIMIZED GA", self.opt_params, "#e0ffe0", "green")

        self.current_start = DEFAULT_START
        self.generate_random_obstacles() 
        self.setup_sim()
        self.run_loop()

    def create_panel(self, parent, title, params, bg_col, ray_col):
        tk.Label(parent, text=title, font=("Arial", 12, "bold"), bg=bg_col).pack(pady=1, side=tk.TOP)
        cv = tk.Canvas(parent, width=450, height=510, bg="white")
        cv.pack(side=tk.TOP)
        lbl_time = tk.Label(parent, text="Time: 0.0s", font=("Consolas", 12, "bold"), bg=bg_col)
        lbl_time.pack(pady=1, side=tk.TOP)
        lbl_stats = tk.Label(parent, text="Success Rate: 0% | Attempts: 1", font=("Arial", 10, "bold"), bg=bg_col, fg="darkblue")
        lbl_stats.pack(pady=1, side=tk.TOP)
        lbl_data = tk.Label(parent, text="Waiting...", font=("Consolas", 11), bg="white", justify=tk.LEFT, relief=tk.RAISED, bd=2, padx=5, pady=5)
        lbl_data.pack(fill=tk.X, padx=10, pady=5, side=tk.TOP)
        return {"cv": cv, "time": lbl_time, "stats": lbl_stats, "data": lbl_data, "col": ray_col, "bg": bg_col}

    def write_log(self, text, color="black"):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def setup_sim(self):
        self.bot_std = self.create_bot_state("Standard", MANUAL_PARAMS, self.panel_std, self.current_start, self.stats_std)
        self.bot_opt = self.create_bot_state("Optimized", self.opt_params, self.panel_opt, self.current_start, self.stats_opt)
        
        self.draw_map(self.panel_std["cv"])
        self.draw_map(self.panel_opt["cv"])
        self.update_stats_display(self.panel_std, self.stats_std)
        self.update_stats_display(self.panel_opt, self.stats_opt)
        
        self.stats_std["status"] = "running"
        self.stats_opt["status"] = "running"
        self.round_checked = False

    def create_bot_state(self, name, params, panel, start_pos, stats_ref):
        cv = panel["cv"]
        poly = cv.create_polygon(0, 0, 0, 0, fill=panel["col"])
        rays = [cv.create_line(0,0,0,0, fill="red") for _ in range(5)]
        return {
            "name": name,
            "x": start_pos[0], "y": start_pos[1], "t": start_pos[2],
            "active": True, "steps": 0, "brain": FuzzyBrain(params),
            "canvas": cv, "poly": poly, "rays": rays, 
            "panel": panel, "start_time": time.time(),
            "stats": stats_ref,
            "path_len": 0.0,    
            "smoothness": 0.0   
        }

    def draw_map(self, cv):
        cv.delete("obs")
        cv.create_oval(GOAL[0]-10, GOAL[1]-10, GOAL[0]+10, GOAL[1]+10, fill="green", tags="obs")
        
        # Draw Currently Selected Fixed Map
        for i, o in enumerate(self.current_fixed_map):
            col = "black" if i < 4 else ("gray" if i < 9 else "red")
            cv.create_rectangle(o, fill=col, tags="obs")
            
        # Draw Random Obstacles (if any)
        for obs in self.random_obstacles:
            cv.create_rectangle(obs, fill="red", outline="black", tags="obs")

    def generate_random_obstacles(self):
        self.random_obstacles = [] 
        for _ in range(3):
            attempts = 0
            while attempts < 50:
                attempts += 1
                rx = random.randint(30, 370)
                ry = random.randint(50, 450)
                if math.hypot(rx - GOAL[0], ry - GOAL[1]) < 60: continue
                
                hit_wall = False
                r_rect = (rx, ry, rx+20, ry+20)
                
                # Check against CURRENT Fixed Walls
                for ox1, oy1, ox2, oy2 in self.current_fixed_map:
                    if (rx < ox2 and rx+20 > ox1 and ry < oy2 and ry+20 > oy1):
                        hit_wall = True; break
                
                if not hit_wall:
                    for exist_obs in self.random_obstacles:
                        ex1, ey1, ex2, ey2 = exist_obs
                        if (rx < ex2 and rx+20 > ex1 and ry < ey2 and ry+20 > ey1):
                            hit_wall = True; break
                
                if not hit_wall:
                    self.random_obstacles.append(r_rect)
                    break 

    def get_valid_random_start(self):
        while True:
            rx = random.randint(20, 380)
            ry = random.randint(250, 480)
            hit = False
            
            # Check against CURRENT fixed map
            for ox1, oy1, ox2, oy2 in self.current_fixed_map:
                if (ox1-15) < rx < (ox2+15) and (oy1-15) < ry < (oy2+15):
                    hit = True; break
            
            for obs in self.random_obstacles:
                ox1, oy1, ox2, oy2 = obs
                if (ox1-15) < rx < (ox2+15) and (oy1-15) < ry < (oy2+15):
                    hit = True; break
            if not hit:
                rt = random.uniform(-3.14, 3.14)
                return (rx, ry, rt)

    # --- RESET LOGIC ---
    def reset_common_logic(self, mode_name):
        any_active = self.bot_std["active"] or self.bot_opt["active"]
        any_goal = (self.stats_std["status"] == "goal") or (self.stats_opt["status"] == "goal")
        
        self.bot_std["active"] = False
        self.bot_opt["active"] = False
        
        if any_active and not any_goal:
            self.write_log(f"--- ATTEMPT {self.attempt_count} ABORTED ({mode_name}) ---", "red")
        else:
            self.attempt_count += 1
            self.stats_std["total"] += 1
            self.stats_opt["total"] += 1
            
            if self.bot_std["active"]: 
                 s_rate = (self.stats_std["wins"] / self.stats_std["total"] * 100)
                 self.write_log(f"[{self.bot_std['name']}] STOPPED | Steps: {self.bot_std['steps']} | SR: {s_rate:.1f}%")
            if self.bot_opt["active"]:
                 s_rate = (self.stats_opt["wins"] / self.stats_opt["total"] * 100)
                 self.write_log(f"[{self.bot_opt['name']}] STOPPED | Steps: {self.bot_opt['steps']} | SR: {s_rate:.1f}%")

            self.write_log(f"--- STARTING ATTEMPT {self.attempt_count} ({mode_name}) ---", "blue")
        
        self.update_stats_display(self.panel_std, self.stats_std)
        self.update_stats_display(self.panel_opt, self.stats_opt)

    def reset_simple(self):
        self.reset_common_logic("SIMPLE")
        self.current_fixed_map = OBS_SIMPLE # Switch to Simple
        self.random_obstacles = [] 
        self.current_start = DEFAULT_START 
        self.root.after(100, self.setup_sim)

    def reset_complex(self):
        self.reset_common_logic("COMPLEX")
        self.current_fixed_map = OBS_COMPLEX # Switch to Complex
        self.random_obstacles = [] 
        self.current_start = DEFAULT_START 
        self.root.after(100, self.setup_sim)

    def reset_random(self):
        self.reset_common_logic("RANDOM")
        self.current_fixed_map = OBS_COMPLEX # Use Complex as base
        self.generate_random_obstacles()
        self.current_start = self.get_valid_random_start()
        self.root.after(100, self.setup_sim)

    def update_stats_display(self, panel, stats):
        total = stats["total"]
        wins = stats["wins"]
        safe_total = max(1, total)
        rate = (wins / safe_total * 100)
        panel["stats"].config(text=f"Success Rate: {rate:.1f}% | Attempts: {total}")

    def check_race_winner(self):
        if not self.round_checked and not self.bot_std["active"] and not self.bot_opt["active"]:
            self.round_checked = True
            s_std = self.stats_std["status"]
            s_opt = self.stats_opt["status"]
            t_std = self.stats_std["time"]
            t_opt = self.stats_opt["time"]

            self.write_log(f"RACE FINISHED:")

            if s_std == "crash" and s_opt == "crash": 
                self.write_log(f"  > DRAW (Both Crashed)", "red")
                return 
            
            winner = "None"
            if s_std == "goal" and s_opt == "crash": 
                self.race_score["std"] += 1
                winner = "Standard"
            elif s_opt == "goal" and s_std == "crash": 
                self.race_score["opt"] += 1
                winner = "Optimized"
            elif s_std == "goal" and s_opt == "goal":
                if t_std < t_opt: 
                    self.race_score["std"] += 1
                    winner = "Standard (Faster)"
                else: 
                    self.race_score["opt"] += 1
                    winner = "Optimized (Faster)"
            
            self.lbl_race.config(text=f"RACE SCORE: Std [ {self.race_score['std']} ] - [ {self.race_score['opt']} ] Opt")
            self.write_log(f"  > WINNER: {winner}", "green")

    def get_sensors(self, x, y, t, cv, rays):
        angles = [0, 0.785, -0.785, 1.57, -1.57]
        readings = []
        
        # Combine currently selected map + dynamic obstacles
        all_obs = list(self.current_fixed_map)
        all_obs.extend(self.random_obstacles) 
            
        for i, offset in enumerate(angles):
            ray_t = t + offset
            vx, vy = math.cos(ray_t), math.sin(ray_t)
            closest = 150.0
            for ox1, oy1, ox2, oy2 in all_obs:
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

        elapsed = time.time() - bot["start_time"]
        
        bot["panel"]["time"].config(text=f"Time: {elapsed:.2f}s | Steps: {bot['steps']}")

        x, y, t = bot["x"], bot["y"], bot["t"]
        sensors = self.get_sensors(x, y, t, bot["canvas"], bot["rays"])
        
        dx, dy = GOAL[0] - x, GOAL[1] - y
        goal_heading = math.atan2(dy, dx)
        angle_err = (goal_heading - t + math.pi) % (2 * math.pi) - math.pi
        
        speed, turn, dbg_front = bot["brain"].compute(sensors, angle_err)

        bot["path_len"] += speed
        bot["smoothness"] += abs(turn)

        f_txt = (f"FRONT: {sensors[0]:.0f}px (C:{dbg_front['C']:.1f} M:{dbg_front['M']:.1f} F:{dbg_front['F']:.1f})\n"
                 f"METRICS:\n"
                 f"  Time Steps: {bot['steps']}\n"
                 f"  Smoothness:  {bot['smoothness']:.2f}")
        bot["panel"]["data"].config(text=f_txt)

        new_t = t + turn
        new_x = x + math.cos(new_t) * speed
        new_y = y + math.sin(new_t) * speed
        
        hit = False
        all_obs = list(self.current_fixed_map)
        all_obs.extend(self.random_obstacles) 
        
        for ox1, oy1, ox2, oy2 in all_obs:
            if ox1 < new_x < ox2 and oy1 < new_y < oy2: hit = True; break
        
        if hit:
            bot["active"] = False
            bot["panel"]["time"].config(fg="red", text=f"CRASH: {elapsed:.2f}s")
            bot["canvas"].itemconfig(bot["poly"], fill="red")
            bot["stats"]["status"] = "crash"
            bot["stats"]["time"] = elapsed
            
            safe_total = max(1, bot["stats"]["total"])
            s_rate = (bot["stats"]["wins"] / safe_total * 100)
            
            self.update_stats_display(bot["panel"], bot["stats"])
            self.write_log(f"[{bot['name']}] CRASH | T: {elapsed:.2f}s | Steps: {bot['steps']} | Sm: {bot['smoothness']:.2f} | SR: {s_rate:.1f}%")
            self.check_race_winner()
            
        elif math.hypot(dx, dy) < 15:
            bot["active"] = False
            bot["panel"]["time"].config(fg="green", text=f"GOAL: {elapsed:.2f}s")
            bot["canvas"].itemconfig(bot["poly"], fill="gold")
            bot["stats"]["wins"] += 1
            bot["stats"]["status"] = "goal"
            bot["stats"]["time"] = elapsed
            
            safe_total = max(1, bot["stats"]["total"])
            s_rate = (bot["stats"]["wins"] / safe_total * 100)
            
            self.update_stats_display(bot["panel"], bot["stats"])
            self.write_log(f"[{bot['name']}] GOAL! | T: {elapsed:.2f}s | Steps: {bot['steps']} | Sm: {bot['smoothness']:.2f} | SR: {s_rate:.1f}%")
            self.check_race_winner()
            
        else:
            bot["x"], bot["y"], bot["t"] = new_x, new_y, new_t
            bot["steps"] += 1
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