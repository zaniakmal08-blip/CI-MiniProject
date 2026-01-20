import tkinter as tk
import math

# ==========================================
# 1. CONFIGURATION & MAP DATA
# ==========================================
GOAL = (200, 40)
START_POSE = (360, 460, 3.14)

OBS_COMPLEX = [
    # Black Obstacles (Border walls)
    (0, 0, 400, 10), (0, 490, 400, 500), (0, 0, 10, 500), (390, 0, 400, 500),
    # Gray Obstacles
    (80, 400, 400, 410), (0, 300, 320, 310), (80, 200, 400, 210),
    (0, 120, 150, 130), (250, 120, 400, 130),
    
    # --- RED OBSTACLES ---
    (180, 410, 190, 440), 
    (240, 250, 250, 300),
    (100, 150, 120, 180), 
    (200, 210, 210, 240),
    (80, 210, 90, 240),
    (150, 310, 160, 350)
]

# ==========================================
# 2. FUZZY LOGIC BRAIN
# ==========================================
class FuzzyLogicBrain:
    def __init__(self):
        # --- INPUTS ---
        self.d_close = [0, 0, 40]      # Close: 0-40px
        self.d_med   = [10, 40, 50]     # Medium: ~5-50px
        self.d_far   = [40, 100, 1000] # Far: >40px

        self.a_right    = [-3.14, -1.0, -0.1]
        self.a_straight = [-0.3, 0.0, 0.3]
        self.a_left     = [0.1, 1.0, 3.14]

        # --- OUTPUTS ---
        self.turn_out = {
            "Hard_Right": -0.8, "Soft_Right": -0.3, "Straight": 0.0,
            "Soft_Left": 0.3, "Hard_Left": 0.8
        }
        self.speed_out = {
            "Stop": 0.0, "Slow": 2.0, "Medium": 4.0, "Fast": 7.0
        }

    def trimf(self, x, params):
        a, b, c = params
        if x <= a or x >= c: return 0.0
        if a < x <= b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def compute(self, sensors, goal_angle):
        # --- 1. FUZZIFICATION ---
        def get_mfs(val):
            return {
                "C": self.trimf(val, self.d_close),
                "M": self.trimf(val, self.d_med),
                "F": self.trimf(val, self.d_far)
            }
        s_mfs = [get_mfs(d) for d in sensors]
        # Sensor Indices:
        # 0: Front
        # 1: Front-Left (FL)
        # 2: Front-Right (FR)
        # 3: Side-Left (Left)
        # 4: Side-Right (Right)

        g_Right = self.trimf(goal_angle, self.a_right)
        g_Str   = self.trimf(goal_angle, self.a_straight)
        g_Left  = self.trimf(goal_angle, self.a_left)

        # --- 2. RULE EVALUATION ---
        turn_rules  = {k: 0.0 for k in self.turn_out}
        speed_rules = {k: 0.0 for k in self.speed_out}

        def fire(strength, turn_action, speed_action):
            turn_rules[turn_action]   = max(turn_rules[turn_action], strength)
            speed_rules[speed_action] = max(speed_rules[speed_action], strength)

        # RULE 1: CRITICAL FRONT (Panic)
        front_danger = s_mfs[0]["C"]
        if sensors[1] < sensors[2]:
            fire(front_danger, "Hard_Right", "Slow")
        else:
            fire(front_danger, "Hard_Left", "Slow")

        # RULE 2: CORRIDOR (Squeezed on both sides)
        in_corridor = min(s_mfs[3]["C"], s_mfs[4]["C"])
        fire(in_corridor, "Straight", "Medium")

        # RULE 3: CENTERING
        fire(min(s_mfs[3]["C"], s_mfs[4]["F"]), "Soft_Right", "Medium") # Left Close
        fire(min(s_mfs[4]["C"], s_mfs[3]["F"]), "Soft_Left", "Medium")  # Right Close

        # RULE 4: DIAGONAL AVOIDANCE
        fire(s_mfs[1]["C"], "Soft_Right", "Slow") # FL Close
        fire(s_mfs[2]["C"], "Soft_Left",  "Slow") # FR Close

        # RULE 5: GOAL SEEKING (Safe forward motion)
        safe_front = max(s_mfs[0]["F"], s_mfs[0]["M"])
        safe_left  = max(s_mfs[1]["F"], s_mfs[1]["M"])
        safe_right = max(s_mfs[2]["F"], s_mfs[2]["M"])
        is_safe = min(safe_front, safe_left, safe_right)
        speed_level = "Fast" if min(s_mfs[0]["F"], s_mfs[1]["F"]) > 0.5 else "Medium"
        fire(min(is_safe, g_Left),  "Soft_Left",  speed_level)
        fire(min(is_safe, g_Right), "Soft_Right", speed_level)
        fire(min(is_safe, g_Str),   "Straight",   speed_level)

        # RULE 6: BALANCED PATH (Left & Right both Medium → go Straight)
        balanced_path = min(s_mfs[3]["M"], s_mfs[4]["M"])
        fire(balanced_path, "Straight", "Medium")

        # ==========================================================
        # NEW RULE 7: CORNER / WALL HUGGING (YOUR NEW RULES)
        # ==========================================================
        
        # 7a. IF Right is Close AND Front-Right is Close AND Front is Medium -> Soft Left, Slow
        #     Sensors: Right(4), FR(2), Front(0)
        rule_7a_strength = min(s_mfs[4]["C"], s_mfs[2]["C"], s_mfs[0]["M"])
        fire(rule_7a_strength, "Soft_Left", "Slow")

        # 7b. IF Left is Close AND Front-Left is Close AND Front is Medium -> Soft Right, Slow
        #     Sensors: Left(3), FL(1), Front(0)
        rule_7b_strength = min(s_mfs[3]["C"], s_mfs[1]["C"], s_mfs[0]["M"])
        fire(rule_7b_strength, "Soft_Right", "Slow")
        
        # ==========================================================


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
        return final_speed, final_turn

# ==========================================
# 3. SIMULATION APP
# ==========================================
class FuzzySimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Robot (With New Wall Rules)")
        self.root.geometry("600x680")

        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack(pady=10)
        self.lbl_info = tk.Label(root, text="Sensors: -", font=("Consolas", 10))
        self.lbl_info.pack()

        # --- BUTTON CONTAINER ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="RESET", command=self.reset_robot, bg="red", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        self.btn_pause = tk.Button(btn_frame, text="STOP", command=self.toggle_pause, bg="orange", fg="black", width=10)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        # Draw goal and obstacles
        self.canvas.create_oval(GOAL[0]-10, GOAL[1]-10, GOAL[0]+10, GOAL[1]+10, fill="green")
        for i, o in enumerate(OBS_COMPLEX):
            col = "black" if i < 4 else ("gray" if i < 9 else "red")
            self.canvas.create_rectangle(o, fill=col)

        # Robot and sensor rays
        self.poly = self.canvas.create_polygon(0, 0, 0, 0, fill="blue")
        self.ray_lines = [self.canvas.create_line(0, 0, 0, 0, fill="red", width=1) for _ in range(5)]

        self.brain = FuzzyLogicBrain()
        self.paused = False
        self.reset_robot()
        self.run_loop()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="RESUME", bg="green", fg="white")
        else:
            self.btn_pause.config(text="STOP", bg="orange", fg="black")

    def reset_robot(self):
        self.state = {"x": START_POSE[0], "y": START_POSE[1], "t": START_POSE[2], "active": True}
        self.canvas.itemconfig(self.poly, fill="blue")

    def get_sensors(self, x, y, t):
        # Sensor angles: F, FL, FR, SL, SR
        angles = [0, 0.785, -0.785, 1.57, -1.57]
        readings = []
        MAX_RANGE = 150.0

        for i, offset in enumerate(angles):
            ray_t = t + offset
            vx, vy = math.cos(ray_t), math.sin(ray_t)
            
            closest_dist = MAX_RANGE

            for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
                if abs(vx) > 0.0001:
                    t1 = (ox1 - x) / vx
                    if 0 < t1 < closest_dist:
                        if oy1 <= y + t1 * vy <= oy2: closest_dist = t1
                    t2 = (ox2 - x) / vx
                    if 0 < t2 < closest_dist:
                        if oy1 <= y + t2 * vy <= oy2: closest_dist = t2
                if abs(vy) > 0.0001:
                    t3 = (oy1 - y) / vy
                    if 0 < t3 < closest_dist:
                        if ox1 <= x + t3 * vx <= ox2: closest_dist = t3
                    t4 = (oy2 - y) / vy
                    if 0 < t4 < closest_dist:
                        if ox1 <= x + t4 * vx <= ox2: closest_dist = t4

            readings.append(closest_dist)
            end_x = x + closest_dist * vx
            end_y = y + closest_dist * vy
            self.canvas.coords(self.ray_lines[i], x, y, end_x, end_y)
        return readings

    def run_loop(self):
        if self.paused:
            self.root.after(100, self.run_loop)
            return

        if not self.state["active"]:
            self.root.after(100, self.run_loop)
            return

        x, y, t = self.state["x"], self.state["y"], self.state["t"]
        sensors = self.get_sensors(x, y, t)

        dx, dy = GOAL[0] - x, GOAL[1] - y
        goal_heading = math.atan2(dy, dx)
        angle_err = (goal_heading - t + math.pi) % (2 * math.pi) - math.pi

        speed, turn = self.brain.compute(sensors, angle_err)
        new_t = t + turn
        new_x, new_y = x + math.cos(new_t) * speed, y + math.sin(new_t) * speed

        # Collision check
        hit = False
        for ox1, oy1, ox2, oy2 in OBS_COMPLEX:
            if ox1 < new_x < ox2 and oy1 < new_y < oy2:
                hit = True
                break

        if hit:
            self.state["active"] = False
            self.canvas.itemconfig(self.poly, fill="red")
        else:
            self.state["x"], self.state["y"], self.state["t"] = new_x, new_y, new_t

        if math.hypot(dx, dy) < 15:
            self.state["active"] = False
            self.canvas.itemconfig(self.poly, fill="green")

        r = 12
        pts = [
            new_x + r * math.cos(new_t), new_y + r * math.sin(new_t),
            new_x + r * math.cos(new_t + 2.5), new_y + r * math.sin(new_t + 2.5),
            new_x + r * math.cos(new_t - 2.5), new_y + r * math.sin(new_t - 2.5)
        ]
        self.canvas.coords(self.poly, *pts)
        self.lbl_info.config(text=f"Spd:{speed:.1f} Turn:{turn:.2f}")
        self.root.after(30, self.run_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzySimApp(root)
    root.mainloop()