import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import random, time, heapq
from collections import deque
from threading import Thread

PUZZLE_SIZE = 3

# ====================  THAY ĐỔI GOAL TẠI ĐÂY ====================
CUSTOM_GOALS = {
    3: (1, 2, 3, 4, 5, 6, 7, 8,0), # Goal mặc định 3x3: ô trống cuối
    #3: (0, 1, 2, 3, 4, 5, 6, 7, 8), # Goal tùy chỉnh 3x3: ô trống đầu
    #3: (1, 2, 3, 4,0, 5, 6, 7, 8), # Goal tùy chỉnh 3x3: ô trống giữa
    4: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)  # Goal mặc định 4x4
}


def create_goal(size):
    """Trả về goal tùy chỉnh"""
    return CUSTOM_GOALS. get(size, tuple(list(range(1, size * size)) + [0]))

# -------------------- Heuristics --------------------
def manhattan_distance(state, size):
    """Manhattan distance """
    goal = create_goal(size)
    total = 0
    for i, val in enumerate(state):
        if val != 0:
            goal_pos = goal. index(val)  # Tìm vị trí của val trong goal
            total += abs(i // size - goal_pos // size) + abs(i % size - goal_pos % size)
    return total

def linear_conflict(state, size):
    """Linear conflict """
    goal = create_goal(size)
    distance = manhattan_distance(state, size)
    conflicts = 0
    
    # Tạo mapping: giá trị -> vị trí trong goal
    goal_positions = {val: idx for idx, val in enumerate(goal)}
    
    # Kiểm tra xung đột hàng
    for row in range(size):
        for col in range(size):
            val = state[row * size + col]
            if val != 0:
                goal_pos = goal_positions[val]
                goal_row = goal_pos // size
                
                if goal_row == row:  # val thuộc hàng này trong goal
                    for k in range(col + 1, size):
                        val2 = state[row * size + k]
                        if val2 != 0:
                            goal_pos2 = goal_positions[val2]
                            if goal_pos2 // size == row and goal_pos2 % size < goal_pos % size:
                                conflicts += 1
    
    # Kiểm tra xung đột cột
    for col in range(size):
        for row in range(size):
            val = state[row * size + col]
            if val != 0:
                goal_pos = goal_positions[val]
                goal_col = goal_pos % size
                
                if goal_col == col:  # val thuộc cột này trong goal
                    for k in range(row + 1, size):
                        val2 = state[k * size + col]
                        if val2 != 0:
                            goal_pos2 = goal_positions[val2]
                            if goal_pos2 % size == col and goal_pos2 // size < goal_pos // size:
                                conflicts += 1
    
    return distance + 2 * conflicts

# -------------------- Algorithms --------------------
def bfs_solve(start, size, timeout=30):
    """BFS - Tìm kiếm theo chiều rộng"""
    goal = create_goal(size)
    if start == goal:
        return [], 0
    
    queue = deque([(start, [])])
    explored = {start}
    nodes = 0
    max_nodes = 50000 if size == 3 else 20000
    start_time = time. time()
    
    while queue and nodes < max_nodes:
        if time.time() - start_time > timeout:
            return None, nodes
        
        state, path = queue.popleft()
        nodes += 1
        
        if state == goal:
            return path, nodes
        
        for next_state, move in get_neighbors(state, size):
            if next_state not in explored:
                explored.add(next_state)
                queue.append((next_state, path + [move]))
    
    return None, nodes

def dfs_solve(start, size, timeout=30, max_depth=50):
    """DFS - Tìm kiếm theo chiều sâu"""
    goal = create_goal(size)
    if start == goal:
        return [], 0
    
    stack = [(start, [], 0)]
    explored = set()
    nodes = 0
    max_nodes = 50000 if size == 3 else 100000
    start_time = time.time()
    
    while stack and nodes < max_nodes:
        if time.time() - start_time > timeout:
            return None, nodes
        
        state, path, depth = stack.pop()
        
        if state in explored or depth > max_depth:
            continue
        
        explored. add(state)
        nodes += 1
        
        if state == goal:
            return path, nodes
        
        for next_state, move in get_neighbors(state, size):
            if next_state not in explored:
                stack.append((next_state, path + [move], depth + 1))
    
    return None, nodes

def a_star_solve(start, size, timeout=30):
    goal = create_goal(size)
    if start == goal:
        return [], 0
    
    frontier = [(linear_conflict(start, size), 0, start, [])]
    explored = set()
    g_scores = {start: 0}
    nodes = 0
    start_time = time. time()
    
    while frontier:
        if time.time() - start_time > timeout:
            return None, nodes
        
        _, cost, current_state, path = heapq.heappop(frontier)
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        nodes += 1
        
        if current_state == goal:
            return path, nodes
        
        for next_state, move in get_neighbors(current_state, size):
            if next_state in explored:
                continue
            
            new_cost = cost + 1
            
            if next_state not in g_scores or new_cost < g_scores[next_state]:
                g_scores[next_state] = new_cost
                f_score = new_cost + linear_conflict(next_state, size)
                heapq.heappush(frontier, (f_score, new_cost, next_state, path + [move]))
    
    return None, nodes

# -------------------- Puzzle Logic--------------------
def is_solvable(state, size): 
    """Kiểm tra xem state có thể đạt được goal hay không """
    goal = create_goal(size)
    
    # Tính inversions cho state
    arr_state = [x for x in state if x != 0]
    inv_state = sum(1 for i in range(len(arr_state)) 
                    for j in range(i + 1, len(arr_state)) 
                    if arr_state[i] > arr_state[j])
    
    # Tính inversions cho goal
    arr_goal = [x for x in goal if x != 0]
    inv_goal = sum(1 for i in range(len(arr_goal)) 
                   for j in range(i + 1, len(arr_goal)) 
                   if arr_goal[i] > arr_goal[j])
    
    if size % 2 == 1:  # Odd size (3x3)
        return (inv_state % 2) == (inv_goal % 2)
    else:  # Even size (4x4)
        blank_row_state = size - (state.index(0) // size)
        blank_row_goal = size - (goal.index(0) // size)
        return ((inv_state + blank_row_state) % 2) == ((inv_goal + blank_row_goal) % 2)

def shuffle_state(size):
    goal = create_goal(size)
    state = goal
    moves = random.randint(20, 50) if size == 3 else random.randint(15, 30)
    for _ in range(moves):
        neighbors = get_neighbors(state, size)
        if neighbors:
            state = random.choice(neighbors)[0]
    return state

def get_neighbors(state, size):
    neighbors = []
    blank_pos = state.index(0)
    row, col = divmod(blank_pos, size)
    
    for move, offset, condition in [('U', -size, row > 0), ('D', size, row < size-1),
                                     ('L', -1, col > 0), ('R', 1, col < size-1)]:
        if condition:
            new_state = list(state)
            new_pos = blank_pos + offset
            new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
            neighbors.append((tuple(new_state), move))
    return neighbors

# -------------------- Solution Viewer --------------------
class SolutionViewer:
    def __init__(self, parent, initial_state, moves, size, algorithm):
        self. window = tk.Toplevel(parent)
        self.window. title(f"Chi tiết giải - {algorithm}")
        self.window.geometry("700x600")
        
        self.initial = list(initial_state)
        self.moves = moves
        self.size = size
        self.algorithm = algorithm
        self.step = 0
        self.playing = False
        self.goal = create_goal(size)
        self.move_names = {'U': '↑ Lên', 'D': '↓ Xuống', 'L': '← Trái', 'R': '→ Phải'}
        
        # Tính tất cả states
        self.states = [tuple(self.initial)]
        current = self.initial[:]
        for move in moves:
            blank = current.index(0)
            offset = {'U': -size, 'D': size, 'L': -1, 'R': 1}[move]
            new_pos = blank + offset
            current[blank], current[new_pos] = current[new_pos], current[blank]
            self.states.append(tuple(current))
        
        self. setup_ui()
        self.show_step(0)
    
    def setup_ui(self):
        main = tk.Frame(self. window)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left = tk.Frame(main)
        left. pack(side=tk.LEFT, fill=tk. BOTH, expand=True, padx=(0, 10))
        
        tk.Label(left, text=f"Animation - {self.algorithm}", 
                font=("Arial", 13, "bold")).pack(pady=5)
        self.board_frame = tk.Frame(left)
        self.board_frame.pack(pady=10)
        self.info_label = tk. Label(left, text="", font=("Arial", 10))
        self.info_label. pack(pady=5)
        
        ctrl = tk.Frame(left)
        ctrl. pack(pady=10)
        self.play_btn = tk.Button(ctrl, text="▶ Phát", command=self.toggle_play, width=8)
        self. play_btn.grid(row=0, column=0, padx=2)
        tk.Button(ctrl, text="⮜", command=lambda: self.show_step(0), width=6).grid(row=0, column=1, padx=2)
        tk. Button(ctrl, text="◀", command=self.prev, width=6).grid(row=0, column=2, padx=2)
        tk.Button(ctrl, text="▶", command=self.next, width=6). grid(row=0, column=3, padx=2)
        tk.Button(ctrl, text="⮞", command=lambda: self.show_step(len(self.moves)), width=6). grid(row=0, column=4, padx=2)
        
        self.slider = tk.Scale(left, from_=0, to=len(self.moves), orient=tk. HORIZONTAL,
                              command=lambda v: self.show_step(int(v)) if not self.playing else None)
        self. slider.pack(fill=tk.X, pady=5)
        
        right = tk.Frame(main, width=250)
        right. pack(side=tk.RIGHT, fill=tk.BOTH)
        
        tk.Label(right, text="Các bước giải", font=("Arial", 11, "bold")).pack(pady=5)
        
        self.table = scrolledtext.ScrolledText(right, width=30, height=30, font=("Courier", 9))
        self. table.pack(fill=tk.BOTH, expand=True)
        self.table.tag_configure("highlight", background="yellow", font=("Courier", 9, "bold"))
        
        self.table.insert(tk. END, "Bước | Hướng  | h(n)\n")
        self.table.insert(tk.END, "-----+---------+-----\n")
        self. table.insert(tk.END, f"  0  | Ban đầu | {manhattan_distance(self. states[0], self. size):3d}\n")
        for i, move in enumerate(self. moves, 1):
            h = manhattan_distance(self. states[i], self.size)
            self.table.insert(tk.END, f" {i:2d}  | {self.move_names[move]:7s} | {h:3d}\n")
    
    def draw_board(self, state):
        for w in self.board_frame. winfo_children():
            w.destroy()
        
        for i, val in enumerate(state):
            row, col = i // self.size, i % self.size
            text = str(val) if val else ""
            
            if val == 0:
                bg, fg = "lightgray", "black"
            elif val == self.goal[i]:
                bg, fg = "lightgreen", "darkgreen"
            else:
                bg, fg = "white", "red"
            
            tk.Label(self.board_frame, text=text, width=4, height=2, font=("Arial", 18, "bold"),
                    relief="ridge", borderwidth=2, bg=bg, fg=fg).grid(row=row, column=col, padx=2, pady=2)
    
    def show_step(self, step):
        if 0 <= step <= len(self.moves):
            self. step = step
            state = self.states[step]
            
            self.draw_board(state)
            self.slider.set(step)
            
            self.table.tag_remove("highlight", 1.0, tk.END)
            self.table.tag_add("highlight", f"{step + 3}. 0", f"{step + 4}.0")
            self.table.see(f"{step + 3}.0")
            
            h = manhattan_distance(state, self.size)
            if step == 0:
                txt = f"Ban đầu | h={h} | 0/{len(self.moves)}"
            elif step == len(self.moves):
                txt = f"✓ Hoàn thành!  | {len(self.moves)} bước"
            else:
                txt = f"Bước {step}: {self.move_names[self.moves[step-1]]} | h={h} | {step}/{len(self. moves)}"
            
            self.info_label.config(text=txt)
    
    def next(self):
        if self.step < len(self. moves):
            self.show_step(self. step + 1)
    
    def prev(self):
        if self.step > 0:
            self.show_step(self.step - 1)
    
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn. config(text="⏸ Dừng" if self.playing else "▶ Phát")
        if self.playing:
            self.animate()
    
    def animate(self):
        if self.playing and self.step < len(self.moves):
            self.next()
            self.window.after(500, self.animate)
        else:
            self. playing = False
            self.play_btn. config(text="▶ Phát")

# -------------------- Main GUI --------------------
class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle Solver - A*, BFS, DFS")
        self.solving = False
        self.solutions = {}
        self.last_state = None
        self.setup_ui()
        self.reset_puzzle()

    def setup_ui(self):
        frame_top = tk.Frame(self.root)
        frame_top. pack(pady=10)
        tk.Label(frame_top, text="Kích thước:", font=("Arial", 10)). pack(side=tk.LEFT)
        self.var_size = tk. IntVar(value=3)
        tk.Radiobutton(frame_top, text="3x3", variable=self.var_size, value=3,
                       command=self.change_size).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(frame_top, text="4x4", variable=self.var_size, value=4,
                       command=self.change_size).pack(side=tk.LEFT, padx=5)
        
        algo_frame = tk. Frame(self.root)
        algo_frame.pack(pady=5)
        tk. Label(algo_frame, text="Chọn thuật toán:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.var_algo = tk. StringVar(value="A*")
        tk.Radiobutton(algo_frame, text="A*", variable=self. var_algo, value="A*",
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk. Radiobutton(algo_frame, text="BFS", variable=self.var_algo, value="BFS",
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(algo_frame, text="DFS", variable=self.var_algo, value="DFS",
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Hiển thị Goal hiện tại
        self.goal_label = tk.Label(self.root, text="", font=("Arial", 9), fg="blue")
        self.goal_label.pack(pady=2)
        self.update_goal_display()
        
        self.frame_board = tk.Frame(self.root)
        self.frame_board.pack(pady=10)
        
        frame_btn = tk.Frame(self.root)
        frame_btn. pack(pady=10)
        tk.Button(frame_btn, text="🔀 Xáo trộn", command=self.shuffle, 
                 font=("Arial", 11), width=12, bg="lightblue"). pack(side=tk.LEFT, padx=5)
        self.solve_btn = tk.Button(frame_btn, text="🔍 Xem chi tiết giải", 
                                   command=self.view_solution, font=("Arial", 11), 
                                   width=15, bg="lightgreen")
        self.solve_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(frame_btn, text="📊 So sánh tất cả", command=self.compare_all,
                 font=("Arial", 11), width=15, bg="orange", fg="white").pack(side=tk. LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate', length=400)
        self. info_label = tk. Label(self.root, text="", font=("Arial", 10))
        self.info_label. pack(pady=5)

    def update_goal_display(self):
        """Hiển thị goal hiện tại trên giao diện"""
        goal = create_goal(PUZZLE_SIZE)
        goal_str = " ".join(str(x) if x != 0 else "□" for x in goal)
        self. goal_label.config(text=f"🎯 Goal: [{goal_str}]")

    def change_size(self):
        global PUZZLE_SIZE
        PUZZLE_SIZE = self.var_size.get()
        self.reset_puzzle()
        self.solutions = {}
        self.update_goal_display()

    def reset_puzzle(self):
        self.state = create_goal(PUZZLE_SIZE)
        self.draw_board()
        self.info_label.config(text="")

    def draw_board(self):
        goal = create_goal(PUZZLE_SIZE)
        for w in self.frame_board.winfo_children():
            w.destroy()
        for i, val in enumerate(self.state):
            row, col = divmod(i, PUZZLE_SIZE)
            
            if val == 0:
                bg = "lightgray"
            elif val == goal[i]:
                bg = "lightgreen"
            else:
                bg = "white"
            
            tk.Label(self.frame_board, text=str(val) if val else "", width=4, height=2,
                    font=("Arial", 18), relief="ridge", borderwidth=2,
                    bg=bg). grid(row=row, column=col, padx=1, pady=1)

    def shuffle(self):
        if not self.solving:
            self.state = shuffle_state(PUZZLE_SIZE)
            self.draw_board()
            self.solutions = {}
            self.info_label.config(text="")

    def solve_thread(self, algorithm):
        self. root.after(0, lambda: self.info_label.config(
            text=f"🔍 Đang giải bằng {algorithm}.. .", fg="blue"))
        
        start = time.time()
        
        if algorithm == "A*":
            moves, nodes = a_star_solve(self.state, PUZZLE_SIZE, 30 if PUZZLE_SIZE == 3 else 60)
        elif algorithm == "BFS":
            moves, nodes = bfs_solve(self.state, PUZZLE_SIZE, 30 if PUZZLE_SIZE == 3 else 60)
        else:
            moves, nodes = dfs_solve(self.state, PUZZLE_SIZE, 30 if PUZZLE_SIZE == 3 else 60)
        
        elapsed = time.time() - start
        self.root.after(0, self.solve_complete, algorithm, moves, elapsed, nodes)

    def solve_complete(self, algorithm, moves, elapsed, nodes):
        self.progress.stop()
        self.progress.pack_forget()
        self.solve_btn.config(state='normal')
        self. solving = False
        
        if moves is None:
            messagebox.showerror("Kết quả", 
                f"{algorithm} không tìm thấy lời giải!\nThời gian: {elapsed:.1f}s\nNode mở rộng: {nodes}")
            self.info_label. config(text="")
            return
        
        self.solutions[algorithm] = (moves, elapsed, nodes)
        self.last_state = self. state
        self.info_label.config(
            text=f"✓ {algorithm}: {len(moves)} bước, {elapsed:.2f}s, {nodes} nodes", fg="green")
        
        SolutionViewer(self. root, self.state, moves, PUZZLE_SIZE, algorithm)

    def view_solution(self):
        if self.state == create_goal(PUZZLE_SIZE):
            messagebox.showinfo("Thông báo", 
                "Puzzle đã hoàn thành!\nHãy xáo trộn để tạo puzzle mới.")
            return
        
        if not is_solvable(self.state, PUZZLE_SIZE):
            messagebox. showerror("Lỗi", "Puzzle không thể giải với goal hiện tại!")
            return
        
        algorithm = self.var_algo.get()
        
        if algorithm in self.solutions and self.last_state == self. state:
            moves, _, _ = self.solutions[algorithm]
            SolutionViewer(self.root, self.state, moves, PUZZLE_SIZE, algorithm)
            return
        
        if self.solving:
            messagebox.showwarning("Cảnh báo", "Đang giải puzzle!")
            return
        
        self.solving = True
        self.solve_btn. config(state='disabled')
        self. progress.pack(pady=5)
        self.progress.start(10)
        Thread(target=lambda: self.solve_thread(algorithm), daemon=True).start()

    def compare_all(self):
        if self.state == create_goal(PUZZLE_SIZE):
            messagebox.showinfo("Thông báo", "Puzzle đã hoàn thành!")
            return
        
        if not is_solvable(self.state, PUZZLE_SIZE):
            messagebox.showerror("Lỗi", "Puzzle không thể giải!")
            return
        
        if self.solving:
            messagebox.showwarning("Cảnh báo", "Đang giải puzzle!")
            return
        
        self.solving = True
        self.progress.pack(pady=5)
        self.progress.start(10)
        self.info_label.config(text="🔍 Đang so sánh 3 thuật toán...", fg="blue")
        Thread(target=self.compare_thread, daemon=True). start()

    def compare_thread(self):
        results = []
        algorithms = [("A*", a_star_solve), ("BFS", bfs_solve), ("DFS", dfs_solve)]
        
        for name, func in algorithms:
            start = time.time()
            moves, nodes = func(self.state, PUZZLE_SIZE, 30 if PUZZLE_SIZE == 3 else 60)
            elapsed = time.time() - start
            results. append((name, moves, elapsed, nodes))
            if moves:
                self.solutions[name] = (moves, elapsed, nodes)
        
        self.last_state = self. state
        self. root.after(0, self.compare_complete, results)

    def compare_complete(self, results):
        self.progress. stop()
        self.progress.pack_forget()
        self.solving = False
        
        compare_win = tk.Toplevel(self.root)
        compare_win.title("So sánh thuật toán")
        compare_win. geometry("700x400")
        
        tk.Label(compare_win, text="Kết quả so sánh", 
                font=("Arial", 16, "bold")). pack(pady=15)
        
        table_frame = tk.Frame(compare_win)
        table_frame.pack(padx=20, pady=10)
        
        headers = ["Thuật toán", "Số bước", "Thời gian", "Node mở rộng", "Kết quả"]
        widths = [12, 10, 12, 13, 10]
        
        for i, (header, width) in enumerate(zip(headers, widths)):
            tk.Label(table_frame, text=header, font=("Arial", 11, "bold"), 
                    width=width, relief="ridge", borderwidth=2, 
                    bg="lightblue").grid(row=0, column=i, padx=1, pady=1)
        
        for i, (algo, moves, elapsed, nodes) in enumerate(results, 1):
            tk.Label(table_frame, text=algo, font=("Arial", 10), 
                    width=widths[0], relief="ridge", borderwidth=1). grid(
                        row=i, column=0, padx=1, pady=1)
            
            if moves is None:
                tk. Label(table_frame, text="Thất bại", font=("Arial", 10), fg="red",
                        width=widths[1], relief="ridge", borderwidth=1).grid(
                            row=i, column=1, padx=1, pady=1)
                tk.Label(table_frame, text=f"{elapsed:.2f}s", font=("Arial", 10),
                        width=widths[2], relief="ridge", borderwidth=1).grid(
                            row=i, column=2, padx=1, pady=1)
                tk. Label(table_frame, text=str(nodes), font=("Arial", 10),
                        width=widths[3], relief="ridge", borderwidth=1).grid(
                            row=i, column=3, padx=1, pady=1)
                tk.Label(table_frame, text="❌", font=("Arial", 10),
                        width=widths[4], relief="ridge", borderwidth=1).grid(
                            row=i, column=4, padx=1, pady=1)
            else:
                tk.Label(table_frame, text=str(len(moves)), font=("Arial", 10), 
                        width=widths[1], relief="ridge", borderwidth=1).grid(
                            row=i, column=1, padx=1, pady=1)
                tk. Label(table_frame, text=f"{elapsed:.3f}s", font=("Arial", 10), 
                        width=widths[2], relief="ridge", borderwidth=1). grid(
                            row=i, column=2, padx=1, pady=1)
                tk.Label(table_frame, text=str(nodes), font=("Arial", 10), 
                        width=widths[3], relief="ridge", borderwidth=1). grid(
                            row=i, column=3, padx=1, pady=1)
                tk.Label(table_frame, text="✓", font=("Arial", 10), fg="green",
                        width=widths[4], relief="ridge", borderwidth=1).grid(
                            row=i, column=4, padx=1, pady=1)
        
        analysis_frame = tk. Frame(compare_win, pady=15)
        analysis_frame.pack()
        
        successful = [(algo, moves, t, n) for algo, moves, t, n in results if moves is not None]
        
        if successful:
            best = min(successful, key=lambda x: len(x[1]))
            fastest = min(successful, key=lambda x: x[2])
            most_efficient = min(successful, key=lambda x: x[3])
            
            tk. Label(analysis_frame, text=f"✓ Tối ưu nhất: {best[0]} ({len(best[1])} bước)", 
                    font=("Arial", 11), fg="green").pack()
            tk.Label(analysis_frame, text=f"⚡ Nhanh nhất: {fastest[0]} ({fastest[2]:.3f}s)", 
                    font=("Arial", 11), fg="blue").pack()
            tk.Label(analysis_frame, text=f"💡 Ít node nhất: {most_efficient[0]} ({most_efficient[3]} nodes)", 
                    font=("Arial", 11), fg="purple").pack()
        else:
            tk. Label(analysis_frame, text="Không có thuật toán nào thành công", 
                    font=("Arial", 11), fg="red"). pack()
            self.info_label. config(text="")
        
        tk.Button(compare_win, text="Đóng", command=compare_win.destroy, 
                 font=("Arial", 10), width=15). pack(pady=10)

if __name__ == "__main__":
    root = tk. Tk()
    app = PuzzleApp(root)
    root.mainloop()