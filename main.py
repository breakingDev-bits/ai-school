import ctypes
import os
import sys
import tkinter as tk
from tkinter import ttk

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = {"win32": "nn.dll", "darwin": "libnn.dylib"}.get(sys.platform, "libnn.so")
lib = ctypes.CDLL(os.path.join(HERE, LIB_NAME))


class StepData(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in ("prediction", "error", "w0", "w1", "bias")]


lib.nn_train.argtypes = [ctypes.c_double] * 5 + [ctypes.c_int, ctypes.POINTER(StepData)]
lib.nn_train.restype = ctypes.c_int
lib.nn_predict.argtypes = [ctypes.c_double] * 5
lib.nn_predict.restype = ctypes.c_double


def train(x, y, target, lr, eps, max_steps):
    buf = (StepData * max_steps)()
    n = lib.nn_train(x, y, target, lr, eps, max_steps, buf)
    return list(buf[:n])


POS = "#2a78d6"
NEG = "#d64a2a"
BG = "#ffffff"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Нейрон: навчання")
        self.configure(bg=BG)
        self.history = []
        self.idx = 0
        self.running = False
        self.target = 7.0
        self._build_ui()
        self.reset()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.x_var = tk.StringVar(value="3")
        self.y_var = tk.StringVar(value="4")
        self.t_var = tk.StringVar(value="7")
        self.lr_var = tk.StringVar(value="0.01")
        self.steps_var = tk.StringVar(value="100")
        self.eps_var = tk.StringVar(value="0.001")

        for label, var in (("x", self.x_var), ("y", self.y_var), ("відповідь", self.t_var),
                           ("lr", self.lr_var), ("макс. епох", self.steps_var),
                           ("поріг помилки", self.eps_var)):
            ttk.Label(top, text=label).pack(side="left", padx=(8, 2))
            ttk.Entry(top, textvariable=var, width=6).pack(side="left")

        ctrl = ttk.Frame(self, padding=(8, 0))
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Нова модель", command=self.reset).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Крок", command=self.step).pack(side="left", padx=2)
        self.run_btn = ttk.Button(ctrl, text="Старт", command=self.toggle_run)
        self.run_btn.pack(side="left", padx=2)
        ttk.Button(ctrl, text="В кінець", command=self.to_end).pack(side="left", padx=2)
        ttk.Label(ctrl, text="  затримка, мс").pack(side="left")
        self.speed = tk.Scale(ctrl, from_=10, to=800, orient="horizontal", length=140, bg=BG,
                              highlightthickness=0)
        self.speed.set(150)
        self.speed.pack(side="left")

        body = ttk.Frame(self, padding=8)
        body.pack()
        self.net = tk.Canvas(body, width=380, height=320, bg=BG, highlightthickness=1,
                             highlightbackground="#cccccc")
        self.net.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self.pred_plot = tk.Canvas(body, width=420, height=155, bg=BG, highlightthickness=1,
                                   highlightbackground="#cccccc")
        self.pred_plot.grid(row=0, column=1, pady=(0, 6))
        self.err_plot = tk.Canvas(body, width=420, height=155, bg=BG, highlightthickness=1,
                                  highlightbackground="#cccccc")
        self.err_plot.grid(row=1, column=1)

        test = ttk.Frame(self, padding=8)
        test.pack(fill="x")
        ttk.Label(test, text="Перевірка поточної моделі:  x").pack(side="left")
        self.tx = tk.StringVar(value="1")
        self.ty = tk.StringVar(value="2")
        ttk.Entry(test, textvariable=self.tx, width=6).pack(side="left", padx=2)
        ttk.Label(test, text="y").pack(side="left")
        ttk.Entry(test, textvariable=self.ty, width=6).pack(side="left", padx=2)
        ttk.Button(test, text="Передбачити", command=self.test_predict).pack(side="left", padx=6)
        self.test_lbl = ttk.Label(test, text="")
        self.test_lbl.pack(side="left")

        self.info = ttk.Label(self, padding=(8, 0, 8, 8))
        self.info.pack(fill="x")

    # ---------- логіка ----------
    def reset(self):
        self.running = False
        self.run_btn.config(text="Старт")
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            self.target = float(self.t_var.get())
            lr = float(self.lr_var.get())
            steps = max(1, int(self.steps_var.get()))
            eps = max(0.0, float(self.eps_var.get()))
        except ValueError:
            self.info.config(text="Перевір числа в полях")
            return
        self.x, self.y = x, y
        self.history = train(x, y, self.target, lr, eps, steps)
        self.idx = 0
        self.test_lbl.config(text="")
        self.redraw()

    def step(self):
        if self.idx < len(self.history) - 1:
            self.idx += 1
            self.redraw()
            return True
        return False

    def to_end(self):
        self.running = False
        self.run_btn.config(text="Старт")
        self.idx = len(self.history) - 1
        self.redraw()

    def toggle_run(self):
        self.running = not self.running
        self.run_btn.config(text="Пауза" if self.running else "Старт")
        if self.running:
            self._tick()

    def _tick(self):
        if not self.running:
            return
        if self.step():
            self.after(int(self.speed.get()), self._tick)
        else:
            self.running = False
            self.run_btn.config(text="Старт")

    def test_predict(self):
        if not self.history:
            return
        try:
            x = float(self.tx.get())
            y = float(self.ty.get())
        except ValueError:
            return
        s = self.history[self.idx]
        res = lib.nn_predict(x, y, s.w0, s.w1, s.bias)
        self.test_lbl.config(text=f"{x:g} + {y:g}  →  модель думає: {res:.3f}   (насправді {x + y:g})")

    # ---------- малювання ----------
    def redraw(self):
        if not self.history:
            return
        s = self.history[self.idx]
        self.draw_net(s)
        preds = [h.prediction for h in self.history]
        errs = [h.error for h in self.history]
        self.draw_plot(self.pred_plot, preds, self.target, "Передбачення (пунктир = правильна відповідь)", POS)
        self.draw_plot(self.err_plot, errs, 0.0, "Помилка", NEG)
        self.info.config(text=f"Епоха {self.idx}/{len(self.history) - 1}   "
                              f"передбачення {s.prediction:.4f}   помилка {s.error:.4f}")

    def draw_net(self, s):
        c = self.net
        c.delete("all")
        inputs = [("x", 60, 90, self.x, s.w0), ("y", 60, 190, self.y, s.w1)]
        out_x, out_y = 300, 140

        for name, px, py, val, w in inputs:
            color = POS if w >= 0 else NEG
            c.create_line(px + 28, py, out_x - 32, out_y, width=1 + min(abs(w), 4) * 3, fill=color)
            mx, my = (px + out_x) / 2, (py + out_y) / 2 + (-12 if py < out_y else 12)
            c.create_text(mx, my, text=f"w = {w:.3f}", fill=color, font=("Arial", 11, "bold"))

        # bias
        bx, by = 60, 270
        color = POS if s.bias >= 0 else NEG
        c.create_line(bx + 22, by, out_x - 30, out_y + 12, width=1 + min(abs(s.bias), 4) * 3,
                      fill=color, dash=(4, 3))
        c.create_text((bx + out_x) / 2, (by + out_y) / 2 + 22, text=f"bias = {s.bias:.3f}",
                      fill=color, font=("Arial", 11, "bold"))
        c.create_oval(bx - 22, by - 22, bx + 22, by + 22, fill="#eeeeee", outline="#555555")
        c.create_text(bx, by, text="1")

        for name, px, py, val, w in inputs:
            c.create_oval(px - 28, py - 28, px + 28, py + 28, fill="#eef4ff", outline="#555555", width=2)
            c.create_text(px, py - 8, text=name)
            c.create_text(px, py + 8, text=f"{val:g}", font=("Arial", 12, "bold"))

        c.create_oval(out_x - 34, out_y - 34, out_x + 34, out_y + 34, fill="#fff4e6",
                      outline="#555555", width=2)
        c.create_text(out_x, out_y - 8, text="вихід")
        c.create_text(out_x, out_y + 8, text=f"{s.prediction:.2f}", font=("Arial", 12, "bold"))
        c.create_text(out_x, out_y + 50, text=f"потрібно: {self.target:g}", fill="#555555")
        c.create_text(190, 15, text="передбачення = x·w0 + y·w1 + bias", fill="#555555")

    def draw_plot(self, c, vals, ref, title, color):
        c.delete("all")
        w, h = int(c["width"]), int(c["height"])
        l, r, t, b = 48, 12, 24, 22
        lo = min(min(vals), ref)
        hi = max(max(vals), ref)
        if hi - lo < 1e-9:
            hi, lo = hi + 1, lo - 1
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        n = max(len(vals) - 1, 1)

        def X(i):
            return l + (w - l - r) * i / n

        def Y(v):
            return t + (h - t - b) * (1 - (v - lo) / (hi - lo))

        c.create_text(w / 2, 11, text=title, fill="#333333")
        c.create_rectangle(l, t, w - r, h - b, outline="#aaaaaa")
        for v in (lo + pad, ref, hi - pad):
            c.create_text(l - 4, Y(v), text=f"{v:.2f}", anchor="e", font=("Arial", 8))
        c.create_text(l, h - 8, text="0", font=("Arial", 8))
        c.create_text(w - r, h - 8, text=f"{len(vals) - 1} епох", anchor="e", font=("Arial", 8))
        c.create_line(l, Y(ref), w - r, Y(ref), dash=(5, 4), fill="#555555")
        pts = []
        for i in range(self.idx + 1):
            pts += [X(i), Y(vals[i])]
        if len(pts) >= 4:
            c.create_line(*pts, fill=color, width=2)
        c.create_oval(X(self.idx) - 4, Y(vals[self.idx]) - 4, X(self.idx) + 4, Y(vals[self.idx]) + 4,
                      fill=color, outline="")


if __name__ == "__main__":
    App().mainloop()    