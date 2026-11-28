"""PixelBridge GUI (plan report section 4.8).

OWNER: M1 (Leader). Working shell in the skeleton: load an image, choose the
direction and parameters, run, compare methods side by side, save as PNG.
Steps that are not implemented yet show a "Not implemented yet" message
instead of crashing, so the GUI is usable while the team is still coding.

Display only: images are enlarged/shrunk for the screen with integer pixel
repetition / striding (no library resize), and shown through Tk's own PNG
support (core.io.encode_png -> tk.PhotoImage).
"""
from __future__ import annotations

import base64
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ..core.io import encode_png, read_image, write_image
from ..module_a.palette import FIXED_PALETTES
from ..module_a.pipeline import PixelArtParams, photo_to_pixel_art
from ..module_b.pipeline import METHODS, SUPPORTED_SCALES, upscale, upscale_all

VIEW_SIZE = 420


def display_zoom(img: np.ndarray, max_side: int = VIEW_SIZE) -> int:
    """Largest integer zoom that fits the image in a max_side box
    (negative = keep every n-th pixel for images larger than the box)."""
    longer = max(img.shape[:2])
    if longer <= max_side:
        return max(1, max_side // longer)
    return -(-(-longer // max_side))  # -ceil(longer / max_side)


def zoom_for_display(img: np.ndarray, zoom: int) -> np.ndarray:
    """Integer pixel repetition (zoom > 0) or striding (zoom < 0). Display only."""
    if zoom >= 1:
        return np.repeat(np.repeat(img, zoom, axis=0), zoom, axis=1)
    return img[::-zoom, ::-zoom]


def to_photoimage(img: np.ndarray, max_side: int = VIEW_SIZE, zoom: int | None = None) -> tk.PhotoImage:
    z = display_zoom(img, max_side) if zoom is None else zoom
    data = base64.b64encode(encode_png(zoom_for_display(img, z))).decode("ascii")
    return tk.PhotoImage(data=data, format="png")


class PixelBridgeApp(tk.Tk):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.title("PixelBridge - Team13 Tseerei")
        self.minsize(1100, 560)
        self.input_img: np.ndarray | None = None
        self.output_img: np.ndarray | None = None
        self.input_path = ""

        # direction
        self.direction = tk.StringVar(value="A")
        # Module A parameters
        self.a_width = tk.IntVar(value=64)
        self.a_colors = tk.IntVar(value=16)
        self.a_palette = tk.StringVar(value="auto")
        self.a_palette_method = tk.StringVar(value="kmeans")
        self.a_downsample = tk.StringVar(value="edge")
        self.a_smooth = tk.BooleanVar(value=True)
        self.a_enhance = tk.DoubleVar(value=0.0)
        self.a_dither = tk.StringVar(value="none")
        self.a_strength = tk.DoubleVar(value=1.0)
        self.a_outline = tk.BooleanVar(value=True)
        self.a_cleanup = tk.BooleanVar(value=True)
        # Module B parameters
        self.b_scale = tk.IntVar(value=4)
        self.b_method = tk.StringVar(value="ours")
        self.b_mode = tk.StringVar(value="palette")
        # Character parameters ("auto" = measured from the photo)
        self.c_hair = tk.StringVar(value="auto")
        self.c_bangs = tk.StringVar(value="auto")
        self.c_smile = tk.StringVar(value="auto")
        self.c_eyes = tk.StringVar(value="auto")
        self.c_x = tk.DoubleVar(value=0.25)
        self.c_shared = tk.BooleanVar(value=False)
        self.c_bg_path = ""
        self._attrs_cache = None      # (photo, measured attributes, face found): dropdown changes re-render instantly
        self._bg_cache = None         # (path, background, palette)

        self.status = tk.StringVar(value="Open an image to start.")
        self._images = {}  # keep PhotoImage references alive
        self._build()
        if initial_path:
            self.open_image(initial_path)

    # ------------------------------------------------------------ layout
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text="Open image...", command=self.open_image).pack(fill=tk.X)
        ttk.Label(left, text="Direction").pack(anchor="w", pady=(10, 0))
        ttk.Radiobutton(left, text="A: Photo -> Pixel art", value="A", variable=self.direction,
                        command=self._switch).pack(anchor="w")
        ttk.Radiobutton(left, text="B: Pixel art -> High resolution", value="B",
                        variable=self.direction, command=self._switch).pack(anchor="w")
        ttk.Radiobutton(left, text="C: Character on a landmark", value="C",
                        variable=self.direction, command=self._switch).pack(anchor="w")

        self.frame_a = ttk.LabelFrame(left, text="Module A", padding=6)
        self._spin(self.frame_a, "Output width", self.a_width, 16, 256, 8)
        self._spin(self.frame_a, "Colors", self.a_colors, 2, 64, 1)
        self._combo(self.frame_a, "Palette", self.a_palette, ["auto"] + list(FIXED_PALETTES))
        self._combo(self.frame_a, "Palette method", self.a_palette_method, ["median_cut", "kmeans"])
        self._combo(self.frame_a, "Downsample", self.a_downsample, ["block", "edge"])
        self._combo(self.frame_a, "Dither", self.a_dither, ["none", "fs", "bayer"])
        self._scale(self.frame_a, "Dither strength", self.a_strength)
        self._scale(self.frame_a, "Enhance", self.a_enhance)
        ttk.Checkbutton(self.frame_a, text="Smooth (bilateral)", variable=self.a_smooth).pack(anchor="w")
        ttk.Checkbutton(self.frame_a, text="Outline", variable=self.a_outline).pack(anchor="w")
        ttk.Checkbutton(self.frame_a, text="Cleanup", variable=self.a_cleanup).pack(anchor="w")

        self.frame_b = ttk.LabelFrame(left, text="Module B", padding=6)
        self._combo(self.frame_b, "Scale", self.b_scale, [2, 3, 4])
        self._combo(self.frame_b, "Method", self.b_method, list(METHODS))
        self._combo(self.frame_b, "Mode", self.b_mode, ["palette", "smooth"])

        self.frame_c = ttk.LabelFrame(left, text="Character", padding=6)
        self._combo(self.frame_c, "Hair length", self.c_hair, ["auto", "short", "medium", "long"])
        self._combo(self.frame_c, "Bangs", self.c_bangs, ["auto", "none", "curtain", "full", "side"])
        self._combo(self.frame_c, "Smile", self.c_smile, ["auto", "yes", "no"])
        self._combo(self.frame_c, "Smiling eyes", self.c_eyes, ["auto", "yes", "no"])
        self._scale(self.frame_c, "Position", self.c_x)
        ttk.Button(self.frame_c, text="Background photo...", command=self.open_background).pack(fill=tk.X, pady=2)
        self.c_bg_label = ttk.Label(self.frame_c, text="Background: Namsan (built-in)")
        self.c_bg_label.pack(anchor="w")
        ttk.Checkbutton(self.frame_c, text="Shared palette", variable=self.c_shared).pack(anchor="w")

        self.frame_buttons = ttk.Frame(left)
        ttk.Button(self.frame_buttons, text="Run", command=self.run).pack(fill=tk.X, pady=2)
        self.btn_compare = ttk.Button(self.frame_buttons, text="Compare all methods",
                                      command=self.compare_all)
        self.btn_compare.pack(fill=tk.X, pady=2)
        ttk.Button(self.frame_buttons, text="Use output as input",
                   command=self.output_to_input).pack(fill=tk.X, pady=2)
        ttk.Button(self.frame_buttons, text="Save output...", command=self.save).pack(fill=tk.X, pady=2)

        views = ttk.Frame(right)
        views.pack(fill=tk.BOTH, expand=True)
        self.view_in = self._view(views, "Input")
        self.view_out = self._view(views, "Output")
        ttk.Label(right, textvariable=self.status, anchor="w", relief=tk.SUNKEN).pack(fill=tk.X)
        self._switch()

    def _view(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=4)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        label = ttk.Label(frame, anchor="center")
        label.pack(fill=tk.BOTH, expand=True)
        info = ttk.Label(frame, anchor="center")
        info.pack(fill=tk.X)
        return label, info

    def _spin(self, parent, text, var, lo, hi, step):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=text, width=15).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=lo, to=hi, increment=step, textvariable=var, width=8).pack(side=tk.LEFT)

    def _combo(self, parent, text, var, values):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=text, width=15).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=12).pack(side=tk.LEFT)

    def _scale(self, parent, text, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=text, width=15).pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=var, length=110).pack(side=tk.LEFT)

    def _switch(self):
        for f in (self.frame_a, self.frame_b, self.frame_c, self.frame_buttons):
            f.pack_forget()
        frame = {"A": self.frame_a, "B": self.frame_b, "C": self.frame_c}[self.direction.get()]
        frame.pack(fill=tk.X, pady=8)
        self.frame_buttons.pack(fill=tk.X, pady=8)
        self.btn_compare.configure(state=tk.NORMAL if self.direction.get() == "B" else tk.DISABLED)

    # ------------------------------------------------------------ actions
    def _show(self, view, img, text, zoom: int | None = None):
        label, info = view
        photo = to_photoimage(img, zoom=zoom)
        self._images[id(label)] = photo
        label.configure(image=photo)
        info.configure(text=text)

    def open_image(self, path: str | None = None):
        path = path or filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.input_img = read_image(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Open image", str(e))
            return
        self.input_path = path
        h, w = self.input_img.shape[:2]
        # small images are probably pixel art -> switch to Module B (character mode stays)
        if self.direction.get() != "C":
            self.direction.set("B" if max(h, w) <= 256 else "A")
        self._switch()
        self._show(self.view_in, self.input_img, f"{os.path.basename(path)}  {w}x{h}")
        self.status.set("Loaded. Choose parameters and press Run.")

    def params_a(self) -> PixelArtParams:
        return PixelArtParams(
            out_width=int(self.a_width.get()), n_colors=int(self.a_colors.get()),
            palette=self.a_palette.get(), palette_method=self.a_palette_method.get(),
            downsample=self.a_downsample.get(), smooth=bool(self.a_smooth.get()),
            enhance=float(self.a_enhance.get()), dither=self.a_dither.get(),
            dither_strength=float(self.a_strength.get()),
            outline=bool(self.a_outline.get()), cleanup=bool(self.a_cleanup.get()))

    def run(self):
        if self.input_img is None:
            messagebox.showinfo("Run", "Open an image first.")
            return
        self.status.set("Running...")
        self.update_idletasks()
        try:
            if self.direction.get() == "A":
                result = photo_to_pixel_art(self.input_img, self.params_a())
                self.output_img = result.image
                total = sum(result.timings.values())
                detail = ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
                self.status.set(f"Module A done in {total:.2f}s  ({detail})")
            elif self.direction.get() == "C":
                self.output_img = self.run_character()
            else:
                import time
                t0 = time.perf_counter()
                self.output_img = upscale(self.input_img, int(self.b_scale.get()),
                                          self.b_method.get(), self.b_mode.get())
                self.status.set(f"Module B ({self.b_method.get()}) done in {time.perf_counter() - t0:.2f}s")
        except NotImplementedError as e:
            self.status.set(f"Not implemented yet: {e}")
            return
        except Exception as e:  # noqa: BLE001
            self.status.set(f"Error: {e}")
            messagebox.showerror("Run", str(e))
            return
        h, w = self.output_img.shape[:2]
        if self.direction.get() == "B":
            # show input and output at the same size on screen for a fair comparison
            z = display_zoom(self.output_img)
            if z >= 1:
                ih, iw = self.input_img.shape[:2]
                self._show(self.view_in, self.input_img, f"input {iw}x{ih} (nearest, for display)",
                           zoom=z * int(self.b_scale.get()))
            self._show(self.view_out, self.output_img, f"{w}x{h}", zoom=z)
        else:
            self._show(self.view_out, self.output_img, f"{w}x{h}")

    def open_background(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path:
            self.c_bg_path = path
            self.c_bg_label.configure(text=f"Background: {os.path.basename(path)}")

    def run_character(self) -> np.ndarray:
        """Photo -> character -> landmark background -> x4 (Scale2x twice)."""
        from dataclasses import replace

        from ..character import (compose_scene, default_background, measure_attributes,
                                 pixel_background, render_character)
        from ..character.detect import find_head_box

        if self._attrs_cache is None or self._attrs_cache[0] is not self.input_img:
            self.status.set("Measuring the face (first run takes a few seconds)...")
            self.update_idletasks()
            box, found = find_head_box(self.input_img, return_found=True)
            attrs = measure_attributes(self.input_img, box)
            self._attrs_cache = (self.input_img, attrs, found)
        attrs, found = self._attrs_cache[1], self._attrs_cache[2]
        flag = {"yes": True, "no": False}
        changes = {k: v for k, v in (("length", self.c_hair.get()), ("bangs", self.c_bangs.get())) if v != "auto"}
        for key, var in (("smile", self.c_smile), ("happy_eyes", self.c_eyes)):
            if var.get() != "auto":
                changes[key] = flag[var.get()]
        character = render_character(replace(attrs, **changes))

        if self.c_bg_path:
            if self._bg_cache is None or self._bg_cache[0] != self.c_bg_path:
                bg, pal = pixel_background(read_image(self.c_bg_path))
                self._bg_cache = (self.c_bg_path, bg, pal)
            _, background, palette = self._bg_cache
        else:
            background, palette = default_background(), None
        scene = compose_scene(background, character, x=float(self.c_x.get()),
                              palette=palette if self.c_shared.get() else None)
        a = character.attrs
        warn = "" if found else "No clear face found - use a front-facing photo of one person.  "
        self.status.set(f"{warn}Character: hair {a.length}, bangs {a.bangs}, smile {'yes' if a.smile else 'no'}, "
                        f"smiling eyes {'yes' if a.happy_eyes else 'no'}  (auto = measured from the photo)")
        return upscale(scene, 4, "scale2x")

    def compare_all(self):
        if self.input_img is None:
            return
        scale = int(self.b_scale.get())
        self.status.set("Running all methods...")
        self.update_idletasks()
        results = upscale_all(self.input_img, scale, mode=self.b_mode.get())
        win = tk.Toplevel(self)
        win.title(f"Compare methods x{scale} ({self.b_mode.get()} mode)")
        cols = 4
        for i, (m, r) in enumerate(results.items()):
            cell = ttk.Frame(win, padding=4)
            cell.grid(row=i // cols, column=i % cols)
            if r["image"] is not None:
                photo = to_photoimage(r["image"], 260)
                self._images[f"cmp_{m}"] = photo
                ttk.Label(cell, image=photo).pack()
                ttk.Label(cell, text=f"{m}  {r['seconds']:.2f}s").pack()
            else:
                ttk.Label(cell, text=f"{m}\n(not implemented yet)", width=30).pack(pady=100)
        skipped = [m for m in METHODS if scale not in SUPPORTED_SCALES[m]]
        self.status.set(f"Compared {len(results)} methods" +
                        (f"; skipped (scale not supported): {', '.join(skipped)}" if skipped else ""))

    def output_to_input(self):
        """Round trip: use the pixel art from Module A as the input of Module B."""
        if self.output_img is None:
            return
        self.input_img = self.output_img
        self.output_img = None
        h, w = self.input_img.shape[:2]
        self.direction.set("B" if self.direction.get() == "A" else "A")
        self._switch()
        self._show(self.view_in, self.input_img, f"(previous output) {w}x{h}")
        self.view_out[0].configure(image="")
        self.view_out[1].configure(text="")

    def save(self):
        if self.output_img is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            write_image(path, self.output_img)
            self.status.set(f"Saved {path}")


def main(initial_path: str | None = None):
    PixelBridgeApp(initial_path).mainloop()
