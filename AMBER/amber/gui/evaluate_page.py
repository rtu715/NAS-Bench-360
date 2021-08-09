import os

import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import messagebox
from tkinter import ttk

from .config import *

images_ = []
captions_ = []
images_type_indices = {g: [] for g in GALLERY_SHOW_TYPES}
current = 0


def load_images(wd, frame):
    global images_, captions_
    pngs = sorted([x for x in os.listdir(wd) if x.endswith('png')])
    images_ = [os.path.join(wd, x) for x in pngs]
    captions_ = [x.split('.')[0] for x in pngs]
    if len(pngs) == 0:
        messagebox.showinfo("Warning", "Load failed. No figures found.")
        return
    for g in GALLERY_SHOW_TYPES:
        images_type_indices[g] = []
        for i in range(len(captions_)):
            if captions_[i].startswith(GALLERY_SHOWTYPE_STARTSWTIH[g]):
                images_type_indices[g].append(i)
    move(0, frame)


def move(delta, frame):
    global current, images_, images_type_indices
    gallery_type = frame.gallery_showtype.get()
    if not (0 <= current + delta < len(images_type_indices[gallery_type])):
        if len(images_type_indices[gallery_type]) == 0:
            frame.gallery_showtype.set(GALLERY_SHOW_TYPES[0])
        else:
            current = -1 if current + delta >= len(images_type_indices[gallery_type]) else len(
                images_type_indices[gallery_type])

    current += delta
    image = Image.open(images_[images_type_indices[gallery_type][current]]).resize(GALLERY_SIZE, Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    page_count = "Figure %i / %i :" % (current + 1, len(images_type_indices[gallery_type]))
    frame.demo_label['text'] = page_count + ' ' + captions_[images_type_indices[gallery_type][current]]
    frame.demo_label['image'] = photo
    frame.demo_label.photo = photo


class EvaluateTab(tk.Frame):

    def __init__(self, parent, controller, global_wd, global_thread_spawner=None, prev_=None, next_=None):
        tk.Frame.__init__(self, master=parent, bg=BODY_COLOR)
        self.global_wd = global_wd
        self.global_thread_spawner = global_thread_spawner
        self._create_header(controller, prev_, next_)
        self._create_left_column()
        self._create_gallery_column()
        self.animation_register = []
        self.has_run = False

    def _create_header(self, controller, prev_, next_):
        self.header = tk.Frame(master=self, width=800, height=20, bg=BODY_COLOR)
        self.header.grid(row=0, columnspan=5, sticky='nw')
        # label = tk.Label(self.header, text="Evaluate BioNAS", font=LARGE_FONT, bg=BODY_COLOR)
        # label.grid(row=0, sticky='nw')

        button2 = tk.Button(self.header, text="Discover ->",
                            command=lambda: controller.show_frame(next_), bg=BTN_BG)
        button2.grid(row=0, column=2, sticky='w')

        button1 = tk.Button(self.header, text="<- Train",
                            command=lambda: controller.show_frame(prev_), bg=BTN_BG)
        button1.grid(row=0, column=1, sticky='w')

    def _create_left_column(self):
        self.left_column = tk.Frame(master=self, width=300, height=600, bg=SIDEBAR_COLOR)
        self.left_column.grid(row=1, column=0, columnspan=2, sticky='nw')
        self.left_column.rowconfigure(0, weight=1)
        label = tk.Label(self.left_column, text="Gallery", font=LARGE_FONT, bg=BODY_COLOR,
                         justify=tk.CENTER)
        label.grid(row=0, columnspan=2, sticky='ew')

        button1 = tk.Button(self.left_column, text="Load",
                            command=lambda: load_images(os.path.join(self.global_wd.get()), self),
                            bg=BTN_BG)
        button1.grid(row=6, column=0, columnspan=2, sticky='we')

        button2 = tk.Button(self.left_column, text="<--", command=lambda: move(-1, self), bg=BTN_BG)
        button2.grid(row=7, column=0, columnspan=1, sticky='we')

        button3 = tk.Button(self.left_column, text="-->", command=lambda: move(+1, self), bg=BTN_BG)
        button3.grid(row=7, column=1, columnspan=1, sticky='we')

        sep = ttk.Separator(master=self.left_column, orient=tk.HORIZONTAL)
        sep.grid(row=8, columnspan=2, sticky='ew', pady=10)

        self.gallery_showtype = tk.StringVar(self.left_column)
        self.gallery_showtype.set(GALLERY_SHOW_TYPES[0])

        popupMenu = tk.OptionMenu(self.left_column, self.gallery_showtype, *GALLERY_SHOW_TYPES,
                                  command=self._popup_menu_move)
        popupMenu.config(bg=BODY_COLOR)
        tk.Label(self.left_column, text="Gallery show type:", bg=BODY_COLOR).grid(row=9, columnspan=2, sticky='w')
        popupMenu.grid(row=10, columnspan=2)

    def _popup_menu_move(self, *args):
        global current
        current = 0
        move(0, self)

    def _create_gallery_column(self):
        self.right_column = tk.Frame(master=self, width=500, height=600, bg=BODY_COLOR)
        self.right_column.grid(row=1, column=2, columnspan=3, sticky="nswe")
        label = tk.Label(self.right_column, compound=tk.TOP, font=LARGE_FONT, bg=BODY_COLOR)
        label.grid(row=0, sticky='nsew')
        self.demo_label = label
