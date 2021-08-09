# -*- coding: utf-8 -*-

"""
defines the main App interface for AMBER
ZZ
11.3.2019
"""

import os

import tkinter as tk
from PIL import ImageTk, Image
from pkg_resources import resource_filename
from tkinter import ttk

from .config import *
from .evaluate_page import EvaluateTab
from .initialiate_page import InitializeTab
from .train_page import TrainTab
from .welcome import welcome_page


class AmberApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("800x600+500+100")
        self.resizable(0, 0)
        self.style = ttk.Style()
        # self.style.theme_use("clam")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.global_wd = None
        self.title("Amber - Main")
        # self._create_tabs()
        # self._register_animations()

    def _connect_global_wd(self, wd):
        self.global_wd = wd

    def _enter(self):
        assert self.global_wd is not None, "must connect to a global var `wd` before enter"
        self.welcome_window = welcome_page(master=self, global_wd=self.global_wd)
        self.withdraw()

    def _create_tabs(self):
        self.tc = TabController(master=self)
        self.tc.grid(row=0, column=0, sticky='nsew')
        self.animation_register = []
        for tab in self.tc.tabs:
            try:
                self.animation_register.extend(self.tc.tabs[tab].animation_register)
            except Exception as e:
                print("error: %s" % e)
                pass


class TabController(tk.Frame):
    def __init__(self, master, global_wd, global_thread_spawner, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.global_wd = global_wd
        self.global_thread_spawner = global_thread_spawner
        container = tk.Frame(master=self, width=800, height=750, bg=BODY_COLOR)
        container.grid(row=1, sticky='nsew')
        container.grid_propagate(0)
        self.tabs = {}
        page_links = {
            InitializeTab: {'prev': None, 'next': TrainTab},
            TrainTab: {'prev': InitializeTab, 'next': EvaluateTab},
            EvaluateTab: {'prev': TrainTab, 'next': None}
        }
        self.tab_list = [InitializeTab, TrainTab, EvaluateTab]
        for F in self.tab_list:
            frame = F(parent=container,
                      controller=self,
                      global_wd=self.global_wd,
                      global_thread_spawner=self.global_thread_spawner,
                      prev_=page_links[F]['prev'],
                      next_=page_links[F]['next'])
            frame.config({'bg': BODY_COLOR})
            self.tabs[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self._create_menu()
        # must connect TrainTab with InitTab
        self.tabs[TrainTab].init_page = self.tabs[InitializeTab]
        # other registrations
        self._register_animations()
        self.show_frame(InitializeTab)

    def _create_menu(self):
        self.header = tk.Frame(master=self, width=800, height=50, bg=MENU_COLOR)
        self.header.grid(row=0, sticky='nwe')
        self.header.grid_columnconfigure([0, 1, 2], weight=1)
        self.header.grid_propagate(0)
        img_list = [resource_filename('amber.resources', "GUI/" + x) for x in
                    (
                        'init.png',  # setting
                        'run.png',  # run
                        'eval.png',  # eval
                    )]
        canvas_list = []
        image_list = []
        text_ = ['Initialize', 'Train', 'Evaluate']
        for i in range(len(img_list)):
            assert os.path.isfile(img_list[i])
            canvas = tk.Canvas(self.header, width=150, height=50, bd=0, highlightthickness=0, bg=MENU_COLOR)
            # img = ImageTk.PhotoImage(file=img_list[i])
            image = Image.open(img_list[i]).resize((40, 40), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image)
            image_list.append(img)
            canvas.create_image((5, 5), anchor='nw', image=img)
            canvas.create_text((60, 20), anchor='nw', font=LARGE_FONT,
                               text=text_[i])
            canvas_list.append(canvas)
            canvas.grid(row=0, column=i, sticky='we')
        self.canvases = {self.tab_list[i]: canvas_list[i] for i in range(len(self.tab_list))}
        self.image_list = image_list

    def _register_animations(self):
        self.animation_register = []
        for F in self.tabs:
            for f, animate in self.tabs[F].animation_register:
                self.animation_register.append((f, animate))

    def show_frame(self, cont):
        frame = self.tabs[cont]
        frame.tkraise()
        for t in self.canvases:
            canvas = self.canvases[t]
            if canvas == self.canvases[cont]:
                canvas.config(bg=BODY_COLOR)
            else:
                canvas.config(bg=MENU_COLOR)
