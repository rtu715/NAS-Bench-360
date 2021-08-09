import os

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from .config import *


class welcome_page(tk.Toplevel):
    def __init__(self, master, global_wd, *args, **kwargs):
        super().__init__(master=master)
        self.geometry("600x400+500+100")
        self.title("BioNAS - Welcome")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_propagate(0)
        self.wd = global_wd
        self.frame = tk.Frame(master=self, width=600, height=400, bg=BODY_COLOR)
        self.frame.grid(sticky='nsew')
        # fix the background frame
        self.frame.grid_propagate(0)
        self.create_left_column()
        self.create_right_column()

    def create_left_column(self):
        self.left_column = tk.Frame(master=self.frame, width=150, height=400, bg=MENU_COLOR)
        self.left_column.grid(row=0, column=0, sticky='w')
        # fix left column
        self.left_column.grid_propagate(0)
        # self.left_column.grid_rowconfigure(0, weight=1)
        self.left_column.grid_columnconfigure(0, weight=1)
        button4 = tk.Button(master=self.left_column, text="Quit", command=self._quit, bg=BTN_BG, width=8)
        button4.grid(row=0)
        button2 = tk.Button(master=self.left_column, text="Browse", command=self._ask_folder, bg=BTN_BG, width=8)
        button2.grid(row=1)
        button3 = tk.Button(master=self.left_column, text="Open", command=self._confirm, bg=BTN_BG, width=8)
        button3.grid(row=2)

        sep = ttk.Separator(master=self.left_column, orient=tk.HORIZONTAL)
        sep.grid(row=3, sticky='ew', pady=10)

        lbl1 = tk.Label(master=self.left_column, text="Working Directory:",
                        fg='white',
                        bg=self.left_column['bg'])
        lbl1.grid(row=4, pady=10)
        lbl2 = tk.Label(master=self.left_column, textvariable=self.wd,
                        fg='white',
                        bg=self.left_column['bg'],
                        justify=tk.LEFT
                        )
        lbl2.grid(row=5, sticky='we')

    def _ask_folder(self):
        filename = filedialog.askdirectory()
        self.wd.set(filename)
        print(filename)

    def _confirm(self):
        wd_now = self.wd.get()
        if wd_now == 'None' or not os.path.isdir(wd_now):
            print('bad directory')
            messagebox.showinfo("I/O Error", "Bad working directory: %s" % wd_now)
        else:
            self.master.title("BioNAS - Main - %s" % wd_now)
            self.master.deiconify()
            self.destroy()

    def _quit(self):
        self.master.quit()
        self.master.destroy()

    def create_right_column(self):
        self.right_column = tk.Frame(master=self.frame, width=450, height=400, bg='white')
        self.right_column.grid(row=0, column=1, sticky='nswe')
        # fix right column
        self.right_column.pack_propagate(0)
        logo_label = tk.Label(self.right_column, text=LOGO['ascii'], justify=tk.LEFT, font=LOGO['font'],
                              fg=LOGO['color'],
                              bg=self.right_column['bg'])
        logo_label.grid(row=0, sticky='we', padx=10, pady=10)
        label = tk.Label(self.right_column, text=WELCOME_MSG, font=LARGE_FONT, justify=tk.LEFT,
                         fg='black',
                         bg=self.right_column['bg'])
        label.grid(row=1, sticky='nswe', padx=10, pady=10)
