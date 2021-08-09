import os

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from .config import *


class LabelSeparator(tk.Frame):
    def __init__(self, parent, text="", width="", *args):
        tk.Frame.__init__(self, parent, bg=BODY_COLOR, *args)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        self.separator.grid(row=0, column=0, sticky="ew")

        self.label = tk.Label(self, text=text, bg=BODY_COLOR)
        self.label.grid(row=0, column=0)


def create_widget(arg, master):
    if type(arg) is list and len(arg) > 0:
        str_var = tk.StringVar(master)
        str_var.set(arg[0])
        btn_var = tk.StringVar(master)
        btn_var.set(arg[0])

        def set_fp(x):
            if btn_var.get() == 'Custom..':
                fp = filedialog.askopenfilename()
                str_var.set(fp)
                btn_var.set(os.path.basename(fp)[:15])
            else:
                str_var.set(btn_var.get())

        pop = tk.OptionMenu(master, btn_var, *arg,
                            command=set_fp)
        pop.config(width=10, bg=BODY_COLOR, justify=tk.LEFT)
        return str_var, pop, btn_var

    elif type(arg) is str and arg.startswith('[') and arg.endswith(']'):
        w, h = arg.strip('[]').split(',')

        str_var = tk.Text(master, width=int(w), height=int(h), bg=TEXT_BG)
        return str_var, str_var, str_var

    elif type(arg) is str and arg.startswith('{') and arg.endswith('}'):
        w, h = arg.strip('{}').split(',')
        assert int(h) == 1, "tk.Entry widget only has height=1"
        str_var = tk.StringVar(master, value='')
        widget = tk.Entry(master, width=int(w), textvariable=str_var, state='readonly', bg=TEXT_BG)
        return str_var, widget, str_var

    elif type(arg) is str and arg.startswith('--'):
        # sep = ttk.Separator(master=master, orient=tk.HORIZONTAL)
        sep = LabelSeparator(parent=master, text=arg.strip('-'))
        return None, sep, None

    elif type(arg) is str and arg == "Custom..":
        str_var = tk.StringVar(master, value='None')
        btn_txt_var = tk.StringVar(master, value='Choose..')

        def set_text():
            fp = filedialog.askopenfilename()
            str_var.set(fp)
            btn_txt_var.set(os.path.basename(fp)[:15])

        btn = tk.Button(master, textvariable=btn_txt_var, command=set_text, bg=BTN_BG)
        return str_var, btn, btn_txt_var

    else:
        raise Exception("Failed to create widget: %s" % arg)
