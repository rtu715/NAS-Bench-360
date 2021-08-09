import json
import os

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from .config import *
from .widgets import create_widget
from ..getter import gui_mapper

"""workflow
1. config target model: model_space, model_builder, model_data
2. config interpretability: knowledge_fn, knowledge_data, reward_fn
3. config controller: controller, manager
4. config training environment: env
"""


def parse_layout(master):
    frames = []
    var_dict = {}
    for tab_name in PARAMS_LAYOUT:
        f = tk.Frame(master=master, bg=BODY_COLOR)
        # f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure([0, 1, 2, 3], minsize=100)
        for k, v in PARAMS_LAYOUT[tab_name].items():
            try:
                str_var, widget, btn_var = create_widget(v['value'], f)
                if type(widget) is tk.Text and 'default' in v:
                    widget.insert("end", v['default'])
                widget.grid(**v['wpos'])
            except Exception as e:
                raise Exception("%s\n caused by (%s, %s)" % (e, k, v))
            if str_var is not None:
                var_dict[k] = (str_var, btn_var)
                label = tk.Label(f, text=k + ': ', bg=BODY_COLOR)
                label.grid(**v['lpos'])
        frames.append(f)
    return frames, var_dict


def pretty_print_dict(d):
    print('-' * 80)
    for k, v in d.items():
        print(k, " = ", v)


class InitializeTab(tk.Frame):

    def __init__(self, parent, controller, global_wd, global_thread_spawner=None, prev_=None, next_=None):
        tk.Frame.__init__(self, parent, bg=BODY_COLOR)
        self.global_wd = global_wd
        self.var_dict = {'wd': (global_wd, 0)}
        self.animation_register = []
        self.param_tab_cont = 0
        self._create_header(controller, prev_, next_)
        self._create_left_column()
        self._create_right_column()

    def _create_header(self, controller, prev_, next_):
        self.header = tk.Frame(master=self, width=800, height=20, bg=BODY_COLOR)
        self.header.grid(row=0, columnspan=5, sticky='nw')
        # label = tk.Label(self.header, text="Initialize BioNAS", font=LARGE_FONT, bg=BODY_COLOR)
        # label.grid(row=0, sticky='nw')

        button2 = tk.Button(self.header, text="Train ->",
                            command=lambda: controller.show_frame(next_),
                            bg=BTN_BG)
        button2.grid(row=0, column=2, sticky='w')

        button1 = tk.Button(self.header, text="<- Home",
                            command=lambda: controller.show_frame(prev_),
                            bg=BTN_BG)
        button1.grid(row=0, column=1, sticky='w')

    def _create_left_column(self):
        self.left_column = tk.Frame(master=self, width=300, height=600, bg=SIDEBAR_COLOR)
        self.left_column.grid(row=1, column=0, columnspan=2, sticky='nw')
        self.left_column.rowconfigure(0, weight=1)
        label = tk.Label(self.left_column, text="Configuration", font=LARGE_FONT, bg=BODY_COLOR,
                         justify=tk.CENTER)
        label.grid(row=0, columnspan=2, sticky='ew')

        button0 = tk.Button(self.left_column, text="Load",
                            command=self.load, bg=BTN_BG)
        button0.grid(row=2, column=0, columnspan=1, sticky='we')

        button1 = tk.Button(self.left_column, text="Save", command=self.save, bg=BTN_BG)
        button1.grid(row=2, column=1, columnspan=1, sticky='we')

        button2 = tk.Button(self.left_column, text="<--", command=lambda: self.move(-1), bg=BTN_BG)
        button2.grid(row=3, column=0, columnspan=1, sticky='we')

        button3 = tk.Button(self.left_column, text="-->", command=lambda: self.move(+1), bg=BTN_BG)
        button3.grid(row=3, column=1, columnspan=1, sticky='we')

        # btn = tk.Button(self.left_column, text="Debug", command=self.preset)
        # btn.grid(row=4)

        sep = ttk.Separator(master=self.left_column, orient=tk.HORIZONTAL)
        sep.grid(row=6, columnspan=2, sticky='ew', pady=10)

        self.params_showtype = tk.StringVar(self.left_column)
        self.params_showtype.set(PARAMS_SHOW_TYPES[0])
        showtype_to_index = {PARAMS_SHOW_TYPES[i]: i for i in range(len(PARAMS_SHOW_TYPES))}

        popupMenu = tk.OptionMenu(self.left_column, self.params_showtype, *PARAMS_SHOW_TYPES,
                                  command=lambda x: self.move(
                                      delta=showtype_to_index[self.params_showtype.get()] - self.param_tab_cont
                                  ))
        popupMenu.config(bg=BODY_COLOR)
        tk.Label(self.left_column, text="Parameters:", bg=BODY_COLOR).grid(row=7, columnspan=2, sticky='w')
        popupMenu.grid(row=8, columnspan=2)

    def _create_right_column(self):
        self.right_column = tk.Frame(master=self, width=500, height=600, bg=BODY_COLOR)
        self.right_column.grid(row=1, column=2, columnspan=3, sticky='nw')
        self.frames, var_dict = parse_layout(self.right_column)
        self.var_dict.update(var_dict)
        _ = list(map(lambda x: x.grid(row=0, column=0, sticky='nesw'), self.frames))
        self.frames[0].tkraise()

    def parse_vars(self, verbose=0):
        var_dict = {}
        for k, v in self.var_dict.items():
            if type(v[0]) is tk.StringVar:
                v_ = v[0].get()
            elif type(v[0]) is tk.Text:
                v_ = v[0].get(1.0, "end-1c")
            else:
                raise Exception("Error in parse_vars: %s, %s" % (k, v))
            var_dict[k] = v_
        if verbose:
            pretty_print_dict(var_dict)
        return var_dict

    def move(self, delta):
        self.param_tab_cont += delta
        if self.param_tab_cont >= len(self.frames):
            self.param_tab_cont = 0
        if self.param_tab_cont < 0:
            self.param_tab_cont = len(self.frames) - 1
        self.frames[self.param_tab_cont].tkraise()

    def preset(self):
        types, specs = gui_mapper(self.parse_vars(verbose=0))
        pretty_print_dict(types)
        pretty_print_dict(specs)

    def save(self):
        var_dict = self.parse_vars(verbose=0)
        param_fp = os.path.join(self.global_wd.get(), 'param_config.json')
        if os.path.isfile(param_fp):
            is_overwrite = tk.messagebox.askquestion('File exists',
                                                     'Are you sure you want to overwrite the parameter file?',
                                                     icon='warning')
            if is_overwrite != 'yes':
                return
        with open(param_fp, 'w', newline="\n") as f:
            json.dump(var_dict, f, indent=4)

    def load(self, fp=None):
        if fp is None:
            fp = filedialog.askopenfilename(initialdir=self.global_wd.get())
        if not len(fp):
            fp = os.path.join(self.global_wd.get(), 'param_config.json')
        print(fp)
        if not os.path.isfile(fp):
            return
        with open(fp, 'r') as f:
            var_dict = json.load(f)
        for k, v in var_dict.items():
            # skip setting global_wd
            if k == 'wd':
                continue
            if type(self.var_dict[k][0]) is tk.StringVar:
                self.var_dict[k][0].set(v)
                self.var_dict[k][-1].set(os.path.basename(v)[:15])
            elif type(self.var_dict[k][0]) is tk.Text:
                self.var_dict[k][0].delete(1.0, 'end')
                self.var_dict[k][0].insert('end', v)
        return
