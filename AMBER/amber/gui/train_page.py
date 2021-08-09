import os
import pickle
import signal

# matplotlib.style.use("ggplot")
import gpustat
import matplotlib
import numpy as np
import psutil
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox

from .config import *
from .widgets import create_widget
from ..getter import gui_mapper

matplotlib.use("TkAgg")

f = Figure()
DPI = f.get_dpi()
f.set_size_inches(400.0 / float(DPI), 200.0 / float(DPI))
a = f.add_subplot(111)


def beautify_status_update(tab):
    var_dict = tab.var_dict

    def colorify(p, widget=None):
        if p is None:
            color = 'grey'
        elif p < 30:
            color = 'green'
        elif 30 <= p < 70:
            color = 'orange'
        elif 70 <= p < 100:
            color = 'red'
        else:
            color = 'black'
        if widget is None:
            return color
        else:
            widget.config({'fg': color})

    def get_cpu_usage():
        p = psutil.cpu_percent()
        var_dict['cpu_status'][0].set(p)
        colorify(p, var_dict['cpu_status'][2])

    def get_ram():
        d = psutil.virtual_memory()._asdict()
        usage, p = d['used'], d['percent']
        usage = round(usage / (1024.) ** 3, 1)
        var_dict['ram'][0].set("%iG; %s" % (usage, p))
        colorify(p, var_dict["ram"][2])

    def get_gpu_usage():
        if NUM_GPUS:
            new_q = gpustat.GPUStatCollection.new_query().jsonify()
            var_dict['gpu_status'][0].delete(1.0, tk.END)
            color_list = []
            for gpu_d in new_q['gpus']:
                s = "{index}: {util}; {mem_used}/{mem_total}g \n".format(
                    index=gpu_d['index'],
                    util=gpu_d['utilization.gpu'],
                    mem_used=round(gpu_d['memory.used'] / 1024., 1),
                    mem_total=round(gpu_d['memory.total'] / 1024., 1)
                )
                color_list.append(colorify(gpu_d['utilization.gpu']))
                var_dict['gpu_status'][0].insert(tk.END, s)
            for i in range(len(color_list)):
                color = color_list[i]
                var_dict['gpu_status'][0].tag_add(i, "%i.0" % (i + 1), "%i.end" % (i + 1))
                var_dict['gpu_status'][0].tag_config(i, foreground=color)

    def get_run_status():
        if tab.bn is None:
            var_dict['run_status'][0].set("Waiting")
            p = None
        elif tab.bn.poll() is None:
            var_dict['run_status'][0].set("Running")
            p = 30
        elif tab.bn.poll() == 0:
            var_dict['run_status'][0].set("Finished")
            p = 0
        elif tab.bn.poll() == 1:
            var_dict['run_status'][0].set("Stopped")
            p = 90
        else:
            var_dict['run_status'][0].set("Code:%s" % tab.bn.poll())
            p = -1
        colorify(p, var_dict['run_status'][2])

    func_map = {
        'cpu_status': get_cpu_usage,
        'run_status': get_run_status,
        'ram': get_ram,
        'gpu_status': get_gpu_usage,
    }
    return func_map


class TrainTab(tk.Frame):

    def __init__(self, parent, controller, global_wd, global_thread_spawner=None, prev_=None, next_=None):
        tk.Frame.__init__(self, master=parent, bg=BODY_COLOR)
        self.global_wd = global_wd
        self.global_thread_spawner = global_thread_spawner
        self.init_page = None
        self.bn = None
        self.var_dict = {}
        self._create_header(controller, prev_, next_)
        self._create_left_column()
        self._create_demo_column()
        self.animation_register = [(f, self._animate_r_bias)]
        self.has_run = False
        self.status_bar_fn_map = beautify_status_update(self)
        self._update_status()

    def _create_header(self, controller, prev_, next_):
        self.header = tk.Frame(master=self, width=800, height=20, bg=BODY_COLOR)
        self.header.grid(row=0, columnspan=5, sticky='nw')
        # label = tk.Label(self.header, text="Train BioNAS", font=LARGE_FONT, bg=BODY_COLOR)
        # label.grid(row=0, sticky='nw')

        button2 = tk.Button(self.header, text="Evaluate ->",
                            command=lambda: controller.show_frame(next_),
                            bg=BTN_BG)
        button2.grid(row=0, column=2, sticky='w')

        button1 = tk.Button(self.header, text="<- Initialize",
                            command=lambda: controller.show_frame(prev_),
                            bg=BTN_BG)
        button1.grid(row=0, column=1, sticky='w')

    def _create_left_column(self):
        """
        TODO:
            - integrate with gpustat for monitoring GPU usage
            - update labels with dynamic statistics, e.g. entropy, loss/knowledge, learning rate, etc.
        Returns:
            None
        """
        self.left_column = tk.Frame(master=self, width=300, height=600, bg=SIDEBAR_COLOR)
        self.left_column.grid(row=1, column=0, columnspan=2, sticky='nw')
        self.left_column.grid_rowconfigure(0, weight=1)
        label = tk.Label(self.left_column, text="Train", font=LARGE_FONT, bg=BODY_COLOR,
                         justify=tk.CENTER)
        label.grid(row=0, column=0, columnspan=2, sticky='we')

        button1 = tk.Button(self.left_column, text="Build", command=self._build, bg=BTN_BG)
        button1.grid(row=6, column=0, columnspan=2, sticky='we')

        button2 = tk.Button(self.left_column, text="Run", command=self._run, bg=BTN_BG)
        button2.grid(row=7, column=0, columnspan=2, sticky='we')

        button3 = tk.Button(self.left_column, text="Stop", command=self._stop, bg=BTN_BG)
        button3.grid(row=8, column=0, columnspan=2, sticky='we')

        status_bar = tk.Frame(master=self.left_column, width=300, bg=BODY_COLOR)
        status_bar.grid(row=9, column=0, columnspan=2, sticky='nwe')
        var_dict = {}
        for k, v in STATUS_BAR_LAYOUT.items():
            str_var, widget, btn_var = create_widget(v['value'], status_bar)
            widget.grid(**v['wpos'])
            if str_var is not None:
                var_dict[k] = (str_var, btn_var, widget)
                label = tk.Label(status_bar, text=k + ': ', bg=BODY_COLOR)
                label.grid(**v['lpos'])
                if 'default' in v:
                    if type(str_var) is tk.Text:
                        str_var.delete(1.0, 'end')
                        str_var.insert(1.0, v['default'])
                    else:
                        str_var.set(v['default'])
        self.var_dict.update(var_dict)
        # btn = tk.Button(status_bar, text="Refresh", command=self._update_status)
        # btn.grid(row=10)

    def _create_demo_column(self):
        self.right_column = tk.Frame(master=self, width=500, height=600, bg=BODY_COLOR)
        self.right_column.grid(row=1, column=2, columnspan=3, sticky="nswe")
        canvas = FigureCanvasTkAgg(f, self.right_column)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, rowspan=1, sticky="nswe", padx=5, pady=5)

    def _animate_r_bias(self, i):
        a.clear()
        a.set_xlim((0, 50))
        a.set_title('Reward Moving Baseline')
        try:
            xList = []
            yList = []
            with open(os.path.join(self.global_wd.get(), "buffers.txt"), "r") as fh:
                for line in fh:
                    if not line.startswith('Episode'):
                        continue
                    ele = line.strip().split('\t')
                    x = ele[0].split(':')[1]
                    y = ele[2].split(':')[1]
                    xList.append(float(x))
                    yList.append(float(y))
            a.plot(xList, yList)
            a.set_xlim((0, np.ceil(max(xList) / 50) * 50))
        except FileNotFoundError:
            pass

    def _update_status(self):
        for k, v in self.var_dict.items():
            if k in self.status_bar_fn_map:
                # v_ = func_map[k]()
                # v[0].set(v_)
                self.status_bar_fn_map[k]()
        self.after(REFRESH_INTERVAL, self._update_status)

    def _build(self):
        try:
            var_dict = self.init_page.parse_vars(self.init_page)
        except Exception as e:
            messagebox.showinfo("Error", "Build was unsuccessful. See command-line output for details.")
            print("Failed build caused by %s" % e)
            return
        types, specs = gui_mapper(var_dict)
        fn = os.path.join(self.global_wd.get(), 'bionas_config.pkl')
        with open(fn, 'wb') as f:
            pickle.dump({'types': types, 'specs': specs}, f)

    def _run(self):
        fn = os.path.join(self.global_wd.get(), 'bionas_config.pkl')
        if not os.path.isfile(fn):
            messagebox.showinfo("Error", "BioNAS has not been built; did you click on Build?")
            return
        if self.has_run:
            messagebox.showinfo("Error", "BioNAS has already been running in background.. please wait")
        else:
            self.has_run = True
            cmd = ["python", "-c",
                   "import pickle;from BioNAS import BioNAS;d=pickle.load(open('%s','rb'));bn=BioNAS(**d);bn.run()" % fn]
            self.bn = self.global_thread_spawner(cmd)

    def _stop(self):
        if not os.path.isfile(os.path.join(self.global_wd.get(), 'bionas_config.pkl')):
            messagebox.showinfo("Error", "BioNAS has not been built; did you click on Build?")
            return
        if not self.has_run:
            messagebox.showinfo("Error", "BioNAS is not running")
        elif self.bn.poll() is None:
            self.bn.send_signal(signal.SIGINT)

        elif self.bn.poll() == 0:
            messagebox.showinfo("Info", "BioNAS finished running")
            self.has_run = False
            self.bn = None
        else:
            self.has_run = False
            if not self.bn.poll():
                self.bn.kill()
                messagebox.showinfo("Info", "BioNAS subprocess killed")
