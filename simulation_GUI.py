import tkinter as tk
import tkinter.messagebox
import tkinter.ttk
from PIL import ImageTk, Image
from collections import defaultdict
from time import sleep

UNIT = 50  # 픽셀 수
HEIGHT = 8  # 그리드월드 세로
WIDTH = 16  # 그리드월드 가로


class GraphicDisplay(tk.Tk):
    def __init__(self, env, mc_name_list):
        super(GraphicDisplay, self).__init__()
        self.title('Jobshop Simulation')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.backgroundimg, self.jobsimg = self.load_images()
        self.canvas = self._build_canvas()

        self.env = env  # simpy.Environment()
        self.iter = 0
        self.sim_speed = 0.01
        self.mc_name_list = mc_name_list
        self.event_cnt = 0
        self.is_moving = 0

        self.data = defaultdict(dict)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        canvas.create_image(WIDTH * UNIT / 2, HEIGHT * UNIT / 2, image=self.backgroundimg)

        run_entire_button = tk.Button(self, text="Run(entire)", command=self.run_entire)
        run_entire_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.08, HEIGHT * UNIT + 10, window=run_entire_button)  # 왼쪽상단이 (0, 0) 인듯
        run_onestep_forward_button = tk.Button(self, text="Run(1step forward)", command=self.run_onestep_forward)
        run_onestep_forward_button.configure(width=17, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.22, HEIGHT * UNIT + 10, window=run_onestep_forward_button)
        run_onestep_backward_button = tk.Button(self, text="Run(1step backward)", command=self.run_onestep_backward)
        run_onestep_backward_button.configure(width=17, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.385, HEIGHT * UNIT + 10, window=run_onestep_backward_button)
        run_reset_button = tk.Button(self, text="reset", command=self.run_reset)
        run_reset_button.configure(width=8, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.525, HEIGHT * UNIT + 10, window=run_reset_button)
        run_pause_button = tk.Button(self, text="pause", command=self.run_pause)
        run_pause_button.configure(width=8, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.61, HEIGHT * UNIT + 10, window=run_pause_button)

        iter_minus_button = tk.Button(self, text="iter-", command=self.iter_minus)
        iter_minus_button.configure(width=8, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.74, HEIGHT * UNIT + 10, window=iter_minus_button)
        iter_plus_button = tk.Button(self, text="iter+", command=self.iter_plus)
        iter_plus_button.configure(width=8, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.825, HEIGHT * UNIT + 10, window=iter_plus_button)

        change_speed_button = tk.Button(self, text="speed", command=self.change_speed)
        change_speed_button.configure(width=8, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.945, HEIGHT * UNIT + 10, window=change_speed_button)

        self.time_text = canvas.create_text(10, 10, text="time = 0.00", font=('Helvetica', '10', 'normal'), anchor="nw")

        canvas.pack()
        return canvas

    def save_status(self, simiter, time, status):
        if self.event_cnt == 0 or self.previous_status != status:
            self.data[simiter][self.event_cnt] = (time, status)
            self.event_cnt += 1
            self.previous_status = status

    def printing_time(self, sim_t, font='Helvetica', size=12, style='bold', anchor="nw"):
        hour = sim_t / (60.0 * 60)
        minute = sim_t % (60.0 * 60) / 60.0
        second = sim_t % 60.0
        time_str = "time = %.1f secs (%d : %d : %.1f)" % (sim_t, hour, minute, second)
        font = (font, str(size), style)
        self.canvas.delete(self.time_text)
        self.time_text = self.canvas.create_text(10, 10, fill="black", text=time_str, font=font, anchor=anchor)

    def printing_iter(self, font='Helvetica', size=12, style='bold', anchor="ne"):
        iter_str = "iteration: %d / %d (Total %d)" % (self.iter, len(self.data.keys()) - 1, len(self.data.keys()))
        font = (font, str(size), style)
        if hasattr(self, 'iter_text'):
            self.canvas.delete(self.iter_text)
        self.iter_text = self.canvas.create_text(WIDTH * UNIT - 10, 10, fill="black", text=iter_str, font=font, anchor=anchor)

    def draw_status(self, status):
        # delete every previous drawing
        if hasattr(self, 'mc_users'):
            for mc in self.mc_name_list:
                if self.mc_users[mc].keys():
                    self.canvas.delete(self.mc_users[mc]['text'])
                    self.canvas.delete(self.mc_users[mc]['image'])
        if hasattr(self, 'mc_queue'):
            for mc in self.mc_name_list:
                if self.mc_queue[mc].keys():
                    for job in self.mc_queue[mc].keys():
                        self.canvas.delete(self.mc_queue[mc][job]['text'])
                        self.canvas.delete(self.mc_queue[mc][job]['image'])
        # initialize
        self.mc_users = defaultdict(dict)
        self.mc_queue = defaultdict(dict)
        # New draw
        for mc in self.mc_name_list:
            # job using resources
            if status['mc_users'][mc] != 'empty':
                self.mc_users[mc]['image'] = self.canvas.create_image(self.locx(mc), self.locy(mc),
                                                                      image=self.jobsimg[status['mc_users'][mc].pattern])
                self.mc_users[mc]['text'] = self.canvas.create_text(self.locx(mc), self.locy(mc),
                                                                    text=str(status['mc_users'][mc].id))
            # jobs in queues
            if status['mc_queue'][mc] != 'empty':
                x = self.locx(mc) - 40
                for job in status['mc_queue'][mc].keys():
                    self.mc_queue[mc][job] = {'image': self.canvas.create_image(x, self.locy(mc) - 12,
                                                                                image=self.jobsimg[status['mc_queue'][mc][job].pattern]),
                                              'text': self.canvas.create_text(x, self.locy(mc) - 12,
                                                                              text=str(status['mc_queue'][mc][job].id))
                                              }
                    x -= 15
    @staticmethod
    def locx(mc):
        if mc == 'A' or mc == 'B' or mc == 'C':
            return 3.53 * UNIT
        elif mc == 'D' or mc == 'E' or mc == 'F':
            return 9.54 * UNIT
        elif mc == 'G' or mc == 'I' or mc == 'J':
            return 15.545 * UNIT

    @staticmethod
    def locy(mc):
        if mc == 'A' or mc == 'D' or mc == 'G':
            return 2 * UNIT
        elif mc == 'B' or mc == 'E' or mc == 'I':
            return 4 * UNIT
        elif mc == 'C' or mc == 'F' or mc == 'J':
            return 6 * UNIT

    @staticmethod
    def load_images():
        background = ImageTk.PhotoImage(Image.open("./img/background2.png").resize((WIDTH*UNIT, HEIGHT*UNIT)))
        jobsimg = []
        for i in range(12):
            jobsimg.append(ImageTk.PhotoImage(Image.open("./img/circle%d.png" % i).resize((15, 15))))
        for i in range(12):
            jobsimg.append(ImageTk.PhotoImage(Image.open("./img/triangle%d.png" % i).resize((15, 15))))
        return background, jobsimg

    def run_entire(self):
        self.is_moving = 1
        while self.event_cnt <= len(self.data[self.iter]) - 2:
            current_time = self.data[self.iter][self.event_cnt][0]
            next_time = self.data[self.iter][self.event_cnt + 1][0]
            self.after(int(self.sim_speed * (next_time - current_time)), self.run_onestep_forward())
            self.update()
            if self.is_moving == 0:
                break
        self.is_moving = 0

    def run_onestep_forward(self):
        if -1 <= self.event_cnt <= len(self.data[self.iter]) - 2:
            self.event_cnt += 1
            time, status = self.data[self.iter][self.event_cnt]
            self.printing_time(time)
            self.draw_status(status)
        else:
            tk.messagebox.showwarning("Error", "simulation ends")

    def run_onestep_backward(self):
        if 1 <= self.event_cnt <= len(self.data[self.iter]):
            self.event_cnt -= 1
            time, status = self.data[self.iter][self.event_cnt]
            self.printing_time(time)
            self.draw_status(status)
        else:
            tk.messagebox.showwarning("Error", "you're at the simulation starting point")

    def run_reset(self):
        self.is_moving = 0
        sleep(0.01)
        self.event_cnt = 0
        time, status = self.data[self.iter][self.event_cnt]
        self.printing_time(time)
        self.printing_iter()
        self.draw_status(status)

    def run_pause(self):
        self.is_moving = 0

    def iter_plus(self):
        if -1 <= self.iter < len(self.data.keys()) - 1:
            self.iter += 1
            self.run_reset()
        else:
            tk.messagebox.showwarning("Error", "iteration ends")

    def iter_minus(self):
        if 1 <= self.iter < len(self.data.keys()) + 1:
            self.iter -= 1
            self.run_reset()
        else:
            tk.messagebox.showwarning("Error", "you're at the first iteration")

    def change_speed(self):
        win_entry = tk.Toplevel(self)
        win_entry.title("enter simulation speed")
        win_entry.geometry('300x120+550+450')

        def check_ok(event=None):
            tk.messagebox.showinfo("simulation speed info", "simulation speed changes to %s" % input_num.get())
            self.sim_speed = int(input_num.get()) / 10000.0
            win_entry.destroy()
        input_num = tk.StringVar()

        label = tk.Label(win_entry)
        label.config(text="enter simulation speed \n (integer, the smaller the faster) \n (default: 100 real_time: 1M)")
        label.pack(ipadx=5, ipady=5)
        textbox = tk.ttk.Entry(win_entry, width=10, textvariable=input_num)
        textbox.pack(ipadx=5, ipady=5)
        action = tk.ttk.Button(win_entry, text="OK", command=check_ok)
        action.pack(ipadx=5, ipady=5)
        win_entry.bind("<Return>", check_ok)
        win_entry.mainloop()


if __name__ == '__main__':
    pass
