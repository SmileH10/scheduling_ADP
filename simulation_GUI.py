import tkinter as tk
import tkinter.messagebox
import tkinter.ttk
from PIL import ImageTk, Image
from collections import defaultdict

UNIT = 50  # 픽셀 수
HEIGHT = 8  # 그리드월드 세로
WIDTH = 16  # 그리드월드 가로


class GraphicDisplay(tk.Tk):
    def __init__(self, env, mc_name_list):
        super(GraphicDisplay, self).__init__()
        self.title('Jobshop Simulation')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 100))
        self.backgroundimg, self.jobsimg = self.load_images()
        self.canvas = self._build_canvas()

        self.env = env  # simpy.Environment()
        self.iter = 0
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
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 10, window=run_entire_button)  # 왼쪽상단이 (0, 0) 인듯
        run_onestep_forward_button = tk.Button(self, text="Run(1step forward)", command=self.run_onestep_forward)
        run_onestep_forward_button.configure(width=17, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 10, window=run_onestep_forward_button)
        run_onestep_backward_button = tk.Button(self, text="Run(1step backward)", command=self.run_onestep_backward)
        run_onestep_backward_button.configure(width=17, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 10, window=run_onestep_backward_button)
        run_reset_button = tk.Button(self, text="reset", command=self.run_reset)
        run_reset_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 10, window=run_reset_button)

        # iter_plus_button = tk.Button(self, text="iter+", command=self.iter_plus)
        # iter_plus_button.configure(width=10, activebackground="#33B5E5")
        # canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 11, window=iter_plus_button)

        self.time_text = canvas.create_text(10, 10, text="time = 0.00", font=('Helvetica', '10', 'normal'), anchor="nw")



        canvas.pack()
        return canvas

    def save_status(self, simiter, time, status):
        if self.event_cnt == 0 or self.previous_status != status:
            self.data[simiter][self.event_cnt] = (time, status)
            self.event_cnt += 1
            self.previous_status = status

    def printing_time(self, sim_t, font='Helvetica', size=12, style='normal', anchor="nw"):
        time_str = "time = %.2f" % sim_t
        font = (font, str(size), style)
        self.canvas.delete(self.time_text)
        self.time_text = self.canvas.create_text(10, 10, fill="black", text=time_str, font=font, anchor=anchor)
        return self.time_text

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
            return 9.52 * UNIT
        elif mc == 'G' or mc == 'I' or mc == 'J':
            return 15.5 * UNIT

    @staticmethod
    def locy(mc):
        if mc == 'A' or mc == 'D' or mc == 'G':
            return 1 * UNIT
        elif mc == 'B' or mc == 'E' or mc == 'I':
            return 4 * UNIT
        elif mc == 'C' or mc == 'F' or mc == 'J':
            return 7 * UNIT

    @staticmethod
    def load_images():
        background = ImageTk.PhotoImage(Image.open("./img/background.png").resize((WIDTH*UNIT, HEIGHT*UNIT)))
        circles = [_ for _ in range(3)]
        for i in range(3):
            circles[i] = ImageTk.PhotoImage(Image.open("./img/circle%d.png" % i).resize((15, 15)))
        return (background, circles)

    def run_entire(self):
        while self.is_moving != 1:
            self.is_moving = 1
            while self.event_cnt <= len(self.data[self.iter]) - 2:
                current_time = self.data[self.iter][self.event_cnt][0]
                next_time = self.data[self.iter][self.event_cnt + 1][0]
                self.after(100 * (next_time - current_time), self.run_onestep_forward())
                self.update()
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
        self.event_cnt = 0
        time, status = self.data[self.iter][self.event_cnt]
        self.printing_time(time)
        self.draw_status(status)

    def iter_plus(self):
        self.iter += 1
        self.run_reset()


if __name__ == '__main__':
    pass
