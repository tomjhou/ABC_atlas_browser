import tkinter as tk
import enum
from tkinter import ttk
from functools import partial
import ctypes
import threading
from typing import Optional

PADDING_PIXELS = 5  # How much padding to put around GUI buttons
USE_TWO_BUTTON_COLUMNS = True  # If true, buttons are in 2 columns, otherwise will be in 1 column


class basic_flag:

    def __init__(self, root_window=None):
        self.flag = False
        self.root_window = root_window

    def set(self):
        self.flag = True

    def check(self):
        if self.root_window is not None:
            self.root_window.update()
        if self.flag:
            print('Stop flag triggered')
            self.flag = False
            return True
        else:
            return False


def set_widget_text(widget, text):
    if widget is None:
        return
    widget["text"] = text


class basic_gui:

    class ThreadRunning(Exception):
        # Custom exception class, no code needed.
        # Raise this exception to prevent button handler
        # from calling session_end code if operation is running on background thread
        pass

    class thread_with_exception(threading.Thread):
        def __init__(self, *args, **kwargs):
            threading.Thread.__init__(self, *args, **kwargs)
            self.name = "conversion_thread_678"

        def get_id(self):

            # returns id of the respective thread
            if hasattr(self, '_thread_id'):
                return self._thread_id

            # Pycharm complains that _active is protected. But it works anyway, because this is a child class.
            for th_id, thread in threading._active.items():
                if thread is self:
                    return th_id

            return None

        def raise_exception(self):
            thread_id = self.get_id()
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                             ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print('Exception raise failure')

    root = None
    lock1 = threading.Lock()
    current_thread: Optional[thread_with_exception] = None
    button_cancel = None
    button_exit = None

    class StatusLines:

        label_list = []
        title_list = []
        default_title_list = []

        def add(self, parent_frame, values):
            frame = tk.Frame(parent_frame, borderwidth=5)
            frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
            frame.columnconfigure(1, weight=1)

            self.default_title_list = []
            self.title_list = []
            self.label_list = []

            for x in range(len(values)):
                l1 = tk.Label(frame, text=values[x], anchor="e", justify=tk.RIGHT)
                l1.grid(row=x, column=0, sticky=tk.E)
                l2 = tk.Label(frame, borderwidth=1, anchor="w", relief="sunken")
                l2.grid(row=x, column=1, sticky=tk.W + tk.E)

                self.default_title_list.append(values[x])
                self.title_list.append(l1)
                self.label_list.append(l2)

        def clear(self):
            for b in self.label_list:
                # Column 1 is wide text line
                set_widget_text(b, "")
            for x in range(len(self.title_list)):
                # Column 1 is wide text line
                set_widget_text(self.title_list[x], self.default_title_list[x])

        def set_text(self, row, text):
            if type(text) is tuple:
                set_widget_text(self.title_list[row], text[0])
                set_widget_text(self.label_list[row], text[1])
            else:
                set_widget_text(self.title_list[row], self.default_title_list[row])
                set_widget_text(self.label_list[row], text)

    def __init__(self):

        root = tk.Tk()
        self.root = root
        self.button_list = []
        self.status: basic_gui.StatusLines = self.StatusLines()

    @staticmethod
    def set_widget_state(widget, state: bool):
        if widget is None:
            return
        if state:
            widget["state"] = "enabled"
        else:
            widget["state"] = "disabled"

    def set_enabled_status(self, enabled: bool):
        for b in self.button_list:
            self.set_widget_state(b, enabled)

    def add_boxed_radio_button_column(self, parent_frame, button_names=None,
                                      backing_var=None, command=None,
                                      side=None, fill=None,
                                      padx=0, pady=0, text=""):

        if button_names is None:
            button_names = {}

        # Label frame is lighter and allows text label. We use it here just to match the appearance of format box
        frame = tk.LabelFrame(parent_frame, text=text)
        frame.pack(side=side, fill=fill, padx=padx, pady=pady, ipadx=5, ipady=5)

        elt = self.add_radio_button_column(frame, button_names, backing_var, command)

        # Returns inner frame and first element within it, to allow insertion of new widgets just above it.
        return frame, elt

    @staticmethod
    def add_radio_button_column(parent_frame, button_names, backing_var, command=None):

        pady = 6
        first_elt = None
        for v in button_names:
            if isinstance(v[1], enum.Enum):
                # If enumerated type, then convert to string
                value = v[1].name
            else:
                value = v[1]

            b = tk.Radiobutton(parent_frame, text=v[0], variable=backing_var, value=value, command=command)
            b.pack(side=tk.TOP, anchor='w', pady=(pady, 0), padx=10)

            if first_elt is None:
                first_elt = b

            # Extra padding only on first one
            pady = 0

        return first_elt

    def add_boxed_button_column(self, parent_frame, button_names, side=None, fill=None, add_cancel=True, add_exit=True):

        frame = tk.Frame(parent_frame, highlightbackground="black", highlightthickness=1, relief="flat", borderwidth=5)
        frame.pack(side=side, fill=fill, padx=2, pady=2)

        frame_inset = tk.Frame(frame)  # , highlightbackground="black", highlightthickness=1)
        frame_inset.pack(anchor=tk.CENTER, pady=10)
        return self.add_button_column(frame_inset, button_names, add_cancel=add_cancel, add_exit=add_exit)

    BUTTON_INNER_PADDING = 10

    def add_button(self, parent_frame, text, command, fill=True):
        butt = ttk.Button(parent_frame, text=text, command=command)
        if fill:
            butt.pack(fill=tk.X, padx=10, pady=5, ipadx=self.BUTTON_INNER_PADDING, ipady=self.BUTTON_INNER_PADDING)
        else:
            butt.pack(padx=10, pady=5, ipadx=self.BUTTON_INNER_PADDING, ipady=self.BUTTON_INNER_PADDING)
        return butt

    def add_exit_button(self, parent_frame, fill=True):
        self.button_exit = self.add_button(parent_frame, "Exit", partial(self.do_exit), fill=fill)

    def add_button_column(self, parent_frame, button_names, add_cancel=True, add_exit=True):

        inner_padding = 10

        button_list = []
        button_index = 0
        for v in button_names:
            butt = self.add_button(parent_frame, text=v[0], command=partial(self.handle_button, v[1]))
            button_list.append(butt)
            button_index += 1

        if add_cancel:
            butt = self.add_button(parent_frame, text="Cancel", command=partial(self.do_cancel))
            self.button_cancel = butt

        if add_exit:
            self.add_exit_button(parent_frame)

        return button_list

    def do_cancel(self):
        self.lock1.acquire()
        if self.current_thread is not None:
            print('User has requested to cancel conversion. Now awaiting end of thread.')
            # Disable cancel button while awaiting cancellation
            self.set_widget_state(self.button_cancel, False)
            set_widget_text(self.button_cancel, "Cancel pending...")
            self.status.set_text(1, "Waiting for thread to end. This can take up to a minute or two...")

            # This works unless the thread is stuck in external code, e.g. waiting for file read/write operation
            # or performing a long mathematical operation. There seems to be no way to kill a thread in the middle of
            # such a process. Whereas it might be possible to terminate() a Process, but we can't use that because
            # the GUI has to all run on the same Process.
            self.current_thread.raise_exception()
        else:
            # Thread already ended? Should we just ignore?
            pass
        self.lock1.release()

    def do_exit(self):
        self.root.quit()

    def operation_end(self):
        # Call this at end of operation to re-enable buttons
        self.set_enabled_status(True)
        print('Conversion ended')
        self.set_widget_state(self.button_cancel, True)
        set_widget_text(self.button_cancel, "Cancel")
        self.status.set_text(1, "Completed")
        self.current_thread = None

    def handle_button(self, cmd):
        # Disable buttons while conversion is taking place
        self.set_enabled_status(False)
        self.status.clear()
        self.current_thread = self.thread_with_exception(target=partial(self._threaded_worker, cmd))
        # Start conversion in background
        self.current_thread.start()

    def _threaded_worker(self, cmd) -> None:
        try:
            cmd()
        except Exception as e:
            # Somehow cancel does not send code here, but goes straight to finally clause
            print('Error during conversion: ' + str(e))
        finally:
            self.lock1.acquire()
            self.operation_end()
            self.lock1.release()

    def run_gui(self):
        """
            Runs GUI loop. This function will not return until cancel is requested
        """
        tk.mainloop()
