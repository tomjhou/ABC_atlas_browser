import os
import tkinter as tk
from tkinter import ttk, filedialog

from LifClass import LifClass
from basic_gui import basic_gui, basic_flag

PADDING_PIXELS = 5  # How much padding to put around GUI buttons
USE_TWO_BUTTON_COLUMNS = True  # If true, buttons are in 2 columns, otherwise will be in 1 column


class gui(basic_gui):
    root = None
    stopFlag = basic_flag()
    lif_object: LifClass = None
    output_folder = None

    def __init__(self):

        super().__init__()
        self.var_recursive = tk.BooleanVar(self.root)
        self.var_rotate180 = tk.BooleanVar(self.root, True)

        # This is set by radio buttons in folder options
        self.skip_string_var = tk.StringVar(self.root, "skip")
        # This is set by radio buttons in file format
        self.format1_string_var = tk.StringVar(self.root, LifClass.Options.Format.jpg.name)
        # This is set by radio buttons in color format
        self.format2_string_var = tk.StringVar(self.root, LifClass.Options.ColorOptions.RGB_CMY.name)  # "RGB_CMY")
        # This is set by radio buttons in z-stack format
        self.format3_string_var = tk.StringVar(self.root, LifClass.Options.ZStackOptions.max_project.name)

        self.conversion_options = LifClass.Options()
        self.file_path = None

    def do_open_output_folder(self):

        if self.output_folder is not None:
            os.startfile(self.output_folder)

    def get_options_from_gui(self):

        v1 = self.format1_string_var.get()
        v2 = self.format2_string_var.get()
        v3 = self.format3_string_var.get()

        # Determine whether to export jpg, tif, or xml
        self.conversion_options.convert_format = LifClass.Options.Format[v1]

        self.conversion_options.overwrite_existing = (self.skip_string_var.get() == "all")
        self.conversion_options.rotate180 = self.var_rotate180.get()

        self.conversion_options.color_format = LifClass.Options.ColorOptions[v2]

        self.conversion_options.zstack_format = LifClass.Options.ZStackOptions[v3]

    def get_file_list(self, folder_path):

        final_list = []

        files = os.scandir(folder_path)

        f_list = list(files)

        recursive = self.var_recursive.get()

        for f in f_list:
            if f.is_dir():
                if recursive:
                    final_list = final_list + self.get_file_list(f)
                continue

            if not f.is_file():
                continue

            path_ext = os.path.splitext(f)
            if path_ext[1] != '.lif':
                # Exclude files that are not .lif
                continue

            final_list.append(f)

        return final_list

    def start_convert_folder(self):

        # self.root.attributes('-topmost', True)  # Opened windows will be above all windows
        folder_path = filedialog.askdirectory()

        if folder_path == '':
            print('\nNo folder chosen.\n')
            self.status.set_text(0, "No folder chosen")
            return

        self.output_folder = folder_path

        file_list = self.get_file_list(folder_path)

        self.get_options_from_gui()

        num_files_done = 0
        num_files = len(file_list)

        self.lif_object = LifClass(conversion_options=self.conversion_options, root_window=self.root)

        if self.conversion_options.overwrite_existing:
            string1 = "all files"
        else:
            string1 = "all unconverted files"

        if self.var_recursive.get():
            string2 = ", including subfolders"
        else:
            string2 = "."

        print(f'\nProcessing {string1} in folder "{folder_path}"{string2}')

        try:
            for x in range(num_files):

                if self.stopFlag.check():
                    raise LifClass.UserCanceled

                f = file_list[x]
                print(f'\nProcessing file "{f.name}"')
                self.status.set_text(0, f'({x + 1} of {num_files}) {f.name}')
                self.status.set_text(1, ("Folder:", os.path.dirname(f.path)))

                self.root.update()
                self.lif_object.open_file(f.path)
                self.lif_object.convert()
                num_files_done += 1
        except LifClass.UserCanceled:
            pass
        except Exception as e:
            print('Unexpected exception ' + str(e))
        else:
            self.print_results(self.lif_object, num_files_done)

    def print_results(self, lo: LifClass, num_files: int = 0):

        print(f'\nConverted {lo.num_images_converted} images ', end='')
        if num_files > 0:
            print(f'in {num_files} LIF files. ', end='')
        else:
            print('. ', end='')
        if lo.num_xml_written > 0:
            print(f'Wrote {lo.num_xml_written} XML files. ', end='')
        print(f'Skipped {self.lif_object.num_images_skipped} images, '
              f'encountered errors in {self.lif_object.num_images_error} images/files\n')

    def start_convert_single_image_in_file(self):

        self.get_options_from_gui()
        # If converting just one file, then always overwrite
        self.conversion_options.overwrite_existing = True
        self.lif_object = LifClass(conversion_options=self.conversion_options, root_window=self.root)

        try:
            file_path = self.lif_object.prompt_select_file()
        except LifClass.UserCanceled:
            self.status.set_text(0, "User canceled")
            return

        base_name = os.path.basename(file_path)
        self.output_folder = os.path.dirname(file_path)
        print(f'\nProcessing file "{base_name}"')
        self.status.set_text(0, base_name)
        # Update GUI now because opening/converting lif file might take a long time
        self.root.update()

        try:
            self.lif_object.open_file(file_path)
            n = self.lif_object.prompt_select_image_from_single_file()
            self.lif_object.convert(n=n)
        except Exception as e:
            print("Error during conversion: " + str(e))
        else:
            self.print_results(self.lif_object)

    def start_convert_file(self):

        self.get_options_from_gui()
        # If converting just one file, then always overwrite
        self.conversion_options.overwrite_existing = True
        self.lif_object = LifClass(conversion_options=self.conversion_options, root_window=self.root)

        try:
            file_path = self.lif_object.prompt_select_file()
        except LifClass.UserCanceled:
            self.status.set_text(0, "User canceled")
            return

        base_name = os.path.basename(file_path)
        self.output_folder = os.path.dirname(file_path)
        print(f'\nProcessing file "{base_name}"')
        self.status.set_text(0, base_name)
        # Update GUI now because opening/converting lif file might take a long time
        self.root.update()

        try:
            self.lif_object.open_file(file_path)
            self.lif_object.convert()
        except Exception as e:
            print("Error during conversion: " + str(e))
        else:
            self.print_results(self.lif_object)

    def start_convert_image(self):

        self.get_options_from_gui()
        self.lif_object = LifClass(conversion_options=self.conversion_options, root_window=self.root)
        try:
            file_path = self.lif_object.prompt_select_file()
        except LifClass.UserCanceled:
            self.status.set_text(0, "User canceled")
            return

        self.status.set_text(0, self.lif_object.file_base_name)
        self.root.update()

        try:
            self.lif_object.open_file(file_path)
            n = self.lif_object.prompt_select_image_from_single_file()
            self.lif_object.convert(n)
        except Exception as e:
            print("Error during conversion: " + str(e))
        else:
            print(f'\nConverted {self.lif_object.num_images_converted} images')

    def operation_end(self):
        # Re-enable the "stop conversion" button
        self.set_widget_state(self.button_list[-2], True)
        # Re-enable remaining buttons
        super().operation_end()

    def run_gui(self):

        #
        #  -------------------------------  <-- start of frame1, parent = root
        #  ||---------------------------||  <-- start of frame1b, parent = frame1
        #  ||           |   Radio       ||
        #  ||  Buttons  |   Buttons     ||
        #  ||           |               ||
        #  ||---------------------------||  <-- end of frame1b
        #  ||---------------------------||
        #  ||  frame 1a checkboxes      ||  <-- frame1a, parent = frame1
        #  ||---------------------------||
        #  ||---------------------------||  <-- end of frame1
        #  ||  Status text boxes            ||

        # Create GUI
        self.root.title(string="Jhou lab LIF converter")

        frame1 = tk.Frame(self.root)
        frame1.pack(side=tk.TOP, anchor='w', padx=15, pady=10)

        frame1b = tk.Frame(frame1, borderwidth=5)
        frame1b.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        ##################################################
        #
        #    List of buttons at left top panel
        #
        ##################################################
        values = [("Convert single image within a LIF file", self.start_convert_single_image_in_file),
                  ("Convert single LIF file", self.start_convert_file),
                  ("Convert folder", self.start_convert_folder)]

        self.button_list = self.add_boxed_button_column(frame1b, values,
                                                        side=tk.LEFT, fill=tk.X,
                                                        add_exit=False)

        ##################################################
        #
        #    Radio buttons at top right
        #
        ##################################################

        # ****** Output options
        values = [("JPG (smallest files)", LifClass.Options.Format.jpg),
                  ("TIFF (highest quality - best for importing to Illustrator)", LifClass.Options.Format.tiff),
                  ("XML (extracts header info only)", LifClass.Options.Format.xml)]

        (f, elt) = self.add_boxed_radio_button_column(frame1b, values, backing_var=self.format1_string_var,
                                                      side=tk.TOP, fill=tk.X,
                                                      padx=15, pady=8,
                                                      text="Output options")

        cb2 = ttk.Checkbutton(f, text="Rotate 180 degrees?", variable=self.var_rotate180)
        cb2.pack(before=elt, side=tk.TOP, anchor=tk.NW, padx=10, pady=(6, 3))

        # ****** Color layer options
        values = [  # ("Put all color layers into single file", "all_together"),
                  ("Merge RGB into single file (CMY will be in separate files)", LifClass.Options.ColorOptions.RGB_CMY),
                  ("Each color in separate file (best for importing into Illustrator)", LifClass.Options.ColorOptions.all_separate)]

        (f, elt) = self.add_boxed_radio_button_column(frame1b, values, backing_var=self.format2_string_var,
                                                      side=tk.TOP, fill=tk.X,
                                                      padx=15, pady=8,
                                                      text="Color layer options")

        # ****** Z-stack options
        values = [("Generate max projection", LifClass.Options.ZStackOptions.max_project),
                  ("Generate separate image for each z-stack layer", LifClass.Options.ZStackOptions.separate)]

        (f, elt) = self.add_boxed_radio_button_column(frame1b, values, backing_var=self.format3_string_var,
                                                      side=tk.TOP, fill=tk.X,
                                                      padx=15, pady=8,
                                                      text="How to handle z-stacks?")

        # ****** File convert options
        values = [("Skip already converted files (recommended)", "skip"),
                  ("Convert all (overwrites old output)", "all")]

        (f, elt) = self.add_boxed_radio_button_column(frame1b, values, backing_var=self.skip_string_var,
                                                      side=tk.TOP, fill=tk.X,
                                                      padx=15, pady=8,
                                                      text="File convert options")

        cb = ttk.Checkbutton(f, text="Include sub-folders?", variable=self.var_recursive)
        cb.pack(before=elt, side=tk.TOP, anchor=tk.NW, padx=10, pady=(6, 3))

        frame1a = tk.Frame(frame1, borderwidth=5)
        frame1a.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=5)

        ##################################################
        #
        #    Status text boxes at bottom
        #
        ##################################################

        values = ["Current file:",
                  "Status:"]

        self.status.add(self.root, values)

        ##################################################
        #
        #    Open folder and Exit buttons at bottom
        #
        ##################################################

        b = ttk.Button(self.root, text="Open output folder", command=self.do_open_output_folder)
        b.pack(side=tk.TOP, padx=15, pady=(5, 5), ipadx=15, ipady=12)

        self.add_exit_button(self.root, fill=False)

        # Place in top left corner of screen
        self.root.geometry("+%d+%d" % (PADDING_PIXELS, PADDING_PIXELS))

        # Force window to show, otherwise winfo_geometry() will return zero
        frame1.update()
        print("Created main window with geometry " + self.root.winfo_geometry())

        print("\nDon't close this window, as it will report useful messages during conversion.")

        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

        super().run_gui()

        print("Finished")


print('Creating GUI object')
obj = gui()
print('Launching GUI')
obj.run_gui()
