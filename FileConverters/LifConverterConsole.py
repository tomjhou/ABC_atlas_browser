from LifClass import LifClass
import tkinter as tk
from tkinter import filedialog
import os

print('\nSelect what to convert:')
print('1. All .LIF files in one folder (DEFAULT)')
print('2. Single .LIF file')
print('3. Single image within single .LIF file')
answer_source = input('Make selection (default = 1): ')
if answer_source == '':
    answer_source = '1'

conversion_options = LifClass.Options()

print('\nSelect format to convert to:')
print('1. JPG (90% quality, much smaller files, highly recommended, DEFAULT)')
print('2. TIFF (larger files, but compression is lossless. Only use if higher quality is absolutely necessary. Not tested for BigTIFF >4GB files')
answer = input('Make selection (default = 1): ')
if answer == '' or answer == '1':
    conversion_options.convert_format = LifClass.Options.Format.jpg
elif answer == '2':
    conversion_options.convert_format = LifClass.Options.Format.tiff
else:
    conversion_options.convert_format = LifClass.Options.Format.jpg

if answer_source == 1:
    print('\nOverwrite files that already exist?')
    print('y. Use this if files have changed')
    print('n. This saves time by skipping files that were already converte earlier (DEFAULT)')
    answer = input('Make selection (default = n): ')
    conversion_options.overwrite_existing = (answer == 'y')
else:
    # If converting just one file, then always overwrite
    conversion_options.overwrite_existing = True

answer = input('\nWrite each LIF file\'s metadata to XML file? (y/n, default = n):')
conversion_options.write_xml_metadata = (answer == 'y')

if answer_source == '1':
    root = tk.Tk()  # pointing root to Tk() to use it as Tk() in program.
    root.withdraw()  # Hides small tkinter window.
    root.attributes('-topmost', True)  # Opened windows will be active above all windows in spite of selection.
    folder_path = filedialog.askdirectory()

    if folder_path != '':
        files = os.scandir(folder_path)

        num_files_done = 0
        num_images_done = 0

        for f in files:
            if f.is_dir():
                continue

            path_ext = os.path.splitext(f)
            if path_ext[1] != '.lif':
                continue

            print(f'\nProcessing file {f.name}')
            l1 = LifClass(f.path, conversion_options=conversion_options)
            r = l1.convert()
            num_files_done += r[0]
            num_images_done += r[1]

        print(f'\nCompleted conversion of {num_images_done} images in {num_files_done} LIF files in folder\n')
    else:
        print('\nNo folder chosen.\n')

elif answer_source == '2':

    l1 = LifClass(conversion_options=conversion_options)
    l1.convert()

elif answer_source == '3':

    l1 = LifClass(conversion_options=conversion_options)
    n = l1.prompt_select_image_from_single_file()
    l1.convert(n)

else:
    print('No valid option selected. Will not do anything')



