import os
import time
import enum
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2  # install with pip install opencv-python
from typing import Optional

from basic_gui import basic_flag
from reader import LifFile  # This supersedes the install with pip install readlif


class LifClass:
    class Options:
        """
            Options for converting files. These specify output formats and whether to overwrite existing files
        """

        class Format(enum.Enum):
            xml = 0
            jpg = 1
            tiff = 2

        class ColorOptions(enum.Enum):
            all_together = 0
            RGB_CMY = 1
            all_separate = 2

        class ZStackOptions(enum.Enum):
            max_project = 0
            separate = 1

        overwrite_existing = True
        rotate180 = True
#        separate_CMY = True  # Put cyan, magenta and yellow into their own file if needed to avoid overlap

        def __init__(self):
            self.convert_format = self.Format.jpg
            self.color_format = self.ColorOptions.RGB_CMY
            self.zstack_format = self.ZStackOptions.max_project

    # The following variables are cumulative, i.e. if you convert more than one file with the same object
    # they will not be reset between conversions.
    num_images_converted = 0
    num_images_skipped = 0
    num_images_error = 0
    num_xml_written = 0

    conversion_options = Options()
    lif_modified_time = None
    root_window = None
    file_path = None
    file_base_name = None

    lif_file_object: Optional[LifFile] = None

    class UserCanceled(Exception):
        # Custom exception class, no code needed.
        pass

    def __init__(self, conversion_options: Options = None, root_window=None):

        if root_window is not None:
            self.root_window = root_window

        self.stopFlag = basic_flag(self.root_window)  # Used to stop ongoing conversion

        if conversion_options is not None:
            self.conversion_options = conversion_options

    def prompt_select_file(self):

        if self.root_window is None:
            self.root_window = tk.Tk()
            self.root_window.withdraw()
            self.root_window.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(filetypes=[("LIF files", "*.lif; *.xlef")])

        if file_path == "":
            raise self.UserCanceled

        return file_path

    def open_file(self, file_path):

        self.file_path = file_path
        self.file_base_name = os.path.basename(file_path)
        self.lif_modified_time = os.path.getmtime(file_path)

        try:
            self.lif_file_object = LifFile(file_path)
        except Exception as e:
            print(f'  Error encountered while opening LIF file: {e}')
            print('  Please check whether file is corrupted.')
            self.num_images_error += 1

    def prompt_select_image_from_single_file(self):

        img_list = [i for i in self.lif_file_object.get_iter_image()]
        print(f'\nChoose image to convert within file "{self.file_base_name}":')
        for n in range(len(img_list)):
            print(f'{n}: "{img_list[n].name}", width {img_list[n].dims.x} x height {img_list[n].dims.y}')
        answer = input('Select image (default = 0): ')
        if answer == '':
            answer = '0'

        return int(answer)

    def stop_conversion(self):
        self.stopFlag.set()

    # Convert one or more images in a single file
    def convert(self, n: int = -1):

        # Access a specific image directly
        # img_0 = new.get_image(0)
        # Create a list of images using a generator
        try:
            img_list = [i for i in self.lif_file_object.get_iter_image()]
        except Exception as e:
            print(f'\n  Error while reading image list: {e}')
            print('  Unable to convert file, possibly due to file corruption or truncation. Will skip')
            self.num_images_error += 1
            return

        xml_metadata = self._recursive_metadata_find(
            self.lif_file_object.xml_root, )  # self.xml_root.findall("./Element/Children/Element/Data/Image")

        if self.conversion_options.convert_format == LifClass.Options.Format.xml:
            # Extract xml header info only
            paths = os.path.splitext(self.file_path)
            xml_path = paths[0] + ".xml"

            with open(xml_path, 'w') as f:
                f.write(self.lif_file_object.xml_header)

            print(f'  Wrote XML file "{os.path.basename(xml_path)}"')
            self.num_xml_written += 1
            return

        if n < 0:
            # Convert all images in file
            print(f'  Found {len(img_list)} image(s) in file "{self.file_base_name}".')
            for n in range(len(img_list)):
                if self.stopFlag.check():
                    raise self.UserCanceled
                print(f'   {n + 1}: ', end="")
                self.convert_image(img_list[n], xml_metadata[n])
        else:
            # Convert single image
            self.convert_image(img_list[n], xml_metadata[n])

        return

    # Convert a single image within this file.
    def convert_image(self, img, xml_metadata):

        if img.dims.m > 1:
            print(f'SKIPPING image which consists of {img.dims.m} unmerged tiles: "{img.name}"')
            # This is a set of unmerged tiles. Just skip
            self.num_images_skipped += 1
            return

        f_path = self.generate_filepath(img.name)

        if Path(f_path).is_file() and not self.conversion_options.overwrite_existing:
            # File already exists. Now check timestamp
            ts = os.path.getmtime(f_path)

            if ts > self.lif_modified_time:
                print(f'SKIPPING image already converted: "{img.name}"')  # "{os.path.basename(f_path)}"')
                self.num_images_skipped += 1
                return

        print(f'Processing image: "{img.name}"')

        # Determine whether this is a z-stack
        z_depth = img.dims.z

        xml_chans = xml_metadata['metadata_xml'].findall("./Data/Image/ImageDescription/Channels/ChannelDescription")
        n_chan = len(xml_chans)
        xml_scales = xml_metadata['metadata_xml'].findall("./Data/Image/Attachment/ChannelScalingInfo")

        bit_depth = img.bit_depth

        # bit_depth is a list with one element per channel.
        if type(bit_depth) is tuple:
            bit_depth = bit_depth[0]

        if bit_depth == 16:
            pixel_type_string = "uint16"
            max_val = 65535
            # Must use unsigned ints, otherwise values above 50% will become negative and truncated to black.
            pixel_type = np.uint16
        elif bit_depth == 8:
            pixel_type_string = "uint8"
            max_val = 255
            pixel_type = np.uint8
        else:
            pixel_type_string = "uint16"
            max_val = 65535
            pixel_type = np.uint16

        img_green = None
        img_red = None
        img_blue = None
        img_cyan = None
        img_magenta = None
        img_yellow = None
        img_grey = None

        make_cyan_file = False
        make_magenta_file = False
        make_yellow_file = False

        print(f'      Image size is: width {img.dims.x} x height {img.dims.y}')
        print(f'      Found {n_chan} color channels, bit depth is {bit_depth}')

        if self.conversion_options.convert_format == self.Options.Format.jpg:
            if img.dims.x > 65535 or img.dims.y > 65535:
                print(f'        JPG only supports up to 65,535 x 65,535 pixels, skipping')
                return

        d = None

        separate_CMY = self.conversion_options.color_format == self.Options.ColorOptions.RGB_CMY
        separate_ALL = self.conversion_options.color_format == self.Options.ColorOptions.all_separate
        do_max_project = self.conversion_options.zstack_format == self.Options.ZStackOptions.max_project

        if do_max_project:
            export_z_loop_count = 1
        else:
            export_z_loop_count = z_depth

        for z in range(export_z_loop_count):

            for m in range(n_chan):

                if self.stopFlag.check():
                    raise self.UserCanceled

                color = xml_chans[m].attrib["LUTName"]
                print(f'        Generating image for channel {color}: ')
                if (n_chan - m) <= len(xml_scales):
                    xml_scale = xml_scales[-(n_chan - m)]
                    white_value = xml_scale.attrib["WhiteValue"]
                    black_value = xml_scale.attrib["BlackValue"]
                else:
                    # Some images, like snapshots, may not have ChannelScalingInfo. Just use defaults.
                    white_value = 1
                    black_value = 0

                white_value = float(white_value)
                black_value = float(black_value)

                start = time.time()

                # Access a specific item, after possibly calculating z-stack
                if z_depth > 1 and do_max_project:
                    print(f'        Found z-stack of depth {z_depth}, will scan all images and select brightest value for '
                          f'each pixel (which may come from different z-planes).')

                    ar = None

                    for k in range(z_depth):
                        # Keep GUI responsive and check for user interruption
                        if self.stopFlag.check():
                            raise self.UserCanceled
                        print('.', end="")
                        f = img.get_frame(z=k, t=0, c=m)
                        if k == 0:
                            ar = np.array(f)
                        else:
                            ar = np.maximum(ar, np.array(f))
                    print()
                else:
                    f = img.get_frame(z=z, t=0, c=m)
                    ar = np.array(f)

                # Iterate over different items
                # frame_list   = [i for i in img_0.get_iter_t(c=0, z=0)]
                # z_list       = [i for i in img_0.get_iter_z(t=0, c=0)]
                # channel_list = [i for i in img_0.get_iter_c(t=0, z=0)]

                d = ar.shape

                scale = 1.0 / (white_value - black_value)

                offset1 = round(black_value * max_val)  # Using round() because that seems to match LASX

                # Bin several rows together to speed up processing. This is more advantageous
                # if rows are small. As they get bigger, the advantage diminishes, and may even
                # reverse for unknown reasons (memory limit?)
                chunk_h = img.dims.x  # * bit_depth / 8
                if chunk_h < 4000:
                    chunk_v = 32
                elif chunk_h < 8000:
                    chunk_v = 16
                elif chunk_h < 16000:
                    chunk_v = 8
                elif chunk_h < 32000:
                    chunk_v = 4
                else:
                    chunk_v = 2

                row = 0

                # Rescale image intensity with respect to black_level and white_level.
                while row < d[0]:
                    if row + chunk_v > d[0]:
                        # Final chunk may be smaller than the previous ones.
                        chunk_v = d[0] - row
                    # Convert one chunk to float
                    one_row = ar[row:row + chunk_v, ].astype(float)
                    one_row = (one_row - offset1) * scale
                    # Truncate underflow and overflow values.
                    one_row[one_row < 0] = 0
                    one_row[one_row > max_val] = max_val
                    # Demote back to original data type and rewrite
                    ar[row:row + chunk_v, ] = one_row.astype(pixel_type)
                    row += chunk_v

                end = time.time()
                print(f'          Completed in {end - start:.1f} seconds.')

                if separate_ALL:
                    img_zeros = np.zeros(d, dtype=pixel_type_string)
                    if color == "Green":
                        self.write_file(np.dstack((img_zeros, ar, img_zeros)), img.name, bit_depth, suffix="green")
                    elif color == "Red":
                        self.write_file(np.dstack((img_zeros, img_zeros, ar)), img.name, bit_depth, suffix="red")
                    elif color == "Blue":
                        self.write_file(np.dstack((ar, img_zeros, img_zeros)), img.name, bit_depth, suffix="blue")
                    elif color == "Cyan":
                        self.write_file(np.dstack((ar, ar, img_zeros)), img.name, bit_depth, suffix="cyan")
                    elif color == "Magenta":
                        self.write_file(np.dstack((ar, img_zeros, ar)), img.name, bit_depth, suffix="magenta")
                    elif color == "Yellow":
                        self.write_file(np.dstack((img_zeros, ar, ar)), img.name, bit_depth, suffix="yellow")

                    continue

                if color == "Green":
                    img_green = ar
                elif color == "Red":
                    img_red = ar
                elif color == "Blue":
                    img_blue = ar
                elif color == "Cyan":
                    img_cyan = ar
                elif color == "Magenta":
                    img_magenta = ar
                elif color == "Yellow":
                    img_yellow = ar
                elif color == "Gray" or color == "Grey":
                    img_grey = ar
                else:
                    print(f'        Unrecognized color {color}, will skip this channel')
                    continue

            if not separate_ALL:
                # Write merged file
                if self.stopFlag.check():
                    raise self.UserCanceled

                if img_magenta is not None:
                    if img_red is None and img_blue is None:
                        # Neither red nor blue channels already exist, so just write normally
                        img_red = img_magenta
                        img_blue = img_magenta
                    elif img_red is None and img_blue is not None:
                        # If blue already exists, but red is empty, then just put this into red channel
                        img_red = img_magenta
                    elif img_blue is None and img_red is not None:
                        # If red already exists, and blue is empty, then put into blue
                        img_blue = img_magenta
                    else:
                        if separate_CMY:
                            make_magenta_file = True
                        else:
                            # Both red and blue exist ... just bite the bullet and add it to the existing colors,
                            # at reduced intensity
                            im = img_magenta.astype(np.uint32) >> 1  # divide by two
                            img_red = img_red.astype(np.uint32) + im
                            img_blue = img_blue.astype(np.uint32) + im

                            img_red[img_red > max_val] = max_val
                            img_blue[img_blue > max_val] = max_val

                            # Have to "demote" type back down to original 8 or 16-bit.
                            img_red = img_red.astype(pixel_type)
                            img_blue = img_blue.astype(pixel_type)

                if img_cyan is not None:
                    if img_green is None and img_blue is None:
                        # Neither green nor blue exists, so write normally
                        img_green = img_cyan
                        img_blue = img_cyan
                    elif img_blue is None:
                        # Green channel is occupied but not blue. Put cyan channel in blue only
                        print('    Converting cyan to blue, to avoid overlap with green channel')
                        img_blue = img_cyan
                    elif img_green is None:
                        # Blue channel is occupied but not green. Put cyan channel in green only
                        print('    Converting cyan to green, to avoid overlap with blue')
                        img_blue = img_cyan
                    else:
                        # Have both blue and cyan channels. Merge cyan into main image, at reduced brightness
                        if separate_CMY:
                            make_cyan_file = True
                        else:
                            # If we have both blue and cyan, then merge it into green and blue channels.
                            # Note: we have to temporarily promote datatype to 32-bit, or else numbers will overflow,
                            # and then demote back to original datatype.
                            im = img_cyan.astype(np.uint32) >> 1  # divide by two
                            img_green = img_green.astype(np.uint32) + im
                            img_blue = img_blue.astype(np.uint32) + im

                            img_green[img_green > max_val] = max_val
                            img_blue[img_blue > max_val] = max_val

                            # Have to "demote" type back down to original 8 or 16-bit.
                            img_green = img_green.astype(pixel_type)
                            img_blue = img_blue.astype(pixel_type)

                if img_grey is not None:
                    # Neither red nor green channel exists, just write normally.
                    img_red = img_grey
                    img_green = img_grey
                    img_blue = img_grey

                if img_yellow is not None:
                    if img_red is None and img_green is None:
                        # Neither red nor green channel exists, just write normally.
                        img_red = img_yellow
                        img_green = img_yellow
                    elif img_red is None:
                        print('    Converting yellow to red, to avoid overlap with green')
                        img_red = img_yellow
                    elif img_green is None:
                        print('    Converting yellow to green, to avoid overlap with red')
                        img_green = img_yellow
                    else:
                        if separate_CMY:
                            make_yellow_file = True
                        else:
                            # Have both red and green.
                            im = img_yellow.astype(np.uint32) >> 1  # divide by two
                            img_green = img_green.astype(np.uint32) + im
                            img_red = img_red.astype(np.uint32) + im

                            img_green[img_green > max_val] = max_val
                            img_red[img_red > max_val] = max_val

                            # Have to "demote" type back down to original 8 or 16-bit.
                            img_green = img_green.astype(pixel_type)
                            img_red = img_red.astype(pixel_type)

                if img_red is None and img_green is None and img_blue is None:
                    print('      No colors are present, will not write file')
                    return

                if img_red is None:
                    img_red = np.zeros(d, dtype=pixel_type_string)
                if img_green is None:
                    img_green = np.zeros(d, dtype=pixel_type_string)
                if img_blue is None:
                    img_blue = np.zeros(d, dtype=pixel_type_string)

                if self.stopFlag.check():
                    raise self.UserCanceled

                # imwrite requires BGR order, backwards from usual RGB
                merged = np.dstack((img_blue, img_green, img_red))

                if self.conversion_options.rotate180:
                    merged = np.flip(merged, (0, 1))

                if export_z_loop_count > 1:
                    # Append z layer number to filename
                    img_name = img.name + "_Z" + str(z+1)
                else:
                    img_name = img.name

                self.write_file(merged, img_name, bit_depth)

                if separate_CMY:
                    img_zeros = np.zeros(d, dtype=pixel_type_string)
                    if make_cyan_file:
                        self.write_file(np.dstack((img_cyan, img_cyan, img_zeros)), img.name, bit_depth, suffix="cyan")
                    if make_magenta_file:
                        self.write_file(np.dstack((img_magenta, img_zeros, img_magenta)), img.name, bit_depth, suffix="magenta")
                    if make_yellow_file:
                        self.write_file(np.dstack((img_zeros, img_yellow, img_yellow)), img.name, bit_depth, suffix="yellow")

        self.num_images_converted += 1
        return

    def generate_filepath(self, img_name, suffix="RGB"):

        suffix = "_" + suffix

        # Make sure img_name doesn't have any slashes, as that will mess up filename saving
        # Replace any slashes with dashes.
        img_name = img_name.replace('/', '-')
        img_name = img_name.replace('\\', '-')
        # Split path into root and extension
        paths = os.path.splitext(self.file_path)
        if self.conversion_options.convert_format == self.Options.Format.jpg:
            ext = ".jpg"
        elif self.conversion_options.convert_format == self.Options.Format.tiff:
            ext = ".tiff"
        elif self.conversion_options.convert_format == self.Options.Format.xml:
            ext = ".xml"
        else:
            raise f"Unknown format {self.conversion_options.convert_format}"

        return paths[0] + "_" + img_name + suffix + ext

    def write_file(self, merged, img_name, source_bit_depth, suffix="RGB"):

        new_path = self.generate_filepath(img_name, suffix)
        print(f'      Writing {suffix} file: "' + os.path.basename(new_path) + '"')
        start = time.time()

        if self.conversion_options.convert_format == self.Options.Format.jpg:
            if source_bit_depth == 16:
                # JPG only supports 8-bit depth, so divide by 256 using
                # memory-efficient method
                chunk_v = 4
                row = 0
                d = merged.shape
                while row < d[0]:
                    if row + chunk_v > d[0]:
                        # Final chunk may be smaller than the previous ones.
                        chunk_v = d[0] - row
                    # Convert one chunk to float
                    one_row = merged[row:row + chunk_v, ].astype(float)
                    one_row = one_row / 256
                    # Truncate underflow and overflow values.
                    one_row[one_row > 255] = 255
                    # Demote back to original data type and rewrite
                    merged[row:row + chunk_v, ] = one_row.astype(np.uint16)
                    row += chunk_v
            elif source_bit_depth == 12:
                # JPG only supports 8-bit depth, so divide by 16 using
                # memory-efficient method
                chunk_v = 4
                row = 0
                d = merged.shape
                num_rows = d[0]
                while row < num_rows:
                    if row + chunk_v > num_rows:
                        # Final chunk may be smaller than the previous ones.
                        chunk_v = num_rows - row
                    # Convert one chunk to float
                    one_row = merged[row:row + chunk_v, ].astype(float)
                    one_row = one_row / 16
                    # Truncate underflow and overflow values.
                    one_row[one_row > 255] = 255
                    # Demote back to original data type and rewrite
                    merged[row:row + chunk_v, ] = one_row.astype(np.uint16)
                    row += chunk_v

            cv2.imwrite(new_path, merged, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        elif self.conversion_options.convert_format == self.Options.Format.tiff:

            if source_bit_depth == 12:
                # TIFF supports either 8 or 16 bit depth. 12-bit will need to be multiplied by 16
                # using memory-efficient method
                chunk_v = 4
                row = 0
                d = merged.shape
                num_rows = d[0]
                while row < num_rows:
                    if row + chunk_v > num_rows:
                        # Final chunk may be smaller than the previous ones.
                        chunk_v = num_rows - row
                    # Convert one chunk to float
                    one_row = merged[row:row + chunk_v, ].astype(float)
                    one_row = one_row * 16
                    # Truncate underflow and overflow values.
                    one_row[one_row > 65535] = 65535
                    # Demote back to original data type and rewrite
                    merged[row:row + chunk_v, ] = one_row.astype(np.uint16)
                    row += chunk_v

            cv2.imwrite(new_path, merged)

        end = time.time()
        print(f'      Completed in {end - start:.1f} seconds.')

    # Recursively finds metadata for all elements having a "Data/Image" subunit.
    def _recursive_metadata_find(self, tree, return_list=None, path=""):
        """Creates list of images by parsing the XML header recursively"""

        if return_list is None:
            return_list = []

        children = tree.findall("./Children/Element")
        if len(children) < 1:  # Fix for 'first round'
            children = tree.findall("./Element")
        for item in children:
            folder_name = item.attrib["Name"]
            # Grab the .lif filename name on the first execution
            if path == "":
                appended_path = folder_name
            else:
                appended_path = path + "/" + folder_name
            # This finds empty folders
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0

            is_image = (
                    len(item.findall("./Data/Image")) > 0
            )

            if is_image:
                # If additional XML data extraction is needed, add it here.

                data_dict = {
                    "metadata_xml": item
                }

                return_list.append(data_dict)

            # An image can have sub_children, it is not mutually exclusive
            if has_sub_children:
                self._recursive_metadata_find(item, return_list, appended_path)

        return return_list
