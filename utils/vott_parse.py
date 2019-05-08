# cunyuan

import json, os, subprocess, time
import argparse
from numpy import random
import cv2 as cv

if not os.path.exists('output'):
    os.makedirs('output')


def write_xml(bndbox, size, lbl, filename):
    width, height = size
    xmin, ymin, xmax, ymax = bndbox
    with open(filename, 'w') as f:
        f.writelines("<annotation verified=\"yes\">\n")

        f.writelines("<size>\n")

        f.writelines("<width>%s</width>\n" % width)
        f.writelines("<height>%s</height>\n" % height)

        f.writelines("</size>\n")

        f.writelines("<object>\n")
        f.writelines("<name>%s</name>\n" % lbl)
        f.writelines("<bndbox>\n")

        f.writelines("<xmin>%s</xmin>\n" % xmin)
        f.writelines("<ymin>%s</ymin>\n" % ymin)
        f.writelines("<xmax>%s</xmax>\n" % xmax)
        f.writelines("<ymax>%s</ymax>\n" % ymax)

        f.writelines("</bndbox>\n")
        f.writelines("</object>\n")


def cropToROI(img, img_size_y_x, roi, dst_size):
    """
    The function for cropping large frame tpo a relatively smaller area
    around the ROI.
    Only processes 1-channel input
    :param img: input image, or the frame
    :param img_size_y_x: input image size transposed: (y, x)
    :param roi: (x1, y1, x2, y2), loc of the target. only 1 roi supported.
        absolute size. not ratio!
    :param dst_size: size of output
    :return: a smaller frame of size dst_size = (new_w, new_h)
    """
    xmin, ymin, xmax, ymax = roi
    w, h = roi[2] - roi[0], roi[3] - roi[1]
    W_raw, H_raw = img_size_y_x
    W_dst, H_dst = dst_size
    x_lmargin, y_lmargin, x_rmargin, y_rmargin = xmin, ymin, W_raw - xmax, H_raw - ymax

    if x_lmargin > x_rmargin:
        if x_rmargin != 0:
            dx_r = random.randint(0, min(x_rmargin, abs(W_dst - w)))
        else:
            dx_r = 0
        xmax_dst = xmax + dx_r
        xmin_dst = xmax_dst - W_dst

    else:
        if x_lmargin != 0:
            dx_l = random.randint(0, min(x_lmargin, abs(W_dst - w)))
        else:
            dx_l = 0
        xmin_dst = xmin - dx_l
        xmax_dst = xmin_dst + W_dst

    if y_lmargin > y_rmargin:
        if y_rmargin != 0:
            dy_r = random.randint(0, min(y_rmargin, abs(H_dst - h)))
        else:
            dy_r = 0
        ymax_dst = ymax + dy_r
        ymin_dst = ymax_dst - H_dst
    else:
        if y_lmargin != 0:
            dy_l = random.randint(0, min(y_lmargin, abs(H_dst - h)))
        else:
            dy_l = 0
        ymin_dst = ymin - dy_l
        ymax_dst = ymin_dst + H_dst

    xmin_dst, ymin_dst, xmax_dst, ymax_dst = \
        int(xmin_dst), int(ymin_dst), int(xmax_dst), int(ymax_dst)
    dst_img = img[:, ymin_dst:ymax_dst, xmin_dst:xmax_dst]
    print(ymin_dst, ymax_dst, xmin_dst, xmax_dst)
    new_loc = [xmin - xmin_dst, ymin - ymin_dst, xmax - xmin_dst, ymax - ymin_dst]
    dst_coord = [xmin_dst, ymin_dst, xmax_dst, ymax_dst]
    return (dst_img, dst_coord, new_loc)


def labelj2xml():
    with open('img.json', 'r') as f_vott_json:
        data_dict = json.load(f_vott_json)
        idx = 0
        for frame_name in data_dict["frames"]:
            if data_dict["frames"]["%s" % frame_name] != []:
                each_frame = cv.imread("./img/%s" % frame_name)
                cv.imwrite("./output/%s" % frame_name, each_frame)
                xmin = (data_dict["frames"]["%s" % frame_name][0]["x1"])
                ymin = (data_dict["frames"]["%s" % frame_name][0]["y1"])
                xmax = (data_dict["frames"]["%s" % frame_name][0]["x2"])
                ymax = (data_dict["frames"]["%s" % frame_name][0]["y2"])
                lbl = (data_dict["frames"]["%s" % frame_name][0]["tags"][0])
                width = (data_dict["frames"]["%s" % frame_name][0]["width"])
                height = (data_dict["frames"]["%s" % frame_name][0]["height"])
                write_xml((xmin, ymin, xmax, ymax), (width, height), lbl, "./output/%s.xml" % frame_name[0:-4])
                idx += 1
            else:
                pass
                # print("==\n==\n"*5)
        f_vott_json.close()
    print("%s labeled images processed." % idx)


def labelj2lst(data_path, IF_CROP=False, dst_size=None):
    lst = []
    if IF_CROP:
        if not os.path.exists(data_path + "cropped"):
            os.makedirs(data_path + "cropped")
        fl = open(data_path + 'cropped/train.lst', 'w')
    else:
        fl = open(data_path + 'train.lst', 'w')

    with open('img.json', 'r') as f_vott_json:
        data_dict = json.load(f_vott_json)
        idx = 0
        for frame_name in data_dict["frames"]:
            if data_dict["frames"]["%s" % frame_name] != []:
                each_frame = cv.imread("./img/%s" % frame_name)
                xmin = (data_dict["frames"]["%s" % frame_name][0]["x1"])
                ymin = (data_dict["frames"]["%s" % frame_name][0]["y1"])
                xmax = (data_dict["frames"]["%s" % frame_name][0]["x2"])
                ymax = (data_dict["frames"]["%s" % frame_name][0]["y2"])
                lbl = (data_dict["frames"]["%s" % frame_name][0]["tags"][0])
                width = (data_dict["frames"]["%s" % frame_name][0]["width"])
                height = (data_dict["frames"]["%s" % frame_name][0]["height"])

                if IF_CROP:
                    img_name = data_path + "cropped/" + frame_name
                    each_frame, _, new_roi = cropToROI(each_frame, (each_frame.shape[1], each_frame.shape[0]),
                                                       (xmin, ymin, xmax, ymax),
                                                       dst_size)
                    xmin, ymin, xmax, ymax = new_roi
                    width, height = dst_size

                    if not os.path.exists('output/cropped/'):
                        os.makedirs('output/cropped/')

                    cv.imwrite(data_path + "cropped/" + "%s" % frame_name, each_frame)
                else:
                    img_name = data_path + frame_name
                    cv.imwrite(data_path + "%s" % frame_name, each_frame)

                lst_tmp = str(idx) + '\t4' + '\t5' + '\t' + str(width) + '\t' + str(height) + '\t' \
                          + str('1') + '\t' \
                          + str(xmin / width) + '\t' + str(ymin / height) + '\t' \
                          + str(xmax / width) + '\t' + str(ymax / height) + '\t' \
                          + img_name + '\n'
                # print(lst_tmp)
                fl.write(lst_tmp)

                idx += 1
            else:
                pass
                # print("==\n==\n"*5)
            # if idx==1000: break
    f_vott_json.close()
    fl.close()

    print("%s labeled images processed." % idx)


def gen_negapos(data_path):
    if not os.path.exists(data_path + "np"):
        os.makedirs(data_path + "np")
        os.makedirs(data_path + "np/pos")
        os.makedirs(data_path + "np/nega")

    with open('img.json', 'r') as f_vott_json:
        data_dict = json.load(f_vott_json)
        idx = 0
        for frame_name in data_dict["frames"]:
            if data_dict["frames"]["%s" % frame_name] != []:
                each_frame = cv.imread("./img/%s" % frame_name)
                xmin = (data_dict["frames"]["%s" % frame_name][0]["x1"])
                ymin = (data_dict["frames"]["%s" % frame_name][0]["y1"])
                xmax = (data_dict["frames"]["%s" % frame_name][0]["x2"])
                ymax = (data_dict["frames"]["%s" % frame_name][0]["y2"])
            pos_patch = each_frame[xmin:xmax, ymin:ymax]
            nega_patch = each_frame.copy()
            nega_patch[xmin:xmax, ymin:ymax] = 0
            cv.imwrite(data_path + "np/pos/p%d.jpg" % idx, pos_patch)
            cv.imwrite(data_path + "np/nega/n%d.jpg" % idx, nega_patch)
            idx += 1


def test():
    labelj2lst("output/", True, (640, 480))
    os.system("python3 im2rec.py output/cropped/train.lst ./ --pack-label")
