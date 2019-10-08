# -*- coding: utf-8 -*-


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['image.cmap'] = 'inferno'


def plot_bboxs(bbox_list, ax, args={"edgecolor":'white', "linewidth":2, "alpha": 0.5}):
    for bb in bbox_list:
        minr, minc, maxr, maxc = bb
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, **args)
        ax.add_patch(rect)

def plot_texts(text_list, cordinate_list, ax, shift=[0, 0],
               fontdict={'color':  'white',
                            'weight': 'normal','size': 10}):
    for text, cordinate in zip(text_list, cordinate_list):
        plt.text(x=cordinate[1]+shift[0], y=cordinate[0]+shift[1], s=str(text),
                 fontdict=fontdict)

def plot_circles(circle_list, ax, args={"color": "white", "linewidth": 1, "alpha": 0.5}):

    for blob in circle_list:
        y, x, r = blob
        c = plt.Circle((x, y), r, **args, fill=False)
        ax.add_patch(c)

def easy_sub_plot(image_list, col_num=3, title_list=None, args={}):

    for i, image in enumerate(image_list):
        k = (i%col_num + 1)
        plt.subplot(1, col_num, k)
        plt.imshow(image, **args)
        plt.title(title_list[i])
        if (k == col_num) | (i == len(image_list)):
            plt.show()
