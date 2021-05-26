#!/usr/bin/python3
# -*- coding: utf-8 -*-
import seaborn
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
seaborn.set_theme(style='darkgrid')


def main():
    fmri = seaborn.load_dataset('fmri')
    seaborn.lineplot(x='timepoint', y='signal', hue='region',
                     style='event', data=fmri)
    plt.show()
    # plt.savefig('./hi.png')
    # print(fmri)


if __name__ == "__main__":
    main()
