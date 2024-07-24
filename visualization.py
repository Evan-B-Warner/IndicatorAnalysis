import pandas as pd
import matplotlib.pyplot as plt

def display_as_df(cols, colnames, head_only=True):
    display_df = pd.DataFrame()
    for i in range(len(cols)):
        display_df[colnames[i]] = cols[i]
    if head_only:
        print(display_df.head())
    else:
        print(display_df)
    return display_df


def make_plot(x, y, lines=None):
    plt.plot(x, y)
    for line in lines:
        plt.plot(line[0], line[1])
    plt.show()
