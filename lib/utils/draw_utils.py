import matplotlib.pyplot as plt
import seaborn as sns


def save_adj(adj, path, name, title='adj', annot=False):
    # 这里使用seaborn库的heatmap函数来绘制热图，可以根据数据的值来着色
    plt.figure(figsize=(10, 8))  # 可选的，设置图形的大小

    # cmap参数可以用于指定颜色映射，这里选择红色越深的颜色映射
    sns.heatmap(adj, cmap="Reds", annot=annot)  # annot=False表示不显示数值标签，如果需要显示可以设置为True

    plt.title(title)  # 设置图形标题
    plt.savefig(f'{path}adj_{name}.png')
    plt.close()
