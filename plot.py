import os
import numpy as np

from tools import dir_create
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from itertools import product as iproduct


EVALUATIONS = [
    ('davies', 'Davies-Bouldin Index'),
    ('silhouette', 'Silhouette Coefficient'),
    ('time', 'Execution Time'),
    ('iter', 'Iterations')
]


def clean(label):
    return label.replace('=', '').lower()

def pca_reduction(X, clusters, dim=2, initcent=False):
    if X.shape[1] > dim:
        pca = PCA(n_components=dim)
        data_xd = pca.fit_transform(X)
    else:
        data_xd = X.copy()

    if initcent is not False:
        cent_xd = data_xd[initcent]
    else:
        set_cluster = sorted(set(clusters))
        k = len(set_cluster)
        cent_xd = np.empty((k, dim), dtype=X.dtype)
        for c in set_cluster:
            cent_xd[c] = np.mean(data_xd[clusters == c], axis=0)
    return data_xd, cent_xd

def plot_plot(x, Y, title, name, xlabel, ylabel, save_path, xticks):
    for label, y in Y:
        plt.plot(x, y, label=label)
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(list(x), xticks)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(save_path, 'plot_%s.png' % name), bbox_inches='tight')
    plt.close()

def pie_plot(cluster, save_path, title, name):
    set_cluster = sorted(set(cluster))
    cluster_number, labels = [], []
    for c in set_cluster:
        c_count = sum([1 if cl == c else 0 for cl in cluster])
        cluster_number.append(c_count)
        try:
            label = 'Cluster %s' % (c + 1)
        except TypeError:
            label = c
        labels.append("%s (%s Hadist)" % (label, c_count))

    explode = np.zeros(len(set_cluster))
    explode[np.argmax(cluster_number)] = 0.1

    plt.pie(cluster_number,
        explode=explode, 
        shadow=True, 
        startangle=90, 
        autopct='%1.1f%%'
    )
    
    plt.title(title, wrap=True)
    plt.legend(labels, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)
    plt.savefig(os.path.join(save_path, 'pie_%s.png' % name), bbox_inches='tight')
    plt.close()

def bar_plot(data, save_path, title, name, xlabel, ylabel, xticks):
    width = 0.25
    n = len(data)
    v = len(data[0][1])
    m = (n + 1) * width

    br = [m * i for i in range(v)]
    T = [0.0 for _ in range(v)]
    for i, (label, Y) in enumerate(data):
        X = [b + (width * i) for b in br]
        T = [t + x for t, x in zip(T, X)]
        plt.bar(X, Y, width=width, label=label)

    T = [t / n for t in T]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(T, xticks)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0)

    plt.title(title, wrap=True)
    plt.savefig(os.path.join(save_path, 'bar_%s.png' % name), bbox_inches='tight')
    plt.close()

def scatter_plot(X, cluster, save_path, title, name, initcent=False):
    if X.shape[1] == 2:
        data_2d = X.copy()
        set_cluster = sorted(set(cluster))
        k = len(set_cluster)
        cent_2d = np.empty((k, 2), dtype=X.dtype)
        for c in set_cluster:
            cent_2d[c] = np.mean(data_2d[cluster == c], axis=0)
    else:
        data_2d, cent_2d = pca_reduction(X, cluster, dim=2, initcent=initcent)

    clus = cluster
    if initcent is not False:
        clus = np.ones(len(cluster))

    for c in sorted(set(clus)):
        X_c = data_2d[clus == c]
        if initcent is False:
            try:
                label = 'Cluster %s' % (c + 1)
            except TypeError:
                label = c
        else:
            label = 'Data'
        plt.scatter(X_c[:, 0], X_c[:, 1], label=label)
    plt.scatter(cent_2d.T[0], cent_2d.T[1], marker="^", c="black", label="Centroids")

    plt.title(title, wrap=True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(save_path, 'scatter_%s.png' % name), bbox_inches='tight')
    plt.close()
    return data_2d

def scatt3d_plot(X, cluster, save_path, title, name, initcent=False):
    if X.shape[1] == 3:
        data_3d = X.copy()
        set_cluster = sorted(set(cluster))
        k = len(set_cluster)
        cent_3d = np.empty((k, 3), dtype=X.dtype)
        for c in set_cluster:
            cent_3d[c] = np.mean(data_3d[cluster == c], axis=0)
    else:
        data_3d, cent_3d = pca_reduction(X, cluster, dim=3, initcent=initcent)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    clus = cluster
    if initcent is not False:
        clus = np.ones(len(cluster))
    
    for c in sorted(set(clus)):
        X_c = data_3d[clus == c]
        if initcent is False:
            try:
                label = 'Cluster %s' % (c + 1)
            except TypeError:
                label = c
        else:
            label = 'Data'
        ax.scatter(X_c[:, 0], X_c[:, 1], X_c[:, 2], label=label)
    ax.scatter(cent_3d[:, 0], cent_3d[:, 1], cent_3d[:, 2], label='Centroids', c='black')

    plt.title(title, wrap=True)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'scatt3d_%s.png' % name))
    plt.close()
    return data_3d


def chart_plot(dataset, results, options, root_dir):
    for fs_method, _ in options['input_fs_methods']:
        fs_method_dir = dir_create(root_dir, fs_method)

        for fs_percent, fs_percent_label in options['input_fs_percents']:
            fs_percent_dir = dir_create(fs_method_dir, fs_percent_label)
            
            for init, init_label in options['input_k_means']:
                init_dir = dir_create(fs_percent_dir, init)

                for k, k_label in options['input_k_ranges']:

                    def from_keys(var):
                        return '_'.join([fs_method, str(fs_percent), init, str(k), var])

                    k_dir = dir_create(init_dir, k_label)

                    cluster = results[from_keys('cluster')]
                    initcent = results[from_keys('init')]

                    name = '%s_%s' % (init, clean(k_label))
                    title = '%s on %s' % (init_label, k_label)
                    title_init = '%s Initial Centroids on %s' % (init_label, k_label)

                    X = dataset['%s_%s' % (fs_method, fs_percent)]['weight'].copy()

                    pie_plot(cluster, k_dir, title, name)
                    X_2d = scatter_plot(X, cluster, k_dir, title, name)
                    scatter_plot(X_2d, cluster, k_dir, title_init, name + '_init', initcent=initcent)
                    X_3d = scatt3d_plot(X, cluster, k_dir, title, name)
                    scatt3d_plot(X_3d, cluster, k_dir, title_init, name + '_init', initcent=initcent)


def compare_plot(dataset, results, options, root_dir):
    input_fs_methods = options['input_fs_methods']
    input_fs_percents = options['input_fs_percents']
    input_k_means = options['input_k_means']
    input_k_ranges = options['input_k_ranges']

    comparison_dir = dir_create(root_dir, 'comparison_plot')
    
    for _iter in range(2):

        if _iter == 0:
            measure = input_fs_percents
            neg_measure = input_k_ranges
            measure_label = 'Percent (%)'
            dir_label = 'percent'
            neg_dir_label = 'k'
            eval_result = "[results['%s_%s_%s_%s_%s' % (fs_method, m, init, r, ev)] for m, _ in measure]"
        else:
            measure = input_k_ranges
            neg_measure = input_fs_percents
            measure_label = 'Number of Clusters (K)'
            dir_label = 'k'
            neg_dir_label = 'percent'
            eval_result = "[results['%s_%s_%s_%s_%s' % (fs_method, r, init, m, ev)] for m, _ in measure]"

        parent_dir = dir_create(comparison_dir, 'variation_%s' % dir_label)
        comparison_dir_1 = dir_create(parent_dir, 'single')
        comparison_dir_2 = dir_create(parent_dir, 'fs')
        comparison_dir_3 = dir_create(parent_dir, 'init')
        comparison_dir_4 = dir_create(parent_dir, neg_dir_label)
        comparison_dir_5 = dir_create(parent_dir, 'fs_and_init')
        comparison_dir_6 = dir_create(parent_dir, 'fs_and_%s' % neg_dir_label)
        comparison_dir_7 = dir_create(parent_dir, 'init_and_%s' % neg_dir_label)

        X = [m for m, _ in measure]
        Xticks = [m for _, m in measure]

        # single plot
        for (fs_method, fs_label), (init, init_label), (r, r_label), (ev, ev_label) in iproduct(input_fs_methods, input_k_means, neg_measure, EVALUATIONS):
            result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
            Y = [('%s %s %s' % (fs_label, init_label, r_label), result)]

            title = 'Variation of %s on %s\n(%s, %s, %s)' % (measure_label, ev_label, fs_label, init_label, r_label)
            name = '%s_%s_%s_%s' % (ev, fs_method, init, clean(r_label))
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_1, Xticks)
            bar_plot(Y, comparison_dir_1, title, name, measure_label, ev_label, Xticks)

        # based feature selection plot
        for (init, init_label), (r, r_label), (ev, ev_label) in iproduct(input_k_means, neg_measure, EVALUATIONS):
            Y = []
            for fs_method, fs_label in input_fs_methods:
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append((fs_label, result))
            title = 'Variation of %s on %s\n(%s, %s)' % (measure_label, ev_label, init_label, r_label)
            name = '%s_%s_%s' % (ev, init, clean(r_label))
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_2, Xticks)
            bar_plot(Y, comparison_dir_2, title, name, measure_label, ev_label, Xticks)

        # based initialization plot
        for (fs_method, fs_label), (r, r_label), (ev, ev_label) in iproduct(input_fs_methods, neg_measure, EVALUATIONS):
            Y = []
            for init, init_label in input_k_means:
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append((init_label, result))
            title = 'Variation of %s on %s\n(%s, %s)' % (measure_label, ev_label, fs_label, r_label)
            name = '%s_%s_%s' % (ev, fs_method, clean(r_label))
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_3, Xticks)
            bar_plot(Y, comparison_dir_3, title, name, measure_label, ev_label, Xticks)

        # based n cluster / fs percent plot
        for (fs_method, fs_label), (init, init_label), (ev, ev_label) in iproduct(input_fs_methods, input_k_means, EVALUATIONS):
            Y = []
            for r, r_label in neg_measure:
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append((r_label, result))
            title = 'Variation of %s on %s\n(%s, %s)' % (measure_label, ev_label, fs_label, init_label)
            name = '%s_%s_%s' % (ev, fs_method, init)
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_4, Xticks)
            bar_plot(Y, comparison_dir_4, title, name, measure_label, ev_label, Xticks)
        
        # based feature selection and initialization plot
        for (r, r_label), (ev, ev_label) in iproduct(neg_measure, EVALUATIONS):
            Y = []
            for (fs_method, fs_label), (init, init_label) in iproduct(input_fs_methods, input_k_means):
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append(('%s, %s' % (fs_label, init_label), result))
            title = 'Variation of %s on %s\n(%s)' % (measure_label, ev_label, r_label)
            name = '%s_%s' % (ev, clean(r_label))
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_5, Xticks)
            bar_plot(Y, comparison_dir_5, title, name, measure_label, ev_label, Xticks)

        # based feature selection and n cluster / fs percent plot
        for (init, init_label), (ev, ev_label) in iproduct(input_k_means, EVALUATIONS):
            Y = []
            for (fs_method, fs_label), (r, r_label) in iproduct(input_fs_methods, neg_measure):
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append(('%s, %s' % (fs_label, r_label), result))
            title = 'Variation of %s on %s\n(%s)' % (measure_label, ev_label, init_label)
            name = '%s_%s' % (ev, init)
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_6, Xticks)
            bar_plot(Y, comparison_dir_6, title, name, measure_label, ev_label, Xticks)

        # based initialization and n cluster / fs percent plot
        for (fs_method, fs_label), (ev, ev_label) in iproduct(input_fs_methods, EVALUATIONS):
            Y = []
            for (init, init_label), (r, r_label) in iproduct(input_k_means, neg_measure):
                result = eval(eval_result, {'results': results, 'fs_method': fs_method, 'r': r, 'init': init, 'measure': measure, 'ev': ev})
                Y.append(('%s, %s' % (init_label, r_label), result))
            title = 'Variation of %s on %s\n(%s)' % (measure_label, ev_label, fs_label)
            name = '%s_%s' % (ev, fs_method)
            plot_plot(X, Y, title, name, measure_label, ev_label, comparison_dir_7, Xticks)
            bar_plot(Y, comparison_dir_7, title, name, measure_label, ev_label, Xticks)
