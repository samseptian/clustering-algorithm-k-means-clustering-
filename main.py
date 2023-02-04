import os
import sys
import datetime

from tools import user_inputs, auto_inputs, dir_create, save_xlsx, summary, info
from preprocessing import Preprocessing
from feature import FeatureSelection
from clustering import kmeans
from plot import pie_plot, chart_plot, compare_plot
from sort import result_sort

# initialization for root directory
ROOT = os.path.dirname(os.path.abspath(__file__))

def prepare_first_run():
    if 'results' not in os.listdir(ROOT):
        os.mkdir('results')

    if 'cache' not in os.listdir(ROOT):
        os.mkdir('cache')
        for file in ['corrected_stem.json', 'stemmed_cache.json', 'manual_stopwords.txt']:
            if file not in os.listdir(os.path.join(ROOT, 'cache')):
                content = '' if file.endswith('.txt') else '{}'
                with open(os.path.join(ROOT, 'cache', file), 'w') as f:
                    f.write(content)

def main(test=False):

    input_method = user_inputs
    if test:
        input_method = auto_inputs
    dataset, options = input_method()

    root = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = dir_create('results', '%s (%s)' % (root, dataset['filename']))

    pie_plot(dataset['categories'], root_dir, '%s Original Cluster' % dataset['filename'], '%s_original_cluster' % dataset['techname'])

    info('pre-processing dataset...', level=0)
    Preprocessing(dataset)

    for metric, metric_label in options['input_metrics']:
        info('metric: %s...' % metric, level=0)
        metric_dir = dir_create(root_dir, metric)

        results = {}
        no_fs_results = {}
        for fs_method, _ in options['input_fs_methods']:
            info('feature selection %s...' % fs_method, level=1)

            for fs_percent, _ in options['input_fs_percents']:
                info(f'feature percent: {fs_percent}%...', level=2)

                FeatureSelection(dataset, fs_method, fs_percent)
                X_weight = dataset['%s_%s' % (fs_method, fs_percent)]['weight']
                X_vocabs = dataset['%s_%s' % (fs_method, fs_percent)]['vocabs']

                for init, _ in options['input_k_means']:
                    info(f'init: {init}...', level=3)

                    for k, _ in options['input_k_ranges']:
                        info(f'k: {k}...', level=4)

                        try:
                            km = no_fs_results['%s_%s' % (init, k)]
                        except KeyError:
                            km = kmeans(X_weight, X_vocabs, k, init, n_try=5, metric=metric)
                            if fs_percent == 100:
                                no_fs_results['%s_%s' % (init, k)] = km

                        for key in km.keys():
                            results['_'.join([fs_method, str(fs_percent), init, str(k), key])] = km[key]

        result_sort(results, options)

        info('plotting...', level=0)
        chart_plot(dataset, results, options, metric_dir)
        compare_plot(dataset, results, options, metric_dir)

        info('saving result...', level=0)
        save_xlsx(dataset, results, options, metric_dir)
        summary(dataset, results, options, metric_dir, metric_label)

        
if __name__ == '__main__':
    prepare_first_run()
    main(test='--test' in sys.argv[1:])
    print('Finished!')
