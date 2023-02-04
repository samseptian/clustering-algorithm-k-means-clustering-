import os
import numpy
import pandas
import xlsxwriter

LABELS = [
    ('davies', 'Davies-Bouldin Index'),
    ('silhouette', 'Silhouette Coefficient'),
    ('time', 'Execution Time'),
    ('iter', 'Iteration'),
    ('init', 'Initial Centroids')
]


def info(message, level):
    char = '>' if level % 2 == 0 else '-'
    space = ' ' * (level * 2)
    print('%s%s %s' % (space, char, message))


def Boolean(string, info):
    print(info)
    while True:
        choose = input('%s [1.Yes/0.No]>> ' % string)
        print()
        if choose in ['0', '1']:
            return bool(choose)
        else:
            print("Input %s is not valid!" % choose)
            continue

def RangeSelection(string, selection):
    print(string.upper())
    for i, (_, value) in enumerate(selection):
        print('%s. %s' % (i + 1, value))
    print("""Example:
- '1'   => will select %s option.
- '1,2' => will select %s and %s options.
- 'all' => will select ALL options.""" % (selection[0][1], selection[0][1], selection[1][1]))

    valid = True
    while True:
        if not valid:
            print("Input '%s' is not valid!" % choose)
        choose = input('Your Input >> ')
        print()
        try:
            if choose == 'all':
                return selection
            elif choose.isnumeric():
                return [selection[int(choose) - 1]]
            return [selection[int(c) - 1] for c in choose.split(',')]
        except Exception:
            valid = False

def RangeInput(string, label, valid_range):

    def get_erange(r):
        erange = []
        for i in r:
            try:
                erange.append(i)
            except IndexError:
                break
        return '[' + ', '.join([str(e) for e in erange[:3]]) + ', ...' + str(erange[-1]) + ']'

    erange1 = get_erange(range(valid_range[0], valid_range[-1])) 
    erange2 = get_erange(range(valid_range[0], valid_range[-1], 5)) 
    
    print(string.upper())
    print("""Example:
- '{0}'       => for single experiment with value = {0}
- '{0}-{1}:1' => for multiple experiment with increment 1. e.g. {2}
- '{0}-{1}:5' => for multiple experiment with increment 5. e.g. {3}""".format(valid_range[0], valid_range[-1], erange1, erange2))

    valid = True
    while True:
        if not valid:
            print("Input '%s' is not valid!" % choose)
        choose = input('Your Input range[%s, %s] >> ' % (min(valid_range), max(valid_range)))
        print()
        try:
            if choose.isnumeric():
                return [(r, '%s%s' % (label, r)) for r in range(int(choose), int(choose) + 1)]
            ranges, increment = choose.split(':')
            range_from, range_to = ranges.split('-')
            range_from, range_to = range_from.strip(), range_to.strip()
            inputted_ranges = range(int(range_from), int(range_to) + 1, int(increment))
            if all(r in valid_range for r in inputted_ranges):
                return [(r, '%s%s' % (label, r)) for r in inputted_ranges]
            else:
                valid = False
        except Exception as err:
            valid = False

def dir_create(path, directory):
    path_to_create = os.path.join(path, directory)
    if directory not in os.listdir(path):
        os.mkdir(path_to_create)
    return path_to_create


def user_inputs():
    input_datasets = RangeSelection(
        string='Select Dataset(s)',
        selection=[(f.replace('.xlsx', ''), f[:f.index('.')].title()) for f in 
            [d for d in os.listdir('datasets') if d.endswith('.xlsx')]])

    dataset = read_dataset(input_datasets)
    
    input_fs_methods = RangeSelection(
        string='Select Feature Selection(s)', 
        selection=[
            ('pca', 'PCA'),
            ('union', 'Modified Union')
        ])

    input_fs_percents = RangeInput(
        string='Input Feature Selection Percentage',
        label='P=',
        valid_range=range(1, 101))

    if 100 not in [p for p, _ in input_fs_percents]:
        include_no_fs = Boolean(
            string='Include it?',
            info="100 is not included in your range!\nThe plots will not show for P=100 if you don't include it."
        )
        if include_no_fs:
            input_fs_percents += [(100, 'P=100')]

    input_k_means = RangeSelection(
        string='Select Initialization',
        selection=[
            ('kmeans', 'K-Means'),
            ('kmeans++', 'K-Means++')
        ])

    input_k_ranges = RangeInput(
        string='Input K Range',
        label='K=',
        valid_range=range(2, 101))

    input_metrics = RangeSelection(
        string='Select Metrics',
        selection=[
            ('euclidean', 'Euclidean Distance'),
            ('cosine', 'Cosine Distance')
        ]
    )

    return dataset, { 
        'input_fs_methods': input_fs_methods, 
        'input_fs_percents': input_fs_percents,
        'input_k_means': input_k_means, 
        'input_k_ranges': input_k_ranges,
        'input_metrics': input_metrics
    }


def auto_inputs():
    input_datasets = [('bukhari', 'Bukhari'), ('muslim', 'Muslim'), ('tirmidzi', 'Tirmidzi')]
    dataset = read_dataset(input_datasets, test=True)
    input_fs_methods = [
        ('pca', 'PCA'),
        ('union', 'Modified Union')
    ]
    input_fs_percents = [(r, 'P=%s' % r) for r in range(80, 101, 5)]
    input_k_means = [('kmeans', 'K-Means'), ('kmeans++', 'K-Means++')]
    input_k_ranges = [(r, 'K=%s' % r) for r in range(2, 6)]
    input_metrics = [('cosine', 'Cosine Distance'), ('euclidean', 'Euclidean Distance')]
    return dataset, { 
        'input_fs_methods': input_fs_methods, 
        'input_fs_percents': input_fs_percents,
        'input_k_means': input_k_means, 
        'input_k_ranges': input_k_ranges,
        'input_metrics': input_metrics
    }


def read_dataset(input_dataset, test=False):
    dataset = {
        'filename': ' + '.join(label for _, label in input_dataset),
        'techname': '_'.join(name for name, _ in input_dataset)
    }

    raws, labels, categories = [], [], []
    for data_file, data_label in input_dataset:
        dataset_path = os.path.join('datasets', data_file + '.xlsx')
        reader = [list(i) for i in pandas.read_excel(dataset_path).values]

        R, L, C = [], [], []
        for doc in reader:
            R.append(doc[-2])
            L.append('%s %s: %s' % (data_label, str(doc[0]), doc[2]))
            C.append(doc[2])

        set_categories = sorted(set(C))
        if test:
            input_set_categories = [(c, c) for c in numpy.random.choice(set_categories, 3, replace=False)]
        else:
            input_set_categories = RangeSelection(
                string='Select Categories for dataset %s' % data_label,
                selection=[(c, c) for c in set_categories])

        indices = []
        for _, category in input_set_categories:
            index_category = [i for i, c in enumerate(C) if c == category]
            if test:
                indices += [i for i in numpy.random.choice(index_category, min([10, len(index_category)]), replace=False)]
            else:
                indices += index_category

        raws += [R[i] for i in indices]
        labels += [L[i] for i in indices]
        categories += [C[i] for i in indices]

    dataset['raws'] = raws
    dataset['labels'] = labels
    dataset['categories'] = categories
    return dataset


def save_xlsx(dataset, results, options, root_dir):

    for fs_method, _ in options['input_fs_methods']:
        
        for fs_percent, fs_percent_label in options['input_fs_percents']:

            data_key = '%s_%s' % (fs_method, fs_percent)

            data = dataset[data_key]['weight']
            vocabulary = dataset[data_key]['vocabs']
            labels = dataset['labels']
            
            N, V = data.shape
            dataset_name = dataset['filename']
            workbook = xlsxwriter.Workbook(os.path.join(root_dir, fs_method, fs_percent_label, 'report.xlsx'))

            data_info = [
                ('DATASET', dataset_name.upper()),
                ('TOTAL RECORDS', N),
                ('TOTAL COLUMNS', V)
            ]
            sheet_info = workbook.add_worksheet('info')

            for row, (k, v) in enumerate(data_info):
                sheet_info.write(row, 0, k)
                sheet_info.write(row, 1, v)

            sheet_pre = workbook.add_worksheet('preprocessing')

            row = 0
            for col, label in enumerate(['No.', 'Text', 'Lower', 'Tokenized', 'Non-Stopwords', 'Stemmed']):
                sheet_pre.write(row, col, label)
            
            row += 1
            for i in range(N):
                sheet_pre.write(row, 0, labels[i])
                sheet_pre.write(row, 1, dataset['raws'][i])
                sheet_pre.write(row, 2, dataset['preprocessing']['lowers'][i])
                sheet_pre.write(row, 3, ', '.join(dataset['preprocessing']['tokenized'][i]))
                sheet_pre.write(row, 4, ', '.join(dataset['preprocessing']['non_stopwords'][i]))
                sheet_pre.write(row, 5, ', '.join(dataset['preprocessing']['stemmed'][i]))
                row += 1

            sheet_tfidf = workbook.add_worksheet('tfidf')
            for col, feature in enumerate(vocabulary):
                sheet_tfidf.write(0, col + 1, feature)

            for row, d in enumerate(data):
                d = [labels[row]] + list(d)
                sheet_tfidf.write_row(row + 1, 0, d)

            for k, k_label in options['input_k_ranges']:
                sheet_name = k_label
                sheet = workbook.add_worksheet(sheet_name)

                row = 0
                for init, init_label in options['input_k_means']:
                    sheet.write(row, 0, init_label)
                    row += 1

                    for key, label in LABELS:
                        sheet.write(row, 0, label)
                        value = results["%s_%s_%s_%s" % (data_key, init, k, key)]
                        if key == 'init':
                            value = ', '.join([str(num + 1) for num in value])
                        sheet.write(row, 1, value)
                        row += 1

                    sheet.write(row, 0, 'Cluster')
                    sheet.write(row, 1, 'No. Dokumen/Ayat')
                    sheet.write(row, 2, 'Jumlah')
                    sheet.write(row, 3, 'Top Features')
                    row += 1

                    key = "%s_%s_cluster" % (init, k)
                    cluster = results["%s_%s_%s_cluster" % (data_key, init, k)]
                    features = results["%s_%s_%s_features" % (data_key, init, k)]

                    for i in range(k):
                        cluster_i = [labels[j] for j, c in enumerate(cluster) if c == i]
                        join_cluster_i = ', '.join(cluster_i)
                        sheet.write(row, 0, i + 1)
                        sheet.write(row, 1, join_cluster_i)
                        sheet.write(row, 2, len(cluster_i))
                        sheet.write(row, 3, features[i])
                        row += 1
                    row += 1
            workbook.close()


def summary(dataset, results, options, root_dir, metric):
    N, V = dataset['weight']['tfidf'].shape

    data_info = [
        ('DATASET', dataset['filename'].upper()),
        ('TOTAL RECORDS', N),
        ('TOTAL COLUMNS', V),
        ('FEATURE SELECTIONS', ', '.join([fs_method_label for _, fs_method_label in options['input_fs_methods']])),
        ('FEATURE PERCENTS', ', '.join([str(fs_percent) for fs_percent, _ in options['input_fs_percents']])),
        ('CENTROID INITIALIZATIONS', ', '.join([init_label for _, init_label in options['input_k_means']])),
        ('NUMBER OF CLUSTERS', ', '.join([str(k) for k, k in options['input_k_ranges']])),
        ('DISTANCE METRIC', metric),
    ]

    workbook = xlsxwriter.Workbook(os.path.join(root_dir, 'summary.xlsx'))

    sheet = workbook.add_worksheet('summary')

    row = 0
    for k, v in data_info:
        sheet.write(row, 0, k)
        sheet.write(row, 1, v)
        row += 1

    row += 1
    for col, label in enumerate(['FS Method', 'FS Percent', 'Initialization', 'Number of Clusters'] + [label for _, label in LABELS]):
        sheet.write(row, col, label)
    
    row += 1
    for fs_method, fs_method_label in options['input_fs_methods']:
        sheet.write(row, 0, fs_method_label)
        
        for fs_percent, fs_percent_label in options['input_fs_percents']:
            sheet.write(row, 1, fs_percent_label)

            for init, init_label in options['input_k_means']:
                sheet.write(row, 2, init_label)

                for k, k_label in options['input_k_ranges']:
                    sheet.write(row, 3, k_label)

                    def from_keys(var):
                        value = results['%s_%s_%s_%s_%s' % (fs_method, fs_percent, init, k, var)]
                        if var != 'init':
                            return value
                        return ', '.join([str(i + 1) for i in value])

                    col = 4
                    for key, label in LABELS:
                        sheet.write(row, col, from_keys(key))
                        col += 1
                    row += 1
    workbook.close()
