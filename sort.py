sort_based = 'davies'

def result_sort(results, options):

    for fs_method, _ in options['input_fs_methods']:
        for fs_percent, _ in options['input_fs_percents']:
            for k, _ in options['input_k_ranges']:

                def from_keys(init, ev):
                    return '%s_%s_%s_%s_%s' % (fs_method, fs_percent, init, k, ev)

                if results[from_keys('kmeans', sort_based)] < results[from_keys('kmeans++', sort_based)]:
                    for key in ['davies', 'silhouette', 'features', 'init', 'cluster', 'centroid', 'iter', 'time']:
                        kmkey = from_keys('kmeans', key)
                        kmpkey = from_keys('kmeans++', key)
                        results[kmkey], results[kmpkey] = results[kmpkey], results[kmkey]

            for init, _ in options['input_k_means']:
                times = []
                for k, _ in options['input_k_ranges']:
                    times.append(results['%s_%s_%s_%s_%s' % (fs_method, fs_percent, init, k, 'time')])
                times = sorted(times)

                for i, (k, _) in enumerate(options['input_k_ranges']):
                    results['%s_%s_%s_%s_%s' % (fs_method, fs_percent, init, k, 'time')] = times[i]
