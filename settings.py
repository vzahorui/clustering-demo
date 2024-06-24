ACCENT_COLOR = '#1d92b8'
CLUSTERING_ALGORITHMS = ('K-means', 'Agglomerative', 'BIRCH', 'Affinity propagation', 'Mean shift',
                         'Spectral', 'DBSCAN', 'OPTICS', 'HDBSCAN')

DEFAULT_DOT_COLOR = '#31363F'
# the colors below look good if the alpha level is set to 0.6
COLORS = (
    '#E44800',  # orange
    '#256F29',  # emerald
    '#87122B',  # cherry
    '#213BFF',  # vivid blue
    '#E6BB00',  # wheat
    '#8534BF',  # violet
    '#E04644',  # soft red
    '#847AF5',  # light lilac
    '#FFDB66',  # sand
    '#3BBF69',  # light emerald
    '#23DBCF',  # cyan
    '#7A3300',  # brown
    '#D97330',  # bright orange, brick
    '#AB1135',  # beetroot
)
DATA_POINTS_THRESHOLD = 1000

HPARAM_HIGHLIGHT_STYLE = {
        'background': '#1d9fb8',
        'border-radius': '5px',
        'padding': '3px',
}
HPARAMS_OF_CLUSTER_METHODS = {
    'K-means': ('n_clusters',),
    'Agglomerative': ('n_clusters', 'linkage_type'),
    'BIRCH': ('n_clusters', 'cluster_radius', 'branching_factor'),
    'Spectral': ('n_clusters',),
    'Mean shift': ('kernel_bandwidth',),
    'DBSCAN': ('min_samples', 'core_distance'),
    'OPTICS': ('min_samples', 'min_cluster_size'),
    'HDBSCAN': ('min_samples', 'min_cluster_size'),
}
