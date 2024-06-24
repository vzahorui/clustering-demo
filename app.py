import numpy as np
import panel as pn
import param

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import PointDrawTool
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import (KMeans, AgglomerativeClustering, MeanShift, SpectralClustering, DBSCAN, OPTICS, Birch,
                             HDBSCAN, AffinityPropagation)

from settings import (ACCENT_COLOR, CLUSTERING_ALGORITHMS, DEFAULT_DOT_COLOR, COLORS, DATA_POINTS_THRESHOLD,
                      HPARAM_HIGHLIGHT_STYLE, HPARAMS_OF_CLUSTER_METHODS)


class DataMaker(pn.viewable.Viewer):
    """All parameters which define the dataset,  as well as the added behavior caused by the usage of Bokeh tools."""
    cluster_type: list[str] = param.Selector(objects=('Blob', 'Moons', 'Circle'), default='Blob')
    n_samples_in_cluster: int = param.Integer(default=40, bounds=(0, 1000))
    cluster_x_center: float = param.Number(default=0)
    cluster_y_center: float = param.Number(default=0)
    cluster_std: float = param.Number(default=1)
    cluster_shear_angle = param.Number(default=0, bounds=(-360, 360))
    random_state: int = param.Integer(default=42, bounds=(0, None))
    add_cluster: bool = param.Event(
        doc="The event is triggered when the user clicks 'Add cluster' button."
    )
    remove_selected: bool = param.Event(
        doc="The event is triggered when the user clicks 'Remove selected' button."
    )
    revert: bool = param.Event(
        doc="The event is triggered when the user clicks 'Revert' button."
    )

    bokeh_chart_source = ColumnDataSource(default_values={'color': DEFAULT_DOT_COLOR})
    bokeh_figure = figure(title="Dataset for clustering", width=810, height=810,
                          tools="pan,wheel_zoom,box_zoom,reset,lasso_select,tap")

    def __init__(self, **params):
        super().__init__(**params)  # to correctly infer the signature by the editor

        point_renderer = self.bokeh_figure.scatter(
            'x', 'y', source=self.bokeh_chart_source,
            fill_color='color',
            marker='circle', size=15, fill_alpha=0.6, line_color='white'
        )
        draw_tool = PointDrawTool(renderers=[point_renderer])
        self.bokeh_figure.add_tools(draw_tool)
        self.bokeh_figure.toolbar.active_tap = draw_tool

    @param.depends("revert", watch=True, on_init=True)
    def _make_initial_blobs(self):
        """Create two blobs with pseudo random distribution
        and assign their values to the internal data holders.

        This method executes at the very beginning in order not to start from a clean slate."""
        rng = np.random.default_rng(self.random_state)
        blob_init, _ = make_blobs(n_samples=80, n_features=2,
                                  centers=[(rng.uniform(-4, 1), rng.uniform(-4, 1)),
                                           (rng.uniform(-1, 4), rng.uniform(-1, 4))],
                                  random_state=self.random_state, cluster_std=1)
        # here I used dictionary of lists instead of convenient pandas DataFrame because
        # PointDrawTool actions do not render immediately if one of the source columns is a varchar,
        # and in my case it's color
        source_data = {
            'x': blob_init.T[0].tolist(),
            'y': blob_init.T[1].tolist(),
            'color': np.full(blob_init.shape[0], DEFAULT_DOT_COLOR).tolist()
        }
        self.bokeh_chart_source.data = source_data

    @param.depends("add_cluster", watch=True, on_init=False)
    def _add_cluster(self):
        if len(self.bokeh_chart_source.data['x']) > DATA_POINTS_THRESHOLD:
            return  # do not add more points if the view is too cluttered
        if self.cluster_type == 'Blob':
            new_data, _ = make_blobs(n_samples=self.n_samples_in_cluster, n_features=2,
                                     centers=[(self.cluster_x_center, self.cluster_y_center)],
                                     random_state=self.random_state, cluster_std=self.cluster_std)
        elif self.cluster_type in ('Moons', 'Circle'):
            if self.cluster_type == 'Moons':
                generative_function = make_moons
            else:
                generative_function = make_circles
            new_data, _ = generative_function(n_samples=self.n_samples_in_cluster, noise=self.cluster_std / 10,
                                              random_state=self.random_state)
            addition_matrix = np.zeros((new_data.shape[0], new_data.shape[1]))
            addition_matrix[:, 0] = self.cluster_x_center
            addition_matrix[:, 1] = self.cluster_y_center
            new_data = new_data + addition_matrix
        else:
            return  # unknown cluster type

        # shearing
        radian_angle = np.radians(self.cluster_shear_angle)
        if self.cluster_shear_angle == 0:
            m = 0
        else:
            m = 1/np.tan(radian_angle)  # cotangent
        sheared_data = (np.array([[1, m],
                                 [0, 1]]) @ new_data.T).T
        mean_shift_x = np.mean(sheared_data[:, 0] - new_data[:, 0])
        sheared_data[:, 0] = sheared_data[:, 0] - mean_shift_x  # centering data
        new_data = sheared_data.copy()

        source_data = {
            'x': self.bokeh_chart_source.data['x'] + new_data.T[0].tolist(),
            'y': self.bokeh_chart_source.data['y'] + new_data.T[1].tolist(),
            'color': self.bokeh_chart_source.data['color'] + np.full(new_data.shape[0], DEFAULT_DOT_COLOR).tolist()
        }
        self.bokeh_chart_source.data = source_data

    @param.depends("remove_selected", watch=True, on_init=False)
    def _remove_selected(self):
        """Remove points from the dataset selected with Bokeh PolySelectTool and TapTool."""
        index_set = self.bokeh_chart_source.selected.indices
        for key in self.bokeh_chart_source.data:
            self.bokeh_chart_source.data[key] = [
                item for i, item in enumerate(self.bokeh_chart_source.data[key]) if i not in index_set
            ]
        self.bokeh_chart_source.selected.indices = []

    def __panel__(self):

        cluster_addition_text = pn.pane.Markdown("""
                # Dataset configurations
                Use the options below to add new clusters to the plot.
                """)
        cluster_type_widget = pn.widgets.Select.from_param(
            self.param.cluster_type, name='Cluster type', sizing_mode="stretch_width"
        )
        n_samples_widget = pn.widgets.IntInput.from_param(
            self.param.n_samples_in_cluster, name='Number of samples', sizing_mode="stretch_width"
        )
        x_center_widget = pn.widgets.FloatInput.from_param(
            self.param.cluster_x_center, name='X-axis center', sizing_mode="stretch_width"
        )
        y_center_widget = pn.widgets.FloatInput.from_param(
            self.param.cluster_y_center, name='Y-axis center', sizing_mode="stretch_width"
        )
        cluster_std_widget = pn.widgets.FloatInput.from_param(
            self.param.cluster_std, name='Standard deviation', sizing_mode="stretch_width"
        )
        cluster_shear_angle_widget = pn.widgets.FloatInput.from_param(
            self.param.cluster_shear_angle, name='Shearing angle (along x-axis)', sizing_mode="stretch_width"
        )
        random_state = pn.widgets.IntInput.from_param(
            self.param.random_state, name="Random seed", sizing_mode="stretch_width"
        )
        point_addition_text = pn.pane.Markdown('''
                Individual points can be added with Point Draw Tool of the chart.
                Using this tool you can also drag the existing points.
                ''')
        point_removal_text = pn.pane.Markdown('In order to remove points, select them with either Lasso Select or Tap '
                                              'tools of the chart, and then click the button below.')
        add_cluster_button = pn.widgets.Button.from_param(
            self.param.add_cluster, button_type='primary',
            description='Add cluster based on the given parameters',
        )
        remove_selected_button = pn.widgets.Button.from_param(
            self.param.remove_selected, button_type='primary',
            description='Remove selected points',
        )
        revert_button = pn.widgets.Button.from_param(
            self.param.revert, button_type='primary',
            description='Revert to initial state',
            icon='refresh'
        )

        dataset_configs_layout = pn.Column(
            cluster_addition_text,
            pn.Row(cluster_type_widget, n_samples_widget),
            pn.Row(x_center_widget, y_center_widget),
            pn.Row(cluster_std_widget, cluster_shear_angle_widget),
            random_state,
            add_cluster_button,
            point_addition_text,
            point_removal_text,
            remove_selected_button,
            revert_button
        )

        return dataset_configs_layout


class ClusterParams(pn.viewable.Viewer):
    """Selection of the clustering algorithm and its hyperparameters.
    The hyperparameters which are valid for a certain clustering algorithm are highlighted.
    """

    data_store = param.ClassSelector(class_=DataMaker)

    clustering_method: list[str] = param.Selector(objects=CLUSTERING_ALGORITHMS, default='K-means')
    # the parameter below is applicable to K-means, Agglomerative, Spectral, BIRCH
    n_clusters: int = param.Integer(default=2, bounds=(0, 50))
    # the parameter below is applicable to Agglomerative clustering
    linkage_type: list[str] = param.Selector(objects=('single', 'ward', 'complete', 'average'), default='ward')
    # the parameters below are applicable to BIRCH
    cluster_radius: float = param.Number(default=0.5)
    branching_factor: int = param.Integer(default=50)
    # the parameter below is applicable to MeanShift
    kernel_bandwidth: float = param.Number(default=1.2)
    # the parameter below is applicable to DBSCAN
    core_distance: float = param.Number(default=0.8)
    # the parameter below is applicable to DBSCAN, OPTICS, HDBSCAN
    min_samples: int = param.Integer(default=10, bounds=(0, None))
    # the parameter below is applicable to HDBSCAN, OPTICS
    min_cluster_size: int = param.Integer(default=10, bounds=(0, None))

    perform_clustering: bool = param.Event(
        doc="The event is triggered when the user clicks 'Perform clustering' button."
    )

    def __init__(self, **params):
        super().__init__(**params)  # to correctly infer the signature by the editor

        # initial highlighting option for the hyperparameters
        # all empty at first
        self.init_highlight_configs = {
            a: {} for a in ('n_clusters', 'linkage_type', 'cluster_radius', 'branching_factor', 'kernel_bandwidth',
                            'core_distance', 'min_samples', 'min_cluster_size')
        }
        # set highlighting properties based on the default clustering method
        if self.clustering_method in HPARAMS_OF_CLUSTER_METHODS:
            for h in HPARAMS_OF_CLUSTER_METHODS[self.clustering_method]:
                self.init_highlight_configs[h] = HPARAM_HIGHLIGHT_STYLE

    def __panel__(self):
        clustering_method_text = pn.pane.Markdown("""
        # Clustering algorithm configurations

        See the overview of each clustering method and its parameters \
        [here](https://vzahorui.net/clustering/clustering-overview/).
        """)

        clustering_parameters_header_text = pn.pane.Markdown("""
        ## Hyperparameters

        (relevant to the selected method are highlighted)
        """)

        n_clusters_widget = pn.widgets.IntInput.from_param(
            self.param.n_clusters, name='Number of clusters', sizing_mode="stretch_width",
            description='Applicable to K-means, Agglomerative, Spectral and BIRCH',
            styles=self.init_highlight_configs['n_clusters']
        )

        linkage_type_widget = pn.widgets.Select.from_param(
            self.param.linkage_type, name='Type of linkage', sizing_mode="stretch_width",
            description='Applicable to Agglomerative clustering',
            styles=self.init_highlight_configs['linkage_type']
        )

        cluster_radius_widget = pn.widgets.FloatInput.from_param(
            self.param.cluster_radius, name='Cluster radius', sizing_mode="stretch_width",
            description='Applicable to BIRCH',
            styles=self.init_highlight_configs['cluster_radius']
        )

        branching_factor_widget = pn.widgets.IntInput.from_param(
            self.param.branching_factor, name='Branching factor', sizing_mode="stretch_width",
            description='Applicable to BIRCH',
            styles=self.init_highlight_configs['branching_factor']
        )

        kernel_bandwidth_widget = pn.widgets.FloatInput.from_param(
            self.param.kernel_bandwidth, name='Kernel bandwidth', sizing_mode="stretch_width",
            description='Applicable to Mean Shift',
            styles=self.init_highlight_configs['kernel_bandwidth']
        )

        core_distance_widget = pn.widgets.FloatInput.from_param(
            self.param.core_distance, name='Core distance', sizing_mode="stretch_width",
            description='Applicable to DBSCAN',
            styles=self.init_highlight_configs['core_distance']
        )

        min_samples_widget = pn.widgets.IntInput.from_param(
            self.param.min_samples, name='Samples to a cluster', sizing_mode="stretch_width",
            description='Applicable to DBSCAN, OPTICS and HDBSCAN',
            styles=self.init_highlight_configs['min_samples']
        )

        min_cluster_size_widget = pn.widgets.IntInput.from_param(
            self.param.min_cluster_size, name='Minimum cluster size', sizing_mode="stretch_width",
            description='Applicable to HDBSCAN',
            styles=self.init_highlight_configs['min_cluster_size']
        )

        perform_clustering_button = pn.widgets.Button.from_param(
            self.param.perform_clustering, button_type='primary',
            description='Run clustering algorithm'
        )

        @param.depends(self.param.clustering_method, watch=True)
        def highlight_widget(*_):
            all_widgets = (n_clusters_widget, linkage_type_widget, cluster_radius_widget, branching_factor_widget,
                           kernel_bandwidth_widget, core_distance_widget, min_samples_widget, min_cluster_size_widget)

            widgets_of_cluster_methods = {
                'K-means': (n_clusters_widget,),
                'Agglomerative': (n_clusters_widget, linkage_type_widget),
                'BIRCH': (n_clusters_widget, cluster_radius_widget, branching_factor_widget),
                'Spectral': (n_clusters_widget,),
                'Mean shift': (kernel_bandwidth_widget,),
                'DBSCAN': (min_samples_widget, core_distance_widget),
                'OPTICS': (min_samples_widget, min_cluster_size_widget),
                'HDBSCAN': (min_samples_widget, min_cluster_size_widget),
            }

            if self.clustering_method in widgets_of_cluster_methods:
                for widget in widgets_of_cluster_methods[self.clustering_method]:
                    widget.styles = HPARAM_HIGHLIGHT_STYLE
                not_used_widgets_set = set(all_widgets) - set(widgets_of_cluster_methods[self.clustering_method])
                for widget in not_used_widgets_set:
                    widget.styles = {}
            else:  # if the clustering method does not have any hyperparameters
                for widget in all_widgets:
                    widget.styles = {}

        cluster_params_layout = pn.Column(
            clustering_method_text,
            self.param.clustering_method,
            pn.layout.Divider(),
            clustering_parameters_header_text,
            clustering_parameters_header_text,
            pn.Row(n_clusters_widget, linkage_type_widget),
            pn.Row(cluster_radius_widget, branching_factor_widget),
            pn.Row(core_distance_widget, kernel_bandwidth_widget),
            pn.Row(min_samples_widget, min_cluster_size_widget),
            perform_clustering_button
        )

        return cluster_params_layout


class App(pn.viewable.Viewer):
    """Main app which assembles both DataMaker and ClusterParam.
    It holds the logic for applying clustering algorithms to the data.
    """
    data_maker = param.ClassSelector(class_=DataMaker)
    cluster_params = param.ClassSelector(class_=ClusterParams)

    def __init__(self, **params):
        super().__init__(**params)
        self.perform_clustering = self.cluster_params.clustering_method

    @staticmethod
    def map_colors(data_labels):
        """Map labels to colors."""
        unique_labels = np.unique(data_labels)
        label_color_map = {
            label: DEFAULT_DOT_COLOR if label == -1 else COLORS[i % len(COLORS)]
            for i, label in enumerate(unique_labels)
        }
        mapped_labels = [label_color_map[label] for label in data_labels]
        return mapped_labels

    @param.depends("cluster_params.perform_clustering", watch=True, on_init=False)
    def _perform_clustering(self):
        train_data = np.array([self.data_maker.bokeh_chart_source.data['x'],
                               self.data_maker.bokeh_chart_source.data['y']]).T
        data_labels = np.full(train_data.shape[0], -1)
        if self.cluster_params.clustering_method == 'K-means':
            data_labels = KMeans(n_clusters=self.cluster_params.n_clusters,
                                 random_state=self.data_maker.random_state).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'Agglomerative':
            data_labels = AgglomerativeClustering(n_clusters=self.cluster_params.n_clusters,
                                                  linkage=self.cluster_params.linkage_type).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'BIRCH':
            data_labels = Birch(n_clusters=self.cluster_params.n_clusters,
                                threshold=self.cluster_params.cluster_radius,
                                branching_factor=self.cluster_params.branching_factor).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'Affinity propagation':
            data_labels = AffinityPropagation(random_state=self.data_maker.random_state).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'Mean shift':
            data_labels = MeanShift(bandwidth=self.cluster_params.kernel_bandwidth).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'Spectral':
            data_labels = SpectralClustering(n_clusters=self.cluster_params.n_clusters,
                                             random_state=self.data_maker.random_state).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'DBSCAN':
            data_labels = DBSCAN(eps=self.cluster_params.core_distance,
                                 min_samples=self.cluster_params.min_samples).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'OPTICS':
            data_labels = OPTICS(min_samples=self.cluster_params.min_samples,
                                 min_cluster_size=self.cluster_params.min_cluster_size).fit_predict(train_data)
        elif self.cluster_params.clustering_method == 'HDBSCAN':
            data_labels = HDBSCAN(min_samples=self.cluster_params.min_samples,
                                  min_cluster_size=self.cluster_params.min_cluster_size).fit_predict(train_data)
        self.data_maker.bokeh_chart_source.data['color'] = self.map_colors(data_labels)

    def __panel__(self):
        bokeh_plot_widget = pn.pane.Bokeh(self.data_maker.bokeh_figure, sizing_mode='stretch_width')

        template = pn.template.FastGridTemplate(
            title="Clustering algorithms overview",
            accent=ACCENT_COLOR,
            shadow=False
        )
        template.main[:6, :3] = self.data_maker
        template.main[:6, 3:9] = bokeh_plot_widget
        template.main[:6, 9:] = self.cluster_params
        return template


App(data_maker=DataMaker(), cluster_params=ClusterParams()).servable()
