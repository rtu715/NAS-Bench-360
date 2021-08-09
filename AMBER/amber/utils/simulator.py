from __future__ import print_function

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BaseSimulator:
    def __init__(self, n, p, *args, **kwargs):
        """
        Args:
            n:
            p:
            *args:
            **kwargs:
        """
        self.n = n
        self.p = p

    def sample_effect(self):
        pass

    def sample_data(self):
        pass

    def get_ground_truth(self, X):
        pass


class Simulator(BaseSimulator):
    def __init__(self, n, p, beta_a, beta_i, noise_var=1.,
                 discretized=False, *args, **kwargs):
        """

        Args:
            n:
            p:
            beta_a:
            beta_i:
            noise_var:
            discretized:
            *args:
            **kwargs:
        """
        self.n = n
        self.p = p
        self.beta_a = np.array(beta_a).astype('float32')
        self.beta_i = np.array(beta_i).astype('float32')
        self.noise_var = noise_var
        self.discretized = discretized

    def sample_effect(self, drop_a, drop_i):
        """TODO: random sample effect sizes
        :param drop_a: probability of masking of additive
        :param drop_i: prob. for masking of interaction
        """
        self.beta_a = np.random.normal
        self.beta_i = np.random.normal

    def sample_data(self):
        """
        :rtype (X,y): a tuple of X and y
        """
        if self.discretized:
            X = np.array(np.random.randint(low=0, high=3, size=self.n * self.p)).reshape(self.n, self.p).astype(
                'float32')
        else:
            X = np.array(np.random.uniform(low=0, high=3, size=self.n * self.p)).reshape(self.n, self.p).astype(
                'float32')
        X_s = PolynomialFeatures(2, interaction_only=False, include_bias=False).fit_transform(X)
        beta = np.concatenate([self.beta_a, self.beta_i])
        y = X_s.dot(beta) + np.random.normal(loc=0, scale=np.sqrt(self.noise_var), size=self.n)
        return X, y

    def get_ground_truth(self, X):
        X_s = PolynomialFeatures(2, interaction_only=False, include_bias=False).fit_transform(X)
        beta = np.concatenate([self.beta_a, self.beta_i])
        return X_s.dot(beta)


class HigherOrderSimulator(BaseSimulator):
    def __init__(self,
                 n,
                 p,
                 noise_var=0.1,
                 x_var=1.,
                 degree=3,
                 with_input_blocks=False,
                 drop_a=0.2,
                 drop_i=0.8,
                 discretize_beta=False,
                 discretize_x=False,
                 *args, **kwargs):
        """
        A vanilla simulator that simulates an arbitrary high-order Polynomial,
        for benchmarking interaction effects
        Args:
            n:
            p:
            noise_var:
            degree:
            with_input_blocks:
            drop_a:
            drop_i:
            discretize_beta:
            discretize_x:
            max_x:
        """
        self.n = n
        self.p = p
        self.with_input_blocks = with_input_blocks
        self.noise_var = noise_var
        self.x_var = x_var
        self.degree = degree
        self.polynomial_fitter = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        self.polynomial_fitter.fit(np.zeros((self.n, self.p)))
        self.beta_a = np.zeros(p)
        self.beta_i = np.zeros(self.polynomial_fitter.n_output_features_ - p)
        self.powers_i_ = self.polynomial_fitter.powers_[p:]
        self.drop_a = drop_a
        self.drop_i = drop_i
        if discretize_beta:
            self.beta_rng = lambda p: np.random.choice(range(-1, 2), p)
        else:
            self.beta_rng = lambda p: np.random.uniform(-1, 1, p)
        if discretize_x:
            self.x_rng = lambda n: np.random.poisson(x_var, n)
        else:
            self.x_rng = lambda n: np.random.normal(0, np.sqrt(x_var), n)
        self.is_beta_built = False

    def sample_effect(self):
        # additive
        a_idx = np.random.choice(self.p, int(np.ceil(self.p * (1 - self.drop_a))), replace=False)
        self.beta_a[a_idx] = self.beta_rng(len(a_idx))
        # interaction
        i_idx = np.random.choice(len(self.beta_i), int(np.ceil(len(self.beta_i) * (1 - self.drop_i))), replace=False)
        self.beta_i[i_idx] = self.beta_rng(len(i_idx))
        self.is_beta_built = True

    def set_effect(self, beta_a, beta_i):
        self.beta_a = beta_a
        self.beta_i = beta_i
        self.is_beta_built = True

    def sample_data(self, N=None, *args, **kwargs):
        N = self.n if N is None else N
        X = self.x_rng(N * self.p).reshape(N, self.p)
        X_s = self.polynomial_fitter.transform(X)
        if not self.is_beta_built:
            self.sample_effect()
        beta = np.concatenate([self.beta_a, self.beta_i])
        y = X_s.dot(beta) + np.random.normal(0, np.sqrt(self.noise_var), N)
        if self.with_input_blocks:
            X = [X[:, i] if len(X.shape) > 2 else X[:, i].reshape(X.shape[0], 1) for i in range(X.shape[1])]
        return X, y

    def get_ground_truth(self, X):
        if self.with_input_blocks:
            X_ = np.concatenate(X, axis=1)
        else:
            X_ = X
        X_s = self.polynomial_fitter.transform(X_)
        beta = np.concatenate([self.beta_a, self.beta_i])
        return X_s.dot(beta)

    def get_nonzero_powers(self):
        if not self.is_beta_built:
            self.sample_effect()
            self.is_beta_built = True
        return self.powers_i_[np.where(self.beta_i != 0)]


class CorrelatedDataSimulator(HigherOrderSimulator):
    def __init__(self,
                 n,
                 p,
                 noise_var=0.1,
                 data_cov_matrix=None,
                 degree=3,
                 with_input_blocks=False,
                 *args, **kwargs):
        """
        Simulator for correlated Xs, inherited from `HigherOrderSimulator`
        The correlated data is achieved by sampling from a multivariate
        normal distribution
        Args:
            *args:
            **kwargs:

        Returns:

        """
        super().__init__(
            n=n,
            p=p,
            noise_var=noise_var,
            degree=degree,
            with_input_blocks=with_input_blocks,
            *args, **kwargs)
        self.x_rng = self._get_data_rng(data_cov_matrix)

    def _get_data_rng(self, data_cov_matrix):
        from numpy.random import multivariate_normal as mvn
        mu = np.zeros(self.p)
        cov = np.array(data_cov_matrix)
        assert len(mu) == cov.shape[0] == cov.shape[1]
        rng = lambda x: mvn(mu, cov, size=x)
        return rng

    def sample_data(self, N=None, *args, **kwargs):
        N = self.n if N is None else N
        X = self.x_rng(N).reshape(N, self.p)
        X_s = self.polynomial_fitter.transform(X)
        if not self.is_beta_built:
            self.sample_effect()
            self.is_beta_built = True
        beta = np.concatenate([self.beta_a, self.beta_i])
        y = X_s.dot(beta) + np.random.normal(0, np.sqrt(self.noise_var), N)
        if self.with_input_blocks:
            X = [X[:, i] if len(X.shape) > 2 else X[:, i].reshape(X.shape[0], 1) for i in range(X.shape[1])]
        return X, y


class HiddenStateSimulator(HigherOrderSimulator):
    def __init__(self, n, x_index, h_index=None, degree=2, interaction_strength=None, *args, **kwargs):
        """
        Args:
            n:
            x_index:
            h_index:
            degree:
            interaction_strength: interaction strength defines drop_i as well as beta_rng for interaction terms
                effect sizes
            *args:
            **kwargs:
        """
        if "noise_var" in kwargs:
            assert kwargs['noise_var'] == 0, "HiddenStateSimulator must set Noise_var=0; got %s" % kwargs['noise_var']
        self.x_index = x_index
        self.x_len = len(self.x_index)
        self.interaction_strength = interaction_strength
        self.h_index = h_index if h_index is not None else []
        self.h_len = len(self.h_index)
        # the order for concat is x + h
        p = self.x_len + self.h_len
        if interaction_strength is None:
            super().__init__(n=n, p=p, degree=degree, noise_var=0, drop_a=0, *args, **kwargs)
        else:
            super().__init__(n=n, p=p, degree=degree, noise_var=0, drop_a=0, drop_i=1 - interaction_strength, *args,
                             **kwargs)
        # overwrite
        self.polynomial_fitter = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        self.polynomial_fitter.fit(np.zeros((self.n, self.p)))
        self.beta_a = np.zeros(p)
        self.beta_i = np.zeros(self.polynomial_fitter.n_output_features_ - p)
        self.powers_i_ = self.polynomial_fitter.powers_[p:]
        if self.interaction_strength is None:
            self.beta_i_rng = self.beta_rng
        else:
            # normal distribution has 95% prob. of falling within mu +/- 2*sigma
            self.beta_i_rng = lambda n: np.sign(np.random.uniform(-1, 1, n)) * np.random.uniform(
                self.interaction_strength, 0.1, n)

    def sample_effect(self):
        # additive
        a_idx = np.random.choice(self.p, int(np.ceil(self.p * (1 - self.drop_a))), replace=False)
        self.beta_a[a_idx] = self.beta_rng(len(a_idx))
        # interaction
        i_idx = np.random.choice(len(self.beta_i), int(np.ceil(len(self.beta_i) * (1 - self.drop_i))), replace=False)
        self.beta_i[i_idx] = self.beta_i_rng(len(i_idx))
        self.is_beta_built = True

    def sample_data(self, N=None, hs=None, *args, **kwargs):
        assert self.h_len == 0 or hs is not None, "If h_index is not empty, must parse `hs` in argument"
        N = self.n if N is None else N
        X = self.x_rng(N * self.x_len).reshape(N, self.x_len)
        if hs is not None:
            h = hs[:, self.h_index]
            X = np.concatenate([X, h], axis=1)
        X_s = self.polynomial_fitter.transform(X)
        if not self.is_beta_built:
            self.sample_effect()
        beta = np.concatenate([self.beta_a, self.beta_i], )
        y = X_s.dot(beta) + np.random.normal(0, np.sqrt(self.noise_var), N)
        if self.with_input_blocks:
            X = [X[:, i] if len(X.shape) > 2 else X[:, i].reshape(X.shape[0], 1) for i in range(X.shape[1])]
        return X, y


class _OntologyPolynomial:
    def __init__(self, ontology_simulator, n, noise_var=0.1):
        self.ontology_simulator = ontology_simulator
        self.n = n
        self.hidden_state_simulators = []
        self.hidden_state_nodes = [n for n in ontology_simulator.T.nodes if type(n) is str and n.startswith('h')]
        self.num_nodes = len(ontology_simulator.G)
        self.noise_var = noise_var
        self.is_hs_built = False

    def sample_data(self, N=None):
        N = N if N is not None else self.n
        X = np.zeros((N, self.num_nodes))
        h = np.zeros((N, len(self.hidden_state_nodes)))
        assert (not self.is_hs_built) or len(self.hidden_state_simulators) == len(self.hidden_state_nodes)
        for h_i in range(len(self.hidden_state_nodes)):
            x_index = sorted([x for x in self.ontology_simulator.T.predecessors('h%i' % h_i) if type(x) is int])
            h_index = sorted(
                [int(x.lstrip('h')) for x in self.ontology_simulator.T.predecessors('h%i' % h_i) if type(x) is str])
            interaction_str = np.mean([self.ontology_simulator.T[x]['h%i' % h_i]['weight'] for x in
                                       self.ontology_simulator.T.predecessors('h%i' % h_i)])
            # interaction_str = 0.2
            if not self.is_hs_built:
                self.hidden_state_simulators.append(HiddenStateSimulator(n=self.n, x_index=x_index, h_index=h_index,
                                                                         interaction_strength=interaction_str))
            if h_index:
                x_, y_ = self.hidden_state_simulators[h_i].sample_data(N=N, hs=h)
            else:
                x_, y_ = self.hidden_state_simulators[h_i].sample_data(N=N)
            X[:, x_index] = x_[:, 0:(x_.shape[1] - len(h_index))]
            h[:, h_i] = y_
        self.is_hs_built = True
        y = y_ + np.random.normal(0, np.sqrt(self.noise_var), N)
        return X, y

    def get_ground_truth(self, X, return_h=False):
        assert self.is_hs_built
        N = X.shape[0]
        h = np.zeros((N, len(self.hidden_state_nodes)))
        for h_i in range(len(self.hidden_state_nodes)):
            x_index = sorted([x for x in self.ontology_simulator.T.predecessors('h%i' % h_i) if type(x) is int])
            h_index = sorted(
                [int(x.lstrip('h')) for x in self.ontology_simulator.T.predecessors('h%i' % h_i) if type(x) is str])
            if h_index:
                h_input = np.concatenate([X[:, x_index], h[:, h_index]], axis=1)
                y_ = self.hidden_state_simulators[h_i].get_ground_truth(h_input)
            else:
                h_input = X[:, x_index]
                y_ = self.hidden_state_simulators[h_i].get_ground_truth(h_input)
            h[:, h_i] = y_
        self.is_hs_built = True
        y = y_
        if return_h:
            return y, h
        else:
            return y


class OntologySimulator:
    def __init__(self, n, layout='spring', seed=None, sampler_kwargs=None):
        """
        Args:
            n: number of nodes/Xs
            layout:
            seed:
        Examples:
            from BioNAS.utils.simulator import OntologySimulator, HiddenStateSimulator
            ot = OntologySimulator(20, seed=1710)
            ot.draw('ontology_graph.pdf')
            x,y =ot.sampler.sample_data(1000)
            x_,y_ =ot.sampler.sample_data(100)
            from sklearn.linear_model import LinearRegression
            lm = LinearRegression().fit(x, y)
            print(lm.score(x_, y_)) # test r2=0.505
        """
        import networkx as nx
        self.n = n
        self.seed = seed
        self.backend = nx
        if layout == 'spring':
            self.layout_ = nx.spring_layout
        elif layout == 'graphviz':
            self.layout_ = lambda g: nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
        else:
            raise Exception('cannot understand layout: %s' % layout)
        self.G = nx.generators.random_graphs.powerlaw_cluster_graph(n, m=1, p=0., seed=seed)
        self.set_weights()
        self._build_tree()
        if sampler_kwargs is None:
            sampler_kwargs = {'n': 1000, 'noise_var': 0.1}
        self.sampler = _OntologyPolynomial(self, **sampler_kwargs)

    @property
    def adjacency_matrix(self):
        return self.backend.adjacency_matrix(self.G).todense()

    def set_weights(self):
        np.random.seed(self.seed)
        for e in self.G.edges():
            self.G[e[0]][e[1]]['weight'] = np.random.uniform(0.1, 1)

    def _build_tree(self):
        """
        TODO: still cannot convert G weights to T weights
        """
        G = self.G
        e_G = sorted([e for e in G.edges(data=True)], key=lambda x: x[-1]['weight'], reverse=True)
        e_T = []
        cutoffs = [0.7, 0.4, 0.1]

        hidden_states_dict = {}
        h_count = 0

        for cutoff in cutoffs:
            sg = self.backend.Graph([e for e in e_G if e[-1]['weight'] >= cutoff])

            for cc in self.backend.connected_components(sg):
                h = "h%i" % h_count
                # tmp = []
                tmp = defaultdict(list)
                for i in cc:
                    if i in hidden_states_dict:
                        h_ = hidden_states_dict[i]
                        # tmp.append((h_, h))
                        tmp[(h_, h)].extend([sg[i][j]['weight'] for j in sg[i]])
                    else:
                        # tmp.append((i, h))
                        tmp[(i, h)].extend([sg[i][j]['weight'] for j in sg[i]])
                for i in cc:
                    hidden_states_dict[i] = h
                if tmp:
                    h_count += 1
                    e_T.extend([(k[0], k[1], {'weight': np.mean(tmp[k])}) for k in tmp])
                    # TODO: still cannot convert G weights to T weights
                    # e_T.extend([( k[0], k[1]) for k in tmp ])

        self.T = self.backend.DiGraph(e_T)
        return

    def draw_tree(self, save_fn=None, ax=None):
        # layout configs
        T = self.T
        pos = self.backend.drawing.nx_agraph.graphviz_layout(T, prog='dot')
        if self.backend.is_weighted(T):
            weights = self.backend.get_edge_attributes(T, 'weight')
            for _ in weights:
                weights[_] = np.round(weights[_], 2)
            # nodes
            self.backend.draw_networkx_nodes(T, pos, ax=ax)
            # edges; normalize weights to 10
            self.backend.draw_networkx_edges(T, pos, width=np.array(list(weights.values())) * 5, ax=ax)
            # labels
            self.backend.draw_networkx_labels(T, pos)
            self.backend.draw_networkx_edge_labels(T, pos, edge_labels=weights, font_size=7, ax=ax)
        else:
            self.backend.draw_networkx(T, pos, ax=ax)
        # save/show
        if save_fn:
            plt.savefig(save_fn)

    def draw_graph(self, save_fn=None, ax=None):
        # layout config
        G = self.G
        pos = self.layout_(G)
        weights = self.backend.get_edge_attributes(G, 'weight')
        for _ in weights:
            weights[_] = np.round(weights[_], 2)
        # nodes
        self.backend.draw_networkx_nodes(G, pos, ax=ax)
        # edges; normalize weights to 10
        self.backend.draw_networkx_edges(G, pos, width=np.array(list(weights.values())) * 10, ax=ax)
        # labels
        self.backend.draw_networkx_labels(G, pos)
        self.backend.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=7, ax=ax)
        # save/show
        if save_fn:
            plt.savefig(save_fn)

    def draw(self, save_fn):
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121)
        self.draw_graph()
        ax1.set_title('Graph')
        ax2 = fig.add_subplot(122)
        self.draw_tree()
        ax2.set_title('Tree')
        fig.savefig(save_fn)
