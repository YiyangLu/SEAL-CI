"""SEAL-CI model."""

import torch
import random
from tqdm import trange
from layers import SEAL
from utils import hierarchical_graph_reader, GraphDatasetGenerator
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/seal_ci')

class SEALCITrainer(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """
    def __init__(self, args):
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        self.args = args
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        self.dataset_generator = GraphDatasetGenerator(self.args.graphs)
        self._setup_macro_graph()
        self._create_split()
        self._create_labeled_target()
        self._create_node_indices()

    def _setup_model(self):
        """
        Creating a SEAL model.
        """
        self.model = SEAL(self.args, self.dataset_generator.number_of_features,
                          self.dataset_generator.number_of_labels)

    def _setup_macro_graph(self):
        """
        Creating an edge list for the hierarchical graph.
        """
        self.macro_graph_edges = [[edge[0], edge[1]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = self.macro_graph_edges + [[edge[1], edge[0]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = torch.t(torch.LongTensor(self.macro_graph_edges))

    def _create_split(self):
        """
        Creating a labeled-unlabeled split.
        """
        graph_indices = [index for index in range(len(self.dataset_generator.graphs))]
        random.shuffle(graph_indices)
        self.labeled_indices = graph_indices[0:self.args.labeled_count]
        self.unlabeled_indices = graph_indices[self.args.labeled_count:]

    def _create_labeled_target(self):
        """
        Creating a mask for labeled instances and a target for them.
        """
        self.labeled_mask = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        self.labeled_target = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        self.init_labeled_mask = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        indices = torch.LongTensor(self.labeled_indices)
        self.labeled_mask[indices] = 1
        self.labeled_target[indices] = self.dataset_generator.target[indices]
        self.init_labeled_mask[indices] = 1
        

    def _create_node_indices(self):
        """
        Creating an index of nodes.
        """
        self.node_indices = [index for index in range(self.macro_graph.number_of_nodes())]
        self.node_indices = torch.LongTensor(self.node_indices)

    def fit_a_single_model(self,step):
        """
        Fitting a single SEAL model.
        """
        # self._setup_model()
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=self.args.learning_rate,
        #                              weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            l_predictions, h_predictions, penalty = self.model(self.dataset_generator.graphs, self.macro_graph_edges)
            h_loss = torch.nn.functional.nll_loss(h_predictions[self.labeled_mask == 1],
                                                self.labeled_target[self.labeled_mask == 1])
            l_loss = torch.nn.functional.nll_loss(l_predictions[self.labeled_mask == 1],
                                                self.labeled_target[self.labeled_mask == 1])
            hl_kl = torch.nn.functional.kl_div(h_predictions[self.labeled_mask == 1],
                                                l_predictions[self.labeled_mask == 1], log_target=True, reduction='batchmean')
            loss = h_loss + l_loss + self.args.gamma*penalty+hl_kl
            writer.add_scalar('Loss/train', loss, step*self.args.epochs+epoch)

            loss.backward()
            self.optimizer.step()

    def score_a_single_model(self,label=0,original=False):
        """
        Scoring the SEAL model.
        label: 0 for unlabeled, 1 for labeled
        original: False for updated, True for original 
        """
        self.model.eval()
        l_predictions ,h_predictions, _ = self.model(self.dataset_generator.graphs, self.macro_graph_edges)
        scores, prediction_indices = h_predictions.max(dim=1)

        if original == False:
            correct = prediction_indices[self.labeled_mask == label]
            correct = correct.eq(self.dataset_generator.target[self.labeled_mask == label]).sum().item()
            normalizer = prediction_indices[self.labeled_mask == label].shape[0]
            accuracy = float(correct)/float(normalizer)
        else:
            correct = prediction_indices[self.init_labeled_mask == label]
            correct = correct.eq(self.dataset_generator.target[self.init_labeled_mask == label]).sum().item()
            normalizer = prediction_indices[self.init_labeled_mask == label].shape[0]
            accuracy = float(correct)/float(normalizer)
        return scores, prediction_indices, accuracy
            
    def _choose_best_candidate(self, scores, indices):
        """
        Choosing the best candidate based on predictions.
        :param scores: Scores.
        :param indices: Vector of likely labels.
        :return candidate: Node chosen.
        :return label: Label of node.
        """
        nodes = self.node_indices[self.labeled_mask == 0]
        sub_scores = scores[self.labeled_mask == 0]
        sub_scores, candidate = sub_scores.max(dim=0)
        candidate = nodes[candidate]
        label = indices[candidate]
        return candidate, label

    def _update_target(self, candidate, label):
        """
        Adding the new node to the mask and the target is updated with the predicted label.
        :param candidate: Candidate node identifier.
        :param label: Label of candidate node.
        """
        self.labeled_mask[candidate] = 1
        self.labeled_target[candidate] = label      

    def fit(self):
        """
        Training models sequentially.
        """
        print("\nTraining started.\n")
        budget_size = trange(self.args.budget, desc='Unlabeled Accuracy: ', leave=True)
        self._setup_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        for step in budget_size:
            self.fit_a_single_model(step)
            scores, prediction_indices, accuracy = self.score_a_single_model()
            candidate, label = self._choose_best_candidate(scores, prediction_indices)
            self._update_target(candidate, label)
            _, _, ori_accuracy = self.score_a_single_model(label=0,original=True)
            writer.add_scalar('Accuracy/train', ori_accuracy, step)
            budget_size.set_description("Unlabeled Accuracy (original):%g" % round(ori_accuracy, 4))
        writer.close()

    def score(self):
        """
        Scoring the model.
        """
        print("\nModel scoring.\n")
        scores, prediction_indices, accuracy = self.score_a_single_model(label=0,original=True)
        print("Unlabeled Accuracy (original):%g" % round(accuracy, 4))
        scores, prediction_indices, accuracy = self.score_a_single_model(label=1,original=True)
        print("Labeled Accuracy (original):%g" % round(accuracy, 4))

