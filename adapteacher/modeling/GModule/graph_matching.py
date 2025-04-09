import torch
from torch import nn
import torch.nn.functional as F

import sklearn.cluster as cluster

from adapteacher.modeling.GModule.build_graph import PrototypeComputation
# from model.discriminator import GradientReversal

from adapteacher.modeling.GModule.utils.losses import BCEFocalLoss
from adapteacher.modeling.GModule.utils.affinity import Affinity
from adapteacher.modeling.GModule.utils.attentions import MultiHeadAttention, HyperGraph
from adapteacher.modeling.GModule.utils.graph_network import MAGNN, Topo_Graph
from adapteacher.modeling.GModule.utils.sinkhorn_distance import SinkhornDistance
from adapteacher.modeling.GModule.utils.adaptation_loss import ProtoLoss

import logging


class GRAPHHead(torch.nn.Module):
    def __init__(self, opt, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        
        if mode == 'in':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = opt.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if opt.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            middle_tower.append(nn.ReLU())
        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        # initialization
        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower


class V2GConv(torch.nn.Module):
    # Project the sampled visual features to the graph embeddings:
    # visual features: [0,+INF) -> graph embedding: (-INF, +INF)
    def __init__(self, opt, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        if mode == 'in':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN #d:2
        elif mode == 'out':
            num_convs = opt.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = opt.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        # self.mapping1 = nn.Conv2d(64, in_channels, 1)
        # self.mapping2 = nn.Conv2d(128, in_channels, 1)
        # self.mapping3 = nn.Conv2d(256, in_channels, 1)

        # self._initialize_weights(self.mapping1)
        # self._initialize_weights(self.mapping2)
        # self._initialize_weights(self.mapping3)

        middle_tower = []  
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if opt.MODEL.MIDDLE_HEAD.IN_NORM == 'GN': #d
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif opt.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            if i != (num_convs - 1):
                middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        middle_tower = []
        # for l, feature in enumerate(x):
        #     middle_tower.append(self.middle_tower(feature))
        # for feature_dict in x:
            # feature = x[feature_dict]
            # if feature_dict == 'vgg0':
            #     feature = self.mapping1(feature)
            # elif feature_dict == 'vgg1':
            #     feature = self.mapping2(feature)
            # elif feature_dict == 'vgg2':
            #     feature = self.mapping3(feature)
        for dict_name in x:
            middle_tower.append(self.middle_tower(x[dict_name]))
        # middle_tower.append(self.middle_tower(x['vgg4']))
        return middle_tower

def build_V2G_linear(opt):
    if opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 2:
        head_in_ln = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
        )
    elif opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 1:
        head_in_ln = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
        )
    elif opt.MODEL.MIDDLE_HEAD.NUM_CONVS_IN == 0:
        head_in_ln = nn.LayerNorm(256, elementwise_affine=False)
    else:
        head_in_ln = nn.LayerNorm(256, elementwise_affine=True)
    return head_in_ln


class GModule(torch.nn.Module):

    def __init__(self, opt, in_channels):
        super(GModule, self).__init__()

        init_item = []
        self.opt = opt
        # print(opt.MODEL)
        self.logger = logging.getLogger("fcos_core.trainer")
        self.logger.info('node dis setting: ' + str(opt.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE))
        self.logger.info('use hyper graph: '  + str(opt.MODEL.MIDDLE_HEAD.GM.WITH_HyperGNN))
        self.fpn_strides = opt.MODEL.FCOS.FPN_STRIDES
        self.num_classes = opt.MODEL.FCOS.NUM_CLASSES

        self.with_hyper_graph = opt.MODEL.MIDDLE_HEAD.GM.WITH_HyperGNN
        self.num_hyper_edge = opt.MODEL.MIDDLE_HEAD.GM.HyperEdgeNum
        self.num_hypergnn_layer = opt.MODEL.MIDDLE_HEAD.GM.NUM_HYPERGNN_LAYER
        self.angle_eps = opt.MODEL.MIDDLE_HEAD.GM.ANGLE_EPS

        # One-to-one (o2o) matching or many-to-many (m2m) matching?
        self.matching_cfg = opt.MODEL.MIDDLE_HEAD.GM.MATCHING_CFG  # 'o2o' and 'm2m'
        self.with_cluster_update = opt.MODEL.MIDDLE_HEAD.GM.WITH_CLUSTER_UPDATE  # add spectral clustering to update seeds
        self.with_semantic_completion = opt.MODEL.MIDDLE_HEAD.GM.WITH_SEMANTIC_COMPLETION  # generate hallucination nodes

        # add quadratic matching constraints.
        self.with_quadratic_matching = opt.MODEL.MIDDLE_HEAD.GM.WITH_QUADRATIC_MATCHING

        # Several weights hyper-parameters
        self.weight_matching = opt.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_WEIGHT
        self.weight_nodes = opt.MODEL.MIDDLE_HEAD.GM.NODE_LOSS_WEIGHT
        self.weight_dis = opt.MODEL.MIDDLE_HEAD.GM.NODE_DIS_WEIGHT
        self.lambda_dis = opt.MODEL.MIDDLE_HEAD.GM.NODE_DIS_LAMBDA
        self.weight_topo = 1 # d:1

        # Detailed settings
        self.with_domain_interaction = opt.MODEL.MIDDLE_HEAD.GM.WITH_DOMAIN_INTERACTION
        self.with_complete_graph = opt.MODEL.MIDDLE_HEAD.GM.WITH_COMPLETE_GRAPH
        self.with_node_dis = opt.MODEL.MIDDLE_HEAD.GM.WITH_NODE_DIS
        self.with_global_graph = opt.MODEL.MIDDLE_HEAD.GM.WITH_GLOBAL_GRAPH

        # Test 3 positions to put the node alignment discriminator. (the former is better)
        self.node_dis_place = opt.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE

        # future work
        self.with_cond_cls = opt.MODEL.MIDDLE_HEAD.GM.WITH_COND_CLS  # use conditional kernel for node classification? (didn't use)
        self.with_score_weight = opt.MODEL.MIDDLE_HEAD.GM.WITH_SCORE_WEIGHT  # use scores for node loss (didn't use)

        # Node sampling
        self.graph_generator = PrototypeComputation(self.opt)
        # Pre-processing for the vision-to-graph transformation
        self.head_in_cfg = opt.MODEL.MIDDLE_HEAD.IN_NORM

        if self.head_in_cfg == 'LN':
            self.head_in_ln = build_V2G_linear(self.opt)
            init_item.append('head_in_ln')
        else:
            print(self.head_in_cfg)
            # import ipdb;
            # ipdb.set_trace()
            self.head_in = V2GConv(self.opt, in_channels, in_channels, mode='in')

        # node classification layers
        self.node_cls_middle = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        init_item.append('node_cls_middle')

        
        self.cross_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Cross Graph Interaction

        if self.with_hyper_graph:
            self.intra_domain_graph = HyperGraph(emb_dim=256, K_neigs=self.num_hyper_edge, num_layer=self.num_hypergnn_layer)  # Intra-domain graph aggregation
        else:
            self.intra_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation

        self.morph_gnn = MAGNN(256, 256, True)

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=256)
        self.node_affinity_topo = Affinity(d=64)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Structure-aware Matching Loss
        # Different matching loss choices
        if opt.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'L1':
            self.matching_loss = nn.L1Loss(reduction='sum')
        elif opt.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'MSE':
            self.matching_loss = nn.MSELoss(reduction='sum')
        elif opt.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'BCE':
            self.matching_loss = BCEFocalLoss()
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')

        if self.with_node_dis:
            self.grad_reverse = GradientReversal(self.lambda_dis)
            self.node_dis_2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            init_item.append('node_dis')
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.sinkhorn = SinkhornDistance(eps=0.01, max_iter=100, reduction='mean')

        self.K = 1024
        self.register_buffer('source_queue', torch.zeros(self.num_classes, self.K, 256))
        self.register_buffer('target_queue', torch.zeros(self.num_classes, self.K, 256))
        self.register_buffer('queue_ptr', torch.zeros(self.num_classes, dtype=(torch.long)))

        # Graph-guided Memory Bank
        self.seed_project = nn.Linear(256, 256)  # projection layer for the node completion
        # self.seed_project_tg = nn.Linear(256, 256)
        self.register_buffer('sr_center', torch.randn(self.num_classes, 256))  # seed = bank
        self.register_buffer('tg_center', torch.randn(self.num_classes, 256))

        # self.register_buffer('target_queue_ptr', torch.zeros(1, dtype=(torch.long)))
        # self.graph_s = Topo_Graph(256)
        # self.graph_t = Topo_Graph(256)

        self.prototype = ProtoLoss(nav_t=0.1,beta=0.001, num_classes=self.num_classes)
        self._init_weight(init_item)

    def _init_weight(self, init_item=None):
        nn.init.normal_(self.seed_project.weight, std=0.01)
        nn.init.constant_(self.seed_project.bias, 0)
        # nn.init.normal_(self.seed_project_tg.weight, std=0.01)
        # nn.init.constant_(self.seed_project_tg.bias, 0)
        if 'node_dis' in init_item:
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_dis initialized')
        if 'node_cls_middle' in init_item:
            for i in self.node_cls_middle:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_cls_middle initialized')
        if 'head_in_ln' in init_item:
            for i in self.head_in_ln:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('head_in_ln initialized')

    def forward(self, images, features, targets=None, score_maps=None):
        '''
        We have equal number of source/target feature maps
        features: [sr_feats, tg_feats]
        targets: [sr_targets, None]

        '''
        if targets:
            features, feat_loss = self._forward_train(images, features, targets, score_maps)
            return features, feat_loss

        else:
            features = self._forward_inference(images, features)
            return features, None

    def _forward_train(self, images, features, targets=None, score_maps=None):
        targets_src, targets_tgt = targets
        targets_src = [x['instances'] for x in targets_src]
        targets_tgt = [x['instances'] for x in targets_tgt]
        features_s, features_t = features
        middle_head_loss = {}
        
        # STEP1: vision-to-graph transformation
        # LN is conducted on the node embedding
        # GN/BN are conducted on the whole image feature
        if self.head_in_cfg != 'LN':
            features_s = self.head_in(features_s)
            features_t = self.head_in(features_t)
            nodes_1, labels_1, weights_1 = self.graph_generator(
                self.compute_locations(features_s), features_s, targets_src
            )
            nodes_2, labels_2, weights_2 = self.graph_generator(
                self.compute_locations(features_t), features_t, targets_tgt#score_maps
            )
        else:
            nodes_1 = self.head_in_ln(nodes_1)
            nodes_2 = self.head_in_ln(nodes_2) if nodes_2 is not None else None

        # TODO: Matching can only work for adaptation when both source and target nodes exist.
        # Otherwise, we split the source nodes half-to-half to train SIGMA

        if nodes_2 is not None:  # Both domains have graph nodes

            # STEP2: Conduct Domain-guided Node Completion (DNC)
            (nodes_1, nodes_2), (labels_1, labels_2), (weights_1, weights_2) = \
                self._forward_preprocessing_source_target((nodes_1, nodes_2), (labels_1, labels_2),
                                                          (weights_1, weights_2))
            
            # STEP3: Single-layer HGCN
            if self.with_complete_graph:
                nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
                nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            # STEP4: Update Graph-guided Memory Bank (GMB) with enhanced node embedding
            # self.update_seed(nodes_1, labels_1, nodes_2, labels_2)

            self._dequeue_and_enqueue(nodes_1, nodes_2, labels_1, labels_2)
            loss_sink = self._forward_topology_loss(nodes_1, nodes_2, labels_1, labels_2)
            middle_head_loss.update({'loss_topo': loss_sink * self.weight_topo})

            if self.with_node_dis and self.node_dis_place == 'intra':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'loss_dis': node_dis_loss})

            # STEP5: Conduct Cross Graph Interaction (CGI)
            if self.with_domain_interaction:
                nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)

            if self.with_node_dis and self.node_dis_place == 'inter':
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'loss_dis': node_dis_loss})

            # STEP6: Generate node loss
            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2], dim=0),
                torch.cat([labels_1, labels_2], dim=0),
                torch.cat([weights_1, weights_2], dim=0)
            )

        else:  # Use all source nodes for training if no target nodes in the early training stage
            (nodes_1, nodes_2), (labels_1, labels_2) = \
                self._forward_preprocessing_source(nodes_1, labels_1)
            
            nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
            nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

            self._dequeue_and_enqueue(nodes_1, nodes_2, labels_1, labels_2)
            
            nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)
            node_loss = self._forward_node_loss(
                torch.cat([nodes_1, nodes_2], dim=0),
                torch.cat([labels_1, labels_2], dim=0)
            )

        middle_head_loss.update({'loss_node': self.weight_nodes * node_loss})

        # STEP7: Generate Semantic-aware Node Affinity and Structure-aware Matching loss
        if self.matching_cfg != 'none':
            matching_loss_affinity, affinity = self._forward_aff(nodes_1, nodes_2, labels_1, labels_2)
            middle_head_loss.update({'loss_mat_aff': self.weight_matching * matching_loss_affinity})

            if self.with_quadratic_matching:
                matching_loss_quadratic = self._forward_qu(nodes_1, nodes_2, edges_1.detach(), edges_2.detach(), affinity)
                middle_head_loss.update({'loss_mat_qu': matching_loss_quadratic})

        return features, middle_head_loss

    def _forward_preprocessing_source_target(self, nodes, labels, weights):

        '''
        nodes: sampled raw source/target nodes
        labels: the ground-truth/pseudo-label of sampled source/target nodes
        weights: the confidence of sampled source/target nodes ([0.0,1.0] scores for target nodes and 1.0 for source nodes )

        We permute graph nodes according to the class from 1 to K and complete the missing class.

        '''
        sr_nodes, tg_nodes = nodes
        sr_nodes_label, tg_nodes_label = labels
        sr_loss_weight, tg_loss_weight = weights

        labels_exist = torch.cat([sr_nodes_label, tg_nodes_label]).unique()
        labels_missing = torch.tensor(list(set(torch.arange(self.num_classes).tolist()) - set(labels_exist.tolist())))

        sr_nodes_category_first = []
        tg_nodes_category_first = []

        sr_labels_category_first = []
        tg_labels_category_first = []

        sr_weight_category_first = []
        tg_weight_category_first = []

        for c in labels_exist:

            sr_indx = sr_nodes_label == c
            tg_indx = tg_nodes_label == c

            sr_nodes_c = sr_nodes[sr_indx]
            tg_nodes_c = tg_nodes[tg_indx]

            sr_weight_c = sr_loss_weight[sr_indx]
            tg_weight_c = tg_loss_weight[tg_indx]

            if sr_indx.any() and tg_indx.any():  # If the category appear in both domains, we directly collect them!

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                labels_sr = sr_nodes_c.new_ones(len(sr_nodes_c)) * c
                labels_tg = tg_nodes_c.new_ones(len(tg_nodes_c)) * c

                sr_labels_category_first.append(labels_sr)
                tg_labels_category_first.append(labels_tg)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(tg_weight_c)

            elif tg_indx.any():  # If there're no source nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(tg_nodes_c)

                tg_nodes_category_first.append(tg_nodes_c)
                sr_nodes_c = self.sr_center[c].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(tg_nodes_c.device) + sr_nodes_c if num_nodes < 5 \
                        else torch.normal(mean=sr_nodes_c,
                                          std=torch.abs(sr_nodes_c.std(0)).unsqueeze(0).expand(tg_nodes_c.size())).to(tg_nodes_c.device)
                else:
                    sr_nodes_c = torch.normal(0, 0.01, size=sr_nodes_c.size()).to(tg_nodes_c.device)
                
                sr_nodes_c = self.seed_project(sr_nodes_c)
                sr_nodes_category_first.append(sr_nodes_c)
                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(sr_nodes_c.device) * c)
                sr_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).to(sr_nodes_c.device))

                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(tg_nodes_c.device) * c)
                tg_weight_category_first.append(tg_weight_c)

            elif sr_indx.any():  # If there're no target nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(sr_nodes_c)

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_c = self.tg_center[c].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(tg_nodes_c.device) + tg_nodes_c if num_nodes < 5 \
                        else torch.normal(mean=tg_nodes_c,
                                          std=torch.abs(tg_nodes_c.std(0)).unsqueeze(0).expand(sr_nodes_c.size())).to(tg_nodes_c.device)
                else:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(tg_nodes_c.device)

                tg_nodes_c = self.seed_project(tg_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)
                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(sr_nodes_c.device) * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(tg_nodes_c.device) * c)
                tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).to(tg_nodes_c.device))
                sr_weight_category_first.append(sr_weight_c)

        for c in labels_missing:
            leng = [len(tensor) for tensor in sr_labels_category_first[1:]]
            num_nodes = sum(leng) // len(leng)

            sr_nodes_c = self.sr_center[c].unsqueeze(0).expand(num_nodes, 256)
            sr_nodes_c = torch.normal(0, 0.01, size=sr_nodes_c.size()).to(sr_nodes_c.device) + sr_nodes_c if num_nodes < 5 \
                else torch.normal(mean=sr_nodes_c, std=sr_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).to(sr_nodes_c.device)

            tg_nodes_c = self.sr_center[c].unsqueeze(0).expand(num_nodes, 256)
            tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(tg_nodes_c.device) + tg_nodes_c if num_nodes < 5 \
                else torch.normal(mean=tg_nodes_c, std=tg_nodes_c.std(0).unsqueeze(0).expand(tg_nodes_c.size())).to(tg_nodes_c.device)
            
            sr_nodes_c = self.seed_project(sr_nodes_c)
            tg_nodes_c = self.seed_project(tg_nodes_c)
            sr_nodes_category_first.append(sr_nodes_c)
            tg_nodes_category_first.append(tg_nodes_c)
            sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(sr_nodes_c.device) * c)
            tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).to(tg_nodes_c.device) * c)
            sr_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).to(sr_nodes_c.device))
            tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).to(tg_nodes_c.device))

        nodes_sr = torch.cat(sr_nodes_category_first, dim=0)
        nodes_tg = torch.cat(tg_nodes_category_first, dim=0)

        weight_sr = torch.cat(sr_weight_category_first, dim=0)
        weight_tg = torch.cat(tg_weight_category_first, dim=0)

        label_sr = torch.cat(sr_labels_category_first, dim=0)
        label_tg = torch.cat(tg_labels_category_first, dim=0)

        return (nodes_sr, nodes_tg), (label_sr, label_tg), (weight_sr, weight_tg)

    def _forward_preprocessing_source(self, sr_nodes, sr_nodes_label):
        labels_exist = sr_nodes_label.unique()

        nodes_1_cls_first = []
        nodes_2_cls_first = []
        labels_1_cls_first = []
        labels_2_cls_first = []

        for c in labels_exist:
            sr_nodes_c = sr_nodes[sr_nodes_label == c]
            if sr_nodes_c.size(0) == 1:
                sr_nodes_c = torch.cat((sr_nodes_c, sr_nodes_c.clone()), dim=0)       
            nodes_1_cls_first.append(torch.cat([sr_nodes_c[::2, :]]))
            nodes_2_cls_first.append(torch.cat([sr_nodes_c[1::2, :]]))

            labels_side1 = sr_nodes_c.new_ones(len(nodes_1_cls_first[-1])) * c
            labels_side2 = sr_nodes_c.new_ones(len(nodes_2_cls_first[-1])) * c

            labels_1_cls_first.append(labels_side1)
            labels_2_cls_first.append(labels_side2)

        nodes_1 = torch.cat(nodes_1_cls_first, dim=0)
        nodes_2 = torch.cat(nodes_2_cls_first, dim=0)

        labels_1 = torch.cat(labels_1_cls_first, dim=0)
        labels_2 = torch.cat(labels_2_cls_first, dim=0)

        return (nodes_1, nodes_2), (labels_1, labels_2)

    def _forward_intra_domain_graph(self, nodes):
        nodes = self.morph_gnn(nodes)
        nodes, edges = self.intra_domain_graph([nodes, nodes, nodes])
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        if self.with_global_graph:
            n_1 = len(nodes_1)
            n_2 = len(nodes_2)
            global_nodes = torch.cat([nodes_1, nodes_2], dim=0)
            global_nodes = self.cross_domain_graph(global_nodes, global_nodes, global_nodes)[0]

            nodes1_enahnced = global_nodes[:n_1]
            nodes2_enahnced = global_nodes[n_1:]
        else:
            nodes2_enahnced = self.cross_domain_graph([nodes_1, nodes_1, nodes_2])[0]
            nodes1_enahnced = self.cross_domain_graph([nodes_2, nodes_2, nodes_1])[0]

        return nodes1_enahnced, nodes2_enahnced

    def _forward_node_loss(self, nodes, labels, weights=None):

        labels = labels.long()
        assert len(nodes) == len(labels)

        if weights is None:  # Source domain
            if self.with_cond_cls:
                tg_embeds = self.node_cls_middle(self.tg_seed)
                logits = self.dynamic_fc(nodes, tg_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels,
                                        reduction='mean')
        else:  # Target domain
            if self.with_cond_cls:
                sr_embeds = self.node_cls_middle(self.sr_seed)
                logits = self.dynamic_fc(nodes, sr_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels.long(),
                                        reduction='none')
            node_loss = (node_loss * weights).float().mean() if self.with_score_weight else node_loss.float().mean()

        return node_loss

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        k = 20  # conduct clustering when we have enough graph nodes
        for cls in sr_labels.unique().long():
            bs = sr_nodes[sr_labels == cls].detach()

            if len(bs) > k and self.with_cluster_update:
                # TODO Use Pytorch-based GPU version
                sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                seed_cls = self.sr_seed[cls]
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                indx = (indx == indx[0])[1:]
                bs = bs[indx].mean(0)
            else:
                bs = bs.mean(0)

            momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
            self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

        if tg_nodes is not None:
            for cls in tg_labels.unique().long():
                bs = tg_nodes[tg_labels == cls].detach()
                if len(bs) > k and self.with_cluster_update:
                    seed_cls = self.tg_seed[cls]
                    sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
                                                    assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
                    indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
                    indx = (indx == indx[0])[1:]
                    bs = bs[indx].mean(0)
                else:
                    bs = bs.mean(0)
                momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
                self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def update_center(self):

        for cls in range(self.num_classes):
            bs_s = self.get_non_row(self.source_queue[cls], cls).mean(0)
            bs_t = self.get_non_row(self.target_queue[cls], cls).mean(0)

            momentum_s = torch.nn.functional.cosine_similarity(bs_s.unsqueeze(0), self.sr_center[cls].unsqueeze(0))
             
            self.sr_center[cls] = self.sr_center[cls] * momentum_s + bs_s * (1.0 - momentum_s)

            momentum_t = torch.nn.functional.cosine_similarity(bs_t.unsqueeze(0), self.tg_center[cls].unsqueeze(0))
            self.tg_center[cls] = self.tg_center[cls] * momentum_t + bs_t * (1.0 - momentum_t)

    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        if self.matching_cfg == 'o2o':
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_iter(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float().to(M.device)
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TODO Find a better reduction strategy
            TP_loss = self.matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
            matching_loss = TP_loss + FP_loss

        elif self.matching_cfg == 'm2m':  # Refer to the Appendix
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_inference(self, images, features):
        return features

    def _forward_qu(self, nodes_1, nodes_2, edges_1, edges_2, affinity):

        if self.with_hyper_graph:

            # hypergraph matching (high order)
            translated_indx = list(range(1, self.num_hyper_edge))+[int(0)]
            mathched_index = affinity.argmax(0)
            matched_node_1 = nodes_1[mathched_index]
            matched_edge_1 = edges_1.t()[mathched_index]
            matched_edge_1[matched_edge_1 > 0] = 1

            matched_node_2 =nodes_2
            matched_edge_2 =edges_2.t()
            matched_edge_2[matched_edge_2 > 0] = 1
            n_nodes = matched_node_1.size(0)

            angle_dis_list = []
            for i in range(n_nodes):
                triangle_1 = nodes_1[matched_edge_1[i, :].bool()]  # 3 x 256
                triangle_1_tmp = triangle_1[translated_indx]
                # print(triangle_1.size(), triangle_1_tmp.size())
                sin1 = torch.sqrt(1.- F.cosine_similarity(triangle_1, triangle_1_tmp).pow(2)).sort()[0]
                triangle_2 = nodes_2[matched_edge_2[i, :].bool()]  # 3 x 256
                triangle_2_tmp = triangle_2[translated_indx]
                sin2 = torch.sqrt(1.- F.cosine_similarity(triangle_2, triangle_2_tmp).pow(2)).sort()[0]
                angle_dis = (-1 / self.angle_eps  * (sin1 - sin2).abs().sum()).exp()
                angle_dis_list.append(angle_dis.view(1,-1))

            angle_dis_list = torch.cat(angle_dis_list)
            loss = angle_dis_list.mean()
        else:
            # common graph matching (2nd order)
            R = torch.mm(edges_1, affinity) - torch.mm(affinity, edges_2)
            loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def sinkhorn_iter(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()
        return log_alpha

    def dynamic_fc(self, features, kernel_par):
        weight = kernel_par
        return torch.nn.functional.linear(features, weight, bias=None)

    def dynamic_conv(self, features, kernel_par):
        weight = kernel_par.view(self.num_classes, -1, 1, 1)
        return torch.nn.functional.conv2d(features, weight)

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long().cpu(), :].to(x.device)
    
    # add by Lvxg
    def get_node_center(self, node1, node2, label1, label2, weight1=None, weight2=None):
        unique_label1 = torch.unique(label1)
        unique_label2 = torch.unique(label2)
        assert (unique_label1 == unique_label2).all()

        nodes1 = []; nodes2 = []

        if weight1 is not None and weight2 is not None:
            weights1 = []; weights2 = []
            for l in unique_label1:
                nodes1.append(node1[label1 == l].mean(0,True))
                weights1.append(weight1[label1 == l].double().mean())
                nodes2.append(node2[label2 == l].mean(0,True))
                weights2.append(weight2[label2 == l].double().mean())                

            nodes1 = torch.cat(nodes1, dim=0)
            nodes2 = torch.cat(nodes2, dim=0)

            weights1 = torch.stack(weights1).long()
            weights2 = torch.stack(weights2).long()
        
        else:
            for l in unique_label1:
                nodes1.append(node1[label1 == l].mean(0,True))
                nodes2.append(node2[label2 == l].mean(0,True))

            nodes1 = torch.cat(nodes1, dim=0)
            nodes2 = torch.cat(nodes2, dim=0)

            weights1 = unique_label1.new_ones(unique_label1.shape).long()
            weights2 = unique_label2.new_ones(unique_label2.shape).long()

        return nodes1, nodes2, unique_label1, unique_label2, weights1, weights2
    
    def _dequeue_and_enqueue(self, nodes_1, nodes_2, labels_1, labels_2):
        """
        For each class, store the nodes in separate queues 
        for the source domain and target domain. 
        If a class has no nodes, it will be filled using a normal distribution.
        """
        # assert (labels_1 == labels_2).all()
        unique_label1 = torch.unique(labels_1)
        unique_label2 = torch.unique(labels_2)
        assert (unique_label1 == unique_label2).all()

        for cls in unique_label1.long():
            nodes1 = nodes_1[labels_1 == cls].mean(0, True)
            nodes2 = nodes_2[labels_2 == cls].mean(0, True)

            ptr = int(self.queue_ptr[cls])
            self.source_queue[cls][ptr] = nodes1.detach() #[idx]
            self.target_queue[cls][ptr] = nodes2.detach() # [idx]
            ptr = (ptr + 1) % self.K
            self.queue_ptr[cls] = ptr    

        self.update_center()       

    def fill_null(self, domain):
        if domain == 'source':
            queue = self.source_queue
            n = self.source_queue_ptr
        elif domain == 'target':
            queue = self.target_queue
            n = self.target_queue_ptr

        K = self.num_classes - 1
        rows, cols = queue.shape
        indices = [n + i for i in range(-K, K + 1) if 0 <= n + i < cols if i != 0]
        mean_val = queue[:, indices].mean(dim=1)
        if torch.all(mean_val == 0):
            queue[:, n] = torch.randn((rows, 1), device=(queue.device))
        else:
            queue[:, n] = mean_val.unsqueeze(1)
    
    def _forward_topology_loss(self, nodes_s, nodes_t, labels_s, labels_t):
        """
        nodes: source & target domains
        
        """

        nodes_1, nodes_2, labels_1, labels_2, _, _ = \
            self.get_node_center(nodes_s, nodes_t,labels_s, labels_t)

        edge_1 = self.similarity_mat(nodes_1.detach(), self.sr_center.clone()) # [1:, :]
        edge_2 = self.similarity_mat(nodes_2.detach(), self.tg_center.clone()) # [1:, :]

        # nodes_1 = self.graph_s(nodes_1, edge_1)
        # nodes_2 = self.graph_t(nodes_2, edge_2)

        loss1, loss2 = self.prototype(nodes_1, nodes_t)
        
        loss_sink, _, _ = self.sinkhorn(nodes_s, nodes_t) 

        loss = loss1 + loss2 + loss_sink 

        return loss
    
    def contrastive_loss(self, queries, keys, queue):

        queries = F.normalize(queries, dim=1)
        keys = F.normalize(keys, dim=1)
        queue = F.normalize(queue, dim=1)

        logits_pos = torch.mm(queries, keys.t())
        logits_neg = torch.mm(queries, queue.t())
        
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=keys.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def get_non_row(self, x, idx):
        if torch.all(x==0, dim=1).any().item():
            # zero_raw_idx = (x.sum(dim=1) == 0).nonzero(as_tuple=True)[0][0]
            return x[:self.queue_ptr[idx]]
        else:
            return x
        
    def similarity_mat(self, x1, x2):
        tensor1_normalized = F.normalize(x1, p=2, dim=1, eps=1e-8)
        tensor2_normalized = F.normalize(x2, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.mm(tensor1_normalized, tensor2_normalized.t())
        return similarity_matrix

def build_graph_matching_head(opt, in_channels):
    return GModule(opt, in_channels)

