from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True

_C.DEBUG = CN()
_C.DEBUG.OVERFIT_NUM_IMAGES = -1

# Global tag
_C.EXPERIMENT_NAME = "dafne"

# Automatic Mixed Precision
_C.SOLVER.AMP = CN({"ENABLED": False})

# Optimizer type: one of "sgd", "adam"
_C.SOLVER.OPTIMIZER = "sgd"


# Set area/width/height min
_C.INPUT.MIN_AREA = 10
_C.INPUT.MIN_SIDE = 2

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = ""  # Can be "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# DAFNE Head
# ---------------------------------------------------------------------------- #
_C.MODEL.DAFNE = CN()

# This is the number of foreground classes.
_C.MODEL.DAFNE.NUM_CLASSES = 15
_C.MODEL.DAFNE.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.DAFNE.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.DAFNE.PRIOR_PROB = 0.01
_C.MODEL.DAFNE.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.DAFNE.INFERENCE_TH_TEST = 0.05
_C.MODEL.DAFNE.NMS_TH = 0.1
_C.MODEL.DAFNE.PRE_NMS_TOPK_TRAIN = 2000
_C.MODEL.DAFNE.PRE_NMS_TOPK_TEST = 2000
_C.MODEL.DAFNE.POST_NMS_TOPK_TRAIN = 1000
_C.MODEL.DAFNE.POST_NMS_TOPK_TEST = 1000
_C.MODEL.DAFNE.TOP_LEVELS = 2
_C.MODEL.DAFNE.NORM = "GN"  # Support GN or none
_C.MODEL.DAFNE.USE_SCALE = True
_C.MODEL.DAFNE.LOSS_SMOOTH_L1_BETA = 1.0 / 9.0  # Smooth L1 loss beta
_C.MODEL.DAFNE.ENABLE_LOSS_MODULATION = True  # Use modulated loss
_C.MODEL.DAFNE.ENABLE_LOSS_LOG = True  # Use modulated loss
_C.MODEL.DAFNE.SORT_CORNERS = True  # Use the canonical representation for corners
_C.MODEL.DAFNE.SORT_CORNERS_DATALOADER = True  # Use the canonical representation for corners
_C.MODEL.DAFNE.CENTERNESS = "oriented"  # "Must be one of ["none", "plain", "oriented"]
_C.MODEL.DAFNE.CENTERNESS_ALPHA = 5  # Smoothing parameter used for pow(ctr', 1/alpha)
_C.MODEL.DAFNE.CENTERNESS_USE_IN_SCORE = True
# Must be one of ["direct", "iterative", "offset", "center-to-corner"]
_C.MODEL.DAFNE.CORNER_PREDICTION = "center-to-corner"
_C.MODEL.DAFNE.CORNER_TOWER_ON_CENTER_TOWER = True
_C.MODEL.DAFNE.MERGE_CORNER_CENTER_PRED = False

# Enable the assignment of different sizes for different feature levels
_C.MODEL.DAFNE.ENABLE_LEVEL_SIZE_FILTERING = True
_C.MODEL.DAFNE.ENABLE_IN_BOX_CHECK = True
_C.MODEL.DAFNE.ENABLE_FPN_STRIDE_NORM = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.DAFNE.THRESH_WITH_CTR = False

# If centereness should be on regression or classification branch (true: regression, false: classification)
_C.MODEL.DAFNE.CTR_ON_REG = True

# Focal loss parameters
_C.MODEL.DAFNE.LOSS_ALPHA = 0.25
_C.MODEL.DAFNE.LOSS_GAMMA = 2.0
_C.MODEL.DAFNE.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.DAFNE.USE_RELU = True
_C.MODEL.DAFNE.USE_DEFORMABLE = False

# Loss lambdas
_C.MODEL.DAFNE.LOSS_LAMBDA = CN()
_C.MODEL.DAFNE.LOSS_LAMBDA_NORM = True  # Normalize lambdas to sum up to 1
_C.MODEL.DAFNE.LOSS_LAMBDA.CORNERS = 1.0
_C.MODEL.DAFNE.LOSS_LAMBDA.BOX = 1.0
_C.MODEL.DAFNE.LOSS_LAMBDA.LTRB = 1.0
_C.MODEL.DAFNE.LOSS_LAMBDA.CTR = 1.0
_C.MODEL.DAFNE.LOSS_LAMBDA.CLS = 1.0
_C.MODEL.DAFNE.LOSS_LAMBDA.CENTER = 1.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.DAFNE.NUM_CLS_CONVS = 4
_C.MODEL.DAFNE.NUM_BOX_CONVS = 4
_C.MODEL.DAFNE.NUM_SHARE_CONVS = 0
_C.MODEL.DAFNE.CENTER_SAMPLE = True
_C.MODEL.DAFNE.CENTER_SAMPLE_ONLY = False
_C.MODEL.DAFNE.COMBINE_CENTER_SAMPLE = True
_C.MODEL.DAFNE.POS_RADIUS = 2.0
_C.MODEL.DAFNE.LOC_LOSS_TYPE = "smoothl1"  # Can be iou, giou, smoothl1
_C.MODEL.DAFNE.YIELD_PROPOSAL = False


# Test Time Augmentation
_C.TEST.AUG.VFLIP = True
_C.TEST.AUG.HFLIP = True
# _C.TEST.AUG.ROTATION_ANGLES = (0, 90, 180, 270)
_C.TEST.AUG.ROTATION_ANGLES = ()
_C.TEST.NUM_PRED_VIS = 20

# IoU Threshold at test time
_C.TEST.IOU_TH = 0.5

# Rotation angles for training augmentation
_C.INPUT.ROTATION_AUG_ANGLES = [0.0, 90.0, 180.0, 270.0]
_C.INPUT.RESIZE_TYPE = "shortest-edge"  # Can be one of ["shortest-edge", "both"]
_C.INPUT.RESIZE_HEIGHT_TRAIN = 0  # Only valid if RESIZE_TYPE=="both"
_C.INPUT.RESIZE_WIDTH_TRAIN = 0
_C.INPUT.RESIZE_HEIGHT_TEST = 0  # Only valid if RESIZE_TYPE=="both"
_C.INPUT.RESIZE_WIDTH_TEST = 0

# Can be one of "choice" or "range"
_C.INPUT.ROTATION_AUG_SAMPLE_STYLE = "choice"

# Enable color augmentation such as saturation, brightness etc
_C.INPUT.USE_COLOR_AUGMENTATIONS = False


_C.MODEL.META_ARCHITECTURE = "OneStageDetector"
_C.MODEL.BACKBONE.NAME = "build_dafne_resnet_fpn_backbone"
_C.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = ["res3", "res4", "res5"]
_C.MODEL.PROPOSAL_GENERATOR.NAME = "DAFNe"

_C.MODEL.DLA = CN()
_C.MODEL.DLA.NORM = "BN"
_C.MODEL.DLA.CONV_BODY = "DLA34"

# If true, dota 1.5 will be loaded with the same classes as dota 1.0
# to allow for training dota 1.0 in conjunction with the 1.5 annotations
_C.DATASETS.DOTA_REMOVE_CONTAINER_CRANE = False
