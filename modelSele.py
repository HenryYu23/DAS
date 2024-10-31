import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from PIL import Image

import sys, os
import time
from modelSeleTools.methodsDirectory2Fast import *


def get_voc_2012_test_images(dirname):
    imageset_file = os.path.join(dirname, "ImageSets/Main/test.txt")
    image_dir = os.path.join(dirname, "JPEGImages")

    with open(imageset_file, "r") as file:
        image_names = file.read().strip().split()[:1000]
    
    dataset_dicts = []
    for idx, name in enumerate(image_names):
        image_path = os.path.join(image_dir, f"{name}.jpg")
        record = {}
        record["file_name"] = image_path
        record["image_id"] = idx
        record["width"], record["height"] = Image.open(image_path).size
        dataset_dicts.append(record)
    
    return dataset_dicts

DatasetCatalog.register("voc_2012_test_images", lambda: get_voc_2012_test_images("./datasets/VOC2012/"))
MetadataCatalog.get("voc_2012_test_images").set(thing_classes=[])

def setup(args):
    """
    create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def calculateMultiFast(cfg, model_dirs: list, method_functions: list, repeat=1):
    if cfg.SEMISUPNET.Trainer == 'ateacher':
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == 'baseline':
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    
    assert args.eval_only, "Only eval-only mode supported."
    assert cfg.SEMISUPNET.Trainer == "ateacher"
    assert len(method_functions) >= 1, "Score function needed."

    # dataloader
    if cfg.DATASETS.TEST[0] == "Clipart1k_test":
        dataloader_names = {"source": "voc_2012_test_images",
                            "target": "Clipart1k_test"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
    elif cfg.DATASETS.TEST[0] == "cityscapes_foggy_val":
        dataloader_names = {"target": "cityscapes_foggy_val",
                            "source": "cityscapes_fine_instance_seg_val"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
        # (optional)
        print("using loaded dataloaders")
        
        import pickle
        # with open("./digits/dataloaders/cityscapes_val_source.pkl", "rb") as f:
        #     dataloader_source = pickle.load(f)
        # with open("./digits/dataloaders/cityscapes_val_target.pkl", "rb") as f:
        #     dataloader_target = pickle.load(f)
    elif cfg.DATASETS.TEST[0] == 'cityval':
        dataloader_names = {"target": "cityval",
                            "source": "sim10kval"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
    else:
        dataloader_source = Trainer.build_test_loader(cfg, "VOC_cityscapes_test")
        dataloader_target = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])

    METHODS_DICT = {"ground_truth": []}
    for method_function in method_functions:
        METHODS_DICT[method_function.__name__] = []
    
    for model_dir in model_dirs:
        # [MAIN]
        for method_function in method_functions:
            # 1 model construction
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)  # whole model with teacher and student.

            # loading parameters
            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(model_dir, resume=args.resume)

            result_dict = method_function(cfg, 
                                          ensem_ts_model.modelTeacher,
                                          [dataloader_source, dataloader_target],
                                          repeat)
            
            for key in list(result_dict.keys()):
                if key not in list(METHODS_DICT.keys()) and key != 'score':
                    METHODS_DICT[key] = [result_dict[key]]
                elif key != "score":
                    METHODS_DICT[key].append(result_dict[key])
                else:
                    METHODS_DICT[method_function.__name__].append(result_dict['score'])

            maxKeyLength = max(len(key) for key in METHODS_DICT.keys())
            print()
            for key in METHODS_DICT.keys():
                print(f"'{key:<{maxKeyLength}}'\t: {METHODS_DICT[key]},")
            time.sleep(5)

    return METHODS_DICT
    


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    assert args.eval_only, "Only eval-only mode supported."
    assert cfg.SEMISUPNET.Trainer == "ateacher"

    model_file = "./outputs/AT_c2f/"
    model_names = [f"model_{str(i).zfill(7)}.pth" for i in range(24999, 100000, 5000)]

    model_dirs = [model_file + model_name for model_name in model_names]
    
    result = calculateMultiFast(cfg, model_dirs, [FIS, PDR])

    return None


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )