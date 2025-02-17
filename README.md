# Awesome Functional Maps [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome spectral shape matching methods, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision/tree/master)

## Why awesome functional maps?

This is a collection of papers and resources I curated when learning how to solve shape matching problem with spectral method. I will be continuously updating this list with the latest papers and resources.

## Papers

Since many works possess multiple attributes simultaneously, they are categorized here based on their primary contributions.

### Axiomatic Methods

- [Geometrically consistent elastic matching of 3d shapes: A linear programming solution](https://frank-r-schmidt.de/Publications/2011/WSSC11/WSSC-iccv11.pdf), Windheuser et al. ICCV 2011 

- **[FMaps]** [Functional Maps: A Flexible Representation of Maps Between Shapes](https://www.cs.princeton.edu/~mmerrell/functional_maps.pdf), Ovsjanikov et al. ACM TOG 2012

- **[BCICP]** [Continuous and orientation-preserving correspondences via functional maps](https://dl.acm.org/doi/pdf/10.1145/3272127.3275040), Ren et al. ACM TOG 2018 | [Code](https://github.com/llorz/SGA18_orientation_BCICP_code) 
- **[ZoomOut]** [Zoomout: Spectral upsampling for efficient shape correspondence](https://arxiv.org/pdf/1904.07865), Melzi et al. 2019 ACM TOG | [Code](https://github.com/llorz/SGA19_zoomOut)

- **[Smooth Shells]** [Smooth Shells: Multi-Scale Shape Registration with Functional Maps](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eisenberger_Smooth_Shells_Multi-Scale_Shape_Registration_With_Functional_Maps_CVPR_2020_paper.pdf), Eisenberger et al. CVPR 2020

- **[DiscrteOP]** [Discrete optimization for shape matching](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14200), Ren et al. CGF 2021 | [Code](https://github.com/llorz/SGP21_discreteOptimization)

### Learning-based Methods

- **[FMNet]** [Deep Functional Maps: Structured Prediction for Dense Shape Correspondence](https://cvg.cit.tum.de/_media/spezial/bib/litany-iccv17.pdf), Litany et al. ICCV 2017

- **[UnsupFMNet]** [Unsupervised learning of dense shape correspondence](https://openaccess.thecvf.com/content_CVPR_2019/papers/Halimi_Unsupervised_Learning_of_Dense_Shape_Correspondence_CVPR_2019_paper.pdf), Halimi et al. CVPR 2019 **(Oral)** | [Code](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence)

- **[GeomFmaps]** [Deep geometric functional maps: Robust feature learning for shape correspondence](https://openaccess.thecvf.com/content_CVPR_2020/papers/Donati_Deep_Geometric_Functional_Maps_Robust_Feature_Learning_for_Shape_Correspondence_CVPR_2020_paper.pdf), Donati et al. CVPR 2020 | [Code](https://github.com/LIX-shape-analysis/GeomFmaps)

- **[WSupFMNet]** [Weakly supervised deep functional maps for shape matching](https://proceedings.neurips.cc/paper/2020/file/dfb84a11f431c62436cfb760e30a34fe-Paper.pdf), Sharma et al. NeurIPS 2020 | [Code](https://github.com/bach-zouk/Weakly-supervised-Deep-Functional-map)

- **[DeepShells]** [Deep shells: Unsupervised shape correspondence with optimal transport](https://proceedings.neurips.cc/paper_files/paper/2020/file/769c3bce651ce5feaa01ce3b75986420-Paper.pdf), Eisenberger et al. NeurIPS 2020 | [Code](https://github.com/marvin-eisenberger/deep-shells)

- **[NCP]** [NCP: Neural Correspondence Prior for Effective Unsupervised Shape Matching](https://proceedings.neurips.cc/paper_files/paper/2022/file/b95c7e24501f5d1dddbc5e8526cda7ae-Paper-Conference.pdf), Attaiki et al. NeurIPS 2022 | [Code](https://github.com/pvnieo/NCP)

- **[AttentiveFMaps]** [Learning multi-resolution functional maps with spectral attention for robust shape matching](https://arxiv.org/pdf/2110.09994), Li et al. NeurIPS 2022 | [Code](https://github.com/craigleili/AttentiveFMaps)

- **[DUO-FMNet]** [Deep Orientation-Aware Functional Maps: Tackling Symmetry Issues in Shape Matching](https://www.lix.polytechnique.fr/~maks/papers/CVPR22_DeepCompFmaps.pdf), Donati et al. CVPR 2022 | [Code](https://github.com/nicolasdonati/DUO-FM)

- **[ConsistFMaps]** [Unsupervised deep multi-shape matching](https://arxiv.org/pdf/2203.13907), Cao et al. ECCV 2022 | [Code](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching)

- **[ULRSSM]** [Unsupervised learning of robust spectral shape matching](https://dongliangcao.github.io/assets/pdf/dongliang2023siggraph.pdf), Cao et al. ACM TOG 2023 | [Code](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)


### Partial Shape Matching

- [Fully spectral partial shape matching](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13117), Litany et al. CGF 2017


- **[PSM]** [Partial functional correspondence](https://arxiv.org/pdf/1506.05274), Rodola et al. CGF 2017

- **[DIR]** [A dual iterative refinement method for non-rigid shape matching](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiang_A_Dual_Iterative_Refinement_Method_for_Non-Rigid_Shape_Matching_CVPR_2021_paper.pdf), Xiang et al. CVPR 2021 | [Code](https://github.com/ruixiang440/Dual_Iterative_Refinement_Method)

- **[DPFM]** [DPFM: Deep Partial Functional Maps](https://arxiv.org/pdf/2110.09994), Attaiki et al. 3DV 2021 **(Best Paper)** | [Code](https://github.com/pvnieo/DPFM)

- [Geometrically consistent partial shape matching](https://arxiv.org/pdf/2309.05013), Ehm et al. 3DV 2023

- [Partial-to-Partial Shape Matching with Geometric Consistency](https://openaccess.thecvf.com/content/CVPR2024/papers/Ehm_Partial-to-Partial_Shape_Matching_with_Geometric_Consistency_CVPR_2024_paper.pdf), Ehm et al. CVPR 2024 | [Code](https://github.com/vikiehm/gc-ppsm)

### Shape Interpolation

- **[Neuromorph]** [Neuromorph: Unsupervised shape interpolation and correspondence in one go](https://openaccess.thecvf.com/content/CVPR2021/papers/Eisenberger_Neuromorph_Unsupervised_Shape_Interpolation_and_Correspondence_in_One_Go_CVPR_2021_paper.pdf), Eisenberger et al. CVPR 2021 | [Code](https://github.com/facebookresearch/neuromorph)

### Pointcloud Matching


## Dataset
- **[SCAPE]** [SCAPE: Shape Completion and Animation of People](https://robots.stanford.edu/papers/anguelov.shapecomp.pdf), Anguelov et al. SIGGRAPH 2005

- **[FAUST]** [FAUST: Dataset and Evaluation for 3D Mesh Registration](http://faust.is.tue.mpg.de/), Bogo et al. CVPR 2014

- **[SURREAL]** [SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark](https://www.di.ens.fr/willow/research/surreal/), Varol et al. ECCV 2018

