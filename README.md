# Awesome Shape Matching [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome spectral shape matching methods, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision/tree/master)

## Why awesome shape matching?

This is a collection of papers and resources I curated when learning how to solve shape matching problem with spectral method. I will be continuously updating this list with the latest papers and resources.

## Papers

Since many works possess multiple attributes simultaneously, they are categorized here based on their primary contributions.

### Survey

- [A Survey on Shape Correspondence](https://www.cs.sfu.ca/~haoz/pubs/vanKaick_cgf11_survey.pdf), van Kaick et al. CGF 2011

- [Registration of 3D point clouds and meshes: A survey from rigid to nonrigid](https://orca.cardiff.ac.uk/id/eprint/47333/1/ROSIN%20registration%20of%203d%20point%20clouds%20and%20meshes.pdf), Tam et al. TVCG 2012

- [Recent advances in shape correspondence](https://user.ceng.metu.edu.tr/~ys/pubs/corsurvey-tvcj20.pdf), Sahillioğlu et al. TVC 2020

- [A Survey of Non-Rigid 3D Registration](https://arxiv.org/pdf/2203.07858), Deng et al. 2022
### Axiomatic Methods

- [Non-rigid registration under isometric deformation](https://geometry.stanford.edu/lgl_2024/papers/hawg-nrrid-08/hawg-nrrid-08.pdf), Huang et al. SGP 2008

- [Topologically-Robust 3D Shape Matching Based on Diffusion Geometry and Seed Growing](https://inria.hal.science/inria-00590280/file/cvpr2011_camera_ready.pdf), Sharma et al. CVPR 2011

- [Geometrically consistent elastic matching of 3d shapes: A linear programming solution](https://frank-r-schmidt.de/Publications/2011/WSSC11/WSSC-iccv11.pdf), Windheuser et al. ICCV 2011 

- [Large-Scale Integer Linear Programming for Orientation Preserving 3D Shape Matching](https://cvg.cit.tum.de/_media/spezial/bib/wssc_sgp11.pdf), Windheuser et al. SGP 2011

- **[FMaps]** [Functional Maps: A Flexible Representation of Maps Between Shapes](https://people.csail.mit.edu/jsolomon/assets/fmaps.pdf), Ovsjanikov et a
l. ACM TOG 2012

- **[CQHB]** [Coupled quasi-harmonic bases](https://arxiv.org/pdf/1210.0026), Kovnatsky et al. EG 2013

- [Image Co-Segmentation via Consistent Functional Maps](https://www.cs.utexas.edu/~huangqx/whg-icsfm-13.pdf), Wang et al. ICCV 2013

- **[CFM]** [Coupled Functional Maps](https://cvg.cit.tum.de/_media/spezial/bib/eynard-3dv16.pdf), Eynard et al. 3DV 2016

- **[PMSDP]** [Point Registration via Efficient Convex Relaxation](https://shaharkov.github.io/projects/ProcrustesMatchingSDP_lowres.pdf), Maron et al. SIGGRAPH 2016 | [Code](https://github.com/Haggaim/PM-SDP)

- **[AMRSAM]** [Adjoint Map Representation for Shape Analysis and Matching](https://www.lix.polytechnique.fr/~maks/papers/adjoint_map_paper.pdf), Huang et al. SGP 2017

- [Deblurring and Denoising of Maps between Shapes](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.13254), Ezuz et al. SGP 2017
- [Informative Descriptor Preservation via Commutativity for Shape Matching](https://www.lix.polytechnique.fr/Labo/Ovsjanikov.Maks/papers/fundescEG17.pdf), Nogneng et al. EG 2017

- **[PMF]** [Product Manifold Filter: Non-Rigid Shape Correspondence via Kernel Density Estimation in the Product Space](https://arxiv.org/pdf/1701.00669), Vestner et al. CVPR 2017

- [Spatial Maps: From Low Rank Spectral to Sparse Spatial Functional Representations](https://cvg.cit.tum.de/_media/spezial/bib/gasparetto-3dv17.pdf), Gasparetto et al. 3DV 2017

- [Efficient Deformable Shape Correspondence via Kernel Matching](https://arxiv.org/pdf/1707.08991), Vestner et al. 3DV 2017
- **[BCICP]** [Continuous and Orientation-preserving Correspondences via Functional Maps](https://dl.acm.org/doi/pdf/10.1145/3272127.3275040), Ren et al. ACM TOG 2018 | [Code](https://github.com/llorz/SGA18_orientation_BCICP_code) 

- **[ZoomOut]** [ZoomOut: Spectral Upsampling for Efficient Shape Correspondence](https://arxiv.org/pdf/1904.07865), Melzi et al. 2019 ACM TOG | [Code](https://github.com/llorz/SGA19_zoomOut)

- [Reversible Harmonic Maps between Discrete Surfaces](https://arxiv.org/pdf/1801.02453), Ezuz et al. ACM TOG 2019 

- **[HFM]** [Hierarchical Functional Maps between Subdivision Surfaces](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.13789?saml_referrer), Shoham et al. SGP 2019

- [Elastic Correspondence between Triangle Meshes](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.13624), Ezuz et al. CGF 2019

- [Structured Regularization of Functional Map Computations](https://arxiv.org/pdf/2009.14624), Ren et al. SGP 2019
- **[Smooth Shells]** [Smooth Shells: Multi-Scale Shape Registration with Functional Maps](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eisenberger_Smooth_Shells_Multi-Scale_Shape_Registration_With_Functional_Maps_CVPR_2020_paper.pdf), Eisenberger et al. CVPR 2020 **(Oral)** | [Code](https://github.com/marvin-eisenberger/smooth-shells)

- **[MINA]** [MINA: Convex Mixed-Integer Programming for Non-Rigid Shape Alignment](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bernard_MINA_Convex_Mixed-Integer_Programming_for_Non-Rigid_Shape_Alignment_CVPR_2020_paper.pdf), Bernard et al. CVPR 2020 

- **[DiscrteOP]** [Discrete Optimization for Shape Matching](https://www.lix.polytechnique.fr/~maks/papers/SGP21_DiscMapOpt.pdf), Ren et al. SGP 2021 | [Code](https://github.com/llorz/SGP21_discreteOptimization) | [Pytorch](https://github.com/RobinMagnet/SmoothFunctionalMaps)

- **[MWP]** [Efficient deformable shape correspondence via multiscale spectral manifold wavelets preservation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Efficient_Deformable_Shape_Correspondence_via_Multiscale_Spectral_Manifold_Wavelets_Preservation_CVPR_2021_paper.pdf), Hu et al. CVPR 2021 

- [Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid ShapeCorrespondence with Functional Maps](https://openaccess.thecvf.com/content/CVPR2021/papers/Pai_Fast_Sinkhorn_Filters_Using_Matrix_Scaling_for_Non-Rigid_Shape_Correspondence_CVPR_2021_paper.pdf), Pai et al. CVPR 2021 | [Code](https://github.com/paigautam/CVPR21_FastSinkhornFilters)

- **[SmoothFM]** [Smooth Non-Rigid Shape Matching via Effective Dirichlet Energy Optimization](https://arxiv.org/pdf/2210.02870), Magnet et al. 3DV 2022 | [Code](https://github.com/RobinMagnet/SmoothFunctionalMaps)

- **[ComplexFM]** [Complex Functional Maps : a Conformal Link Between Tangent Bundles](https://arxiv.org/pdf/2112.09546), Donati et al. EG 2022 | [Code](https://github.com/nicolasdonati/QMaps)

- **[SM-Comb]** [A Scalable Combinatorial Solver for Elastic Geometrically Consistent 3D Shape Matching](https://arxiv.org/pdf/2204.12805), Roetzer et al. CVPR 2022 | [Code](https://github.com/paul0noah/sm-comb)

- **[Scalable ZoomOut]** [Scalable and Efficient Functional Map Computations on Dense Meshes](https://hal.science/hal-04352328/file/ScalableFmaps_EG2023_cmp%20%281%29.pdf), Magnet et al. EG 2023 | [Code](https://github.com/RobinMagnet/Scalable_FM)

- **[SIGMA]** [ΣIGMA: Scale-Invariant Global Sparse Shape Matching](https://cvg.cit.tum.de/_media/members/gaom/sigma-6kb.pdf), Gao et al. ICCV 2023 | [Code](https://github.com/maolingao/SIGMA)

- **[Elastic Basis]** [An Elastic Basis for Spectral Shape Correspondence](https://dl.acm.org/doi/pdf/10.1145/3588432.3591518), Hartwig et al. SIGGRAPH 2023 | [Code](https://github.com/flrneha/ElasticBasisForSpectralMatching?tab=readme-ov-file)

### Learning-based Methods

- **[FMNet]** [Deep Functional Maps: Structured Prediction for Dense Shape Correspondence](https://cvg.cit.tum.de/_media/spezial/bib/litany-iccv17.pdf), Litany et al. ICCV 2017

- **[UnsupFMNet]** [Unsupervised Learning of Dense Shape Correspondence](https://openaccess.thecvf.com/content_CVPR_2019/papers/Halimi_Unsupervised_Learning_of_Dense_Shape_Correspondence_CVPR_2019_paper.pdf), Halimi et al. CVPR 2019 **(Oral)** | [Code](https://github.com/OshriHalimi/unsupervised_learning_of_dense_shape_correspondence)

- **[SURFMNet]** [Unsupervised Deep Learning for Structured Shape Matching](https://openaccess.thecvf.com/content_ICCV_2019/papers/Roufosse_Unsupervised_Deep_Learning_for_Structured_Shape_Matching_ICCV_2019_paper.pdf), Roufosse et al. ICCV 2019 | [Code](https://github.com/LIX-shape-analysis/SURFMNet)

- [Cyclic Functional Mapping: Self-supervised correspondence between non-isometric deformable shapes](https://arxiv.org/pdf/1912.01249), Ginzburg et al. ECCV 2020

- **[GeomFmaps]** [Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence](https://openaccess.thecvf.com/content_CVPR_2020/papers/Donati_Deep_Geometric_Functional_Maps_Robust_Feature_Learning_for_Shape_Correspondence_CVPR_2020_paper.pdf), Donati et al. CVPR 2020 | [Code](https://github.com/LIX-shape-analysis/GeomFmaps)

- **[WSupFMNet]** [Weakly Supervised Deep Functional Map for Shape Matching](https://proceedings.neurips.cc/paper/2020/file/dfb84a11f431c62436cfb760e30a34fe-Paper.pdf), Sharma et al. NeurIPS 2020 | [Code](https://github.com/bach-zouk/Weakly-supervised-Deep-Functional-map)

- **[Deep Shells]** [Deep Shells: Unsupervised Shape Correspondence with Optimal Transport](https://proceedings.neurips.cc/paper_files/paper/2020/file/769c3bce651ce5feaa01ce3b75986420-Paper.pdf), Eisenberger et al. NeurIPS 2020 | [Code](https://github.com/marvin-eisenberger/deep-shells)

- [Spectral Shape Recovery and Analysis Via Data-driven Connections](https://link.springer.com/content/pdf/10.1007/s11263-021-01492-6.pdf), Marin et al. IJCV 2021


- **[NCP]** [NCP: Neural Correspondence Prior for Effective Unsupervised Shape Matching](https://proceedings.neurips.cc/paper_files/paper/2022/file/b95c7e24501f5d1dddbc5e8526cda7ae-Paper-Conference.pdf), Attaiki et al. NeurIPS 2022 | [Code](https://github.com/pvnieo/NCP)

- **[SRFeat]** [SRFeat: Learning Locally Accurate and Globally Consistent
Non-Rigid Shape Correspondence](https://arxiv.org/pdf/2209.07806), Li et al. 3DV 2022

- **[AttentiveFMaps]** [Learning Multi-resolution Functional Maps with Spectral Attention for Robust Shape Matching](https://arxiv.org/pdf/2210.06373), Li et al. NeurIPS 2022 | [Code](https://github.com/craigleili/AttentiveFMaps)

- **[DUO-FMNet]** [Deep Orientation-Aware Functional Maps: Tackling Symmetry Issues in Shape Matching](https://www.lix.polytechnique.fr/~maks/papers/CVPR22_DeepCompFmaps.pdf), Donati et al. CVPR 2022 | [Code](https://github.com/nicolasdonati/DUO-FM)

- **[IFMatch]** [Implicit Field Supervision For Robust Non-Rigid Shape Matching](https://arxiv.org/pdf/2203.07694), Sundararaman et al. CVPR 2022 | [Code](https://github.com/Sentient07/IFMatch)

- **[ULRSSM]** [Unsupervised Learning of Robust Spectral Shape Matching](https://dongliangcao.github.io/assets/pdf/dongliang2023siggraph.pdf), Cao et al. ACM TOG 2023 | [Code](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)

- **[Differentiable ZoomOut]** [Memory-Scalable and Simplified Functional Map Learning](https://arxiv.org/pdf/2404.00330), Magnet et al. CVPR 2024 | [Code](https://github.com/RobinMagnet/SimplifiedFmapsLearning)

- [Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching](https://arxiv.org/pdf/2312.03678), Bastian et al. CVPR 2024 | [Code](https://github.com/xieyizheng/hybridfmaps)

- **[SpiderMatch]** [SpiderMatch: 3D Shape Matching with Global Optimality and Geometric Consistency](https://openaccess.thecvf.com/content/CVPR2024/papers/Roetzer_SpiderMatch_3D_Shape_Matching_with_Global_Optimality_and_Geometric_Consistency_CVPR_2024_paper.pdf), Roetzer et al. CVPR 2024 **(Best Student Paper Runner-Up)** | [Code](https://github.com/paul0noah/spider-match?tab=readme-ov-file)


- [Revisiting Map Relations for Unsupervised Non-Rigid Shape Matching](https://dongliangcao.github.io/assets/pdf/dongliang20243dv.pdf), Cao et al. 3DV 2024

- [Synchronous Diffusion for Unsupervised Smooth Non-rigid 3D Shape Matching](https://dongliangcao.github.io/assets/pdf/dongliang2024eccv.pdf), Cao et al. ECCV 2024 

- **[DiscoMatch]** [DiscoMatch: Fast Discrete Optimisation for Geometrically Consistent 3D Shape Matching](https://arxiv.org/pdf/2310.08230), Roetzer et al. ECCV 2024 | [Code](https://github.com/paul0noah/disco-match)

### Partial Shape Matching

- **[FSP]** [Fully Spectral Partial Shape Matching](https://cvg.cit.tum.de/_media/spezial/bib/litany-eg17.pdf), Litany et al. EG 2017


- **[PSM]** [Partial Functional Correspondence](https://arxiv.org/pdf/1506.05274), Rodola et al. CGF 2017

- **[DIR]** [A Dual Iterative Refinement Method for Non-rigid Shape Matching](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiang_A_Dual_Iterative_Refinement_Method_for_Non-Rigid_Shape_Matching_CVPR_2021_paper.pdf), Xiang et al. CVPR 2021 | [Code](https://github.com/ruixiang440/Dual_Iterative_Refinement_Method)

- **[DPFM]** [DPFM: Deep Partial Functional Maps](https://arxiv.org/pdf/2110.09994), Attaiki et al. 3DV 2021 **(Best Paper)** | [Code](https://github.com/pvnieo/DPFM)

- [Learning Spectral Unions of Partial Deformable 3D Shapes](https://arxiv.org/pdf/2104.00514), Moschella et al. EG 2022 | [Code](https://github.com/lucmos/spectral-unions)

- [Geometrically Consistent Partial Shape Matching](https://arxiv.org/pdf/2309.05013), Ehm et al. 3DV 2024

- [Partial-to-Partial Shape Matching with Geometric Consistency](https://openaccess.thecvf.com/content/CVPR2024/papers/Ehm_Partial-to-Partial_Shape_Matching_with_Geometric_Consistency_CVPR_2024_paper.pdf), Ehm et al. CVPR 2024 | [Code](https://github.com/vikiehm/gc-ppsm)

### Shape Interpolation
- [Divergence-Free Shape Correspondence by Deformation](https://cvg.cit.tum.de/_media/spezial/bib/eisenberger2019divfree.pdf), Eisenberger et al. SGP 2019 | [Code](https://github.com/marvin-eisenberger/hamiltonian-interpolation)

- [Hamiltonian Dynamics for Real-World Shape Interpolation](https://arxiv.org/pdf/2004.05199), Eisenberger et al. ECCV 2020 **(Spotlight)** | [Code](https://github.com/marvin-eisenberger/hamiltonian-interpolation)

- **[Neuromorph]** [NeuroMorph: Unsupervised Shape Interpolation and Correspondence in One Go](https://arxiv.org/pdf/2106.09431), Eisenberger et al. CVPR 2021 | [Code](https://github.com/facebookresearch/neuromorph)

- [Spectral Meets Spatial: Harmonising 3D Shape Matching and Interpolation](https://dongliangcao.github.io/assets/pdf/dongliang2024cvpr.pdf), Cao et al. CVPR 2024 | [Code](https://github.com/dongliangcao/Spectral-Meets-Spatial)

- **[SRIF]** [SRIF: Semantic Shape Registration Empowered by Diffusion-based Image Morphing and Flow Estimation](https://dl.acm.org/doi/pdf/10.1145/3680528.3687567), Sun et al. SIGGRAPH Asia 2024 | [Code](https://github.com/rqhuang88/SRIF)
- [Implicit Neural Surface Deformation with Explicit Velocity Fields](https://arxiv.org/pdf/2501.14038), Sang et al. ICLR 2025 | [Code](https://github.com/Sangluisme/Implicit-surf-Deformation)

### Pointcloud Matching

- **[3D-CODED]** [3D-CODED : 3D Correspondences by Deep
Deformation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Thibault_Groueix_Shape_correspondences_from_ECCV_2018_paper.pdf), Groueix et al. ECCV 2018 | [Code](https://github.com/ThibaultGROUEIX/3D-CODED)

- **[Elementary]** [Learning elementary structures for 3D shape generation and matching](https://arxiv.org/pdf/1908.04725), Deprelle et al. NeurIPS 2019 

- **[LIE]** [Correspondence Learning via Linearly-invariant Embedding](https://proceedings.neurips.cc/paper_files/paper/2020/file/11953163dd7fb12669b41a48f78a29b6-Paper.pdf), Marin et al. NeurIPS 2020 | [Code](https://github.com/riccardomarin/Diff-FMaps) | [Pytorch](https://github.com/riccardomarin/Diff-FMAPs-PyTorch)

- **[CorrNet3D]** [CorrNet3D: Unsupervised End-to-end Learning of Dense Correspondence
for 3D Point Clouds](https://openaccess.thecvf.com/content/CVPR2021/papers/Zeng_CorrNet3D_Unsupervised_End-to-End_Learning_of_Dense_Correspondence_for_3D_Point_CVPR_2021_paper.pdf), Zeng et al. CVPR 2021 | [Code](https://github.com/ZENGYIMING-EAMON/CorrNet3D)

- **[DPC]** [DPC: Unsupervised Deep Point Correspondence via Cross and Self Constructio](https://arxiv.org/pdf/2110.08636), Lang et al. 3DV 2021 | [Code](https://github.com/dvirginz/DPC)
- **[NIE]** [Neural Intrinsic Embedding for Non-rigid Point Cloud Matching](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_Neural_Intrinsic_Embedding_for_Non-Rigid_Point_Cloud_Matching_CVPR_2023_paper.pdf), Jiang et al. CVPR 2023 | [Code](https://github.com/rqhuang88/Neural-Intrinsic-Embedding)

- **[SSMSM]** [Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Self-Supervised_Learning_for_Multimodal_Non-Rigid_3D_Shape_Matching_CVPR_2023_paper.pdf), Cao et al. CVPR 2023 **(Spotlight)** | [Code](https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching)

- **[DFR]** [Non-Rigid Shape Registration via Deep Functional Maps Prior](https://arxiv.org/pdf/2311.04494) Jiang et al. NeurIPS 2023 | [Code](https://github.com/rqhuang88/DFR)

- **[COE]** [CoE: Deep Coupled Embedding for Non-Rigid Point Cloud Correspondences](https://arxiv.org/pdf/2412.05557) Zeng et al. 3DV 2025 | [Code](https://github.com/zenghjian/coe?tab=readme-ov-file)

### Shape Collections

- [Consistent Shape Maps via Semidefinite Programming](https://ptolemy.berkeley.edu/projects/robotics/pubs/9/hg-csmsdp-13.pdf), Huang et al. SGP 2013

- [Functional Map Networks for Analyzing and Exploring Large Shape Collections](https://graphics.stanford.edu/courses/cs233-18-spring/ReferencedPapers/hwg-fmnaelsc-14.pdf), Huang et al. ACM TOG 2014

- [Unsupervised cycle-consistent deformation for shape matching](https://arxiv.org/pdf/1907.03165), Groueix et al. SGP 2019

- **[Consistent ZoomOut]** [CONSISTENT ZOOMOUT: Efficient Spectral Map Synchronization](https://www.lix.polytechnique.fr/~maks/papers/ConsistentZoomOut_SGP2020_compressed.pdf), Huang et al. SGP 2020 


- **[IsoMuSh]** [Isometric Multi-Shape Matching](https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Isometric_Multi-Shape_Matching_CVPR_2021_paper.pdf), Gao et al. CVPR 2021 **(Oral)** | [Code](https://github.com/maolingao/IsoMuSh)

- **[UDMSM]** [Unsupervised deep multi-shape matching](https://arxiv.org/pdf/2207.09610), Cao et al. ECCV 2022 | [Code](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching)

- **[SyNoRiM]** [Multiway Non-rigid Point Cloud Registration via
Learned Functional Map Synchronization](https://arxiv.org/pdf/2111.12878), Huang et al. TPAMI 2022 | [Code](https://github.com/huangjh-pub/synorim)

- **[G-MSM]** [G-MSM: Unsupervised Multi-Shape Matching with Graph-Based Affinity Priors](https://openaccess.thecvf.com/content/CVPR2023/papers/Eisenberger_G-MSM_Unsupervised_Multi-Shape_Matching_With_Graph-Based_Affinity_Priors_CVPR_2023_paper.pdf), Eisenberger et al. CVPR 2023 | [Code](https://github.com/marvin-eisenberger/gmsm-matching)

- **[SSCDFM]** [Spatially and Spectrally Consistent Deep Functional Maps](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially_and_Spectrally_Consistent_Deep_Functional_Maps_ICCV_2023_paper.pdf), Sun et al. ICCV 2023 | [Code](https://github.com/rqhuang88/Spatially-and-Spectrally-Consistent-Deep-Functional-Maps)

- [Unsupervised Representation Learning for Diverse Deformable Shape Collections](https://arxiv.org/pdf/2310.18141), Hahner et al. 3DV 2024 | [Code](https://github.com/Fraunhofer-SCAI/DISCO-AE)
### Zero-Shot Correspondence

- **[SNK]** [Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction](https://arxiv.org/pdf/2403.06804), Attaiki et al. NeurIPS 2023 | [Code](https://github.com/pvnieo/SNK)

- **[SATR]** [SATR: Zero-Shot Semantic Segmentation of 3D Shapes](https://arxiv.org/pdf/2304.04909), Abdelreheem et al. ICCV 2023 | [Code](https://github.com/Samir55/SATR)

- [Zero-Shot 3D Shape Correspondence](https://samir55.github.io/3dshapematch/assets/2306.03253.pdf), Abdelreheem et al. SIGGRAPH Asia 2023

## Feature Backbone

- **[PointNet++]** [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413), Qi et al. NeurIPS 2017 | [Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

- **[DGCNN]** [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829), Wang et al. ACM TOG 2018 | [Pytorch](https://github.com/antao97/dgcnn.pytorch)
- **[ACSCNN]** [Shape correspondence using anisotropic Chebyshev spectral CNNs](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Shape_correspondence_using_anisotropic_Chebyshev_spectral_CNNs_CVPR_2020_paper.pdf) Li et al. CVPR 2020 

- **[HSN]** [CNNs on Surfaces using Rotation-Equivariant Features](https://arxiv.org/pdf/2006.01570), Wiersma et al. ACM TOG 2020

- **[DiffusionNet]** [DiffusionNet: Discretization Agnostic Learning on Surfaces](https://arxiv.org/pdf/2012.00888), Sharp et al. ACM TOG 2021 | [Code](https://github.com/nmwsharp/diffusion-net)

- [Understanding and Improving Features Learned in Deep Functional Maps](https://arxiv.org/pdf/2303.16527), Attaiki et al. CVPR 2023 | [Code](https://github.com/pvnieo/clover)
## Dataset
- **[SCAPE]** [SCAPE: Shape Completion and Animation of People](https://ai.stanford.edu/~drago/Papers/shapecomp.pdf), Anguelov et al. SIGGRAPH 2005 | [Website](http://ai.stanford.edu/~drago/Projects/scape/)

- **[FAUST]** [FAUST: Dataset and Evaluation for 3D Mesh Registration](https://files.is.tue.mpg.de/black/papers/FAUST2014.pdf), Bogo et al. CVPR 2014 | [Website](http://faust.is.tue.mpg.de/)

- **[SURREAL]** [SURREAL: Open-Source Reinforcement Learning Framework and Robot Manipulation Benchmark](https://arxiv.org/pdf/1701.01370), Varol et al. ECCV 2018 | [Website](https://www.di.ens.fr/willow/research/surreal/)

- **[SMAL]** [3D Menagerie: Modeling the 3D Shape and Pose of Animals](https://files.is.tue.mpg.de/black/papers/smal_cvpr_2017.pdf), Zuffi et al. CVPR 2017 | [Website](https://smal.is.tue.mpg.de/)

- **[TOPKIDS]** [SHREC’16: Matching of Deformable Shapes with Topological Noise](https://zorah.github.io/files/pdfs/laehner2016topkids.pdf), Lähner et al. | [Website](https://cvg.cit.tum.de/data/datasets/topkids)

- **[SHREC16]** [SHREC’16: Partial Matching of Deformable Shapes](https://cvg.cit.tum.de/_media/spezial/bib/shrec16-partial.pdf), Cosmo et al. | [Website](https://www.dais.unive.it/~shrec2016/)

- **[SHREC19]** [SHREC 2019: Matching Humans with Different Connectivity](https://diglib.eg.org/bitstream/handle/10.2312/3dor20191070/121-128.pdf), Melzi et al. | [Website](https://profs.scienze.univr.it/marin/shrec19/#References)

- **[SHREC20]** [SHREC’20: Shape correspondence with non-isometric deformations](https://orca.cardiff.ac.uk/id/eprint/134584/7/1-s2.0-S0097849320301266-main.pdf), Dyke et al. | [Website](http://robertodyke.com/shrec2020/index2.html
)
- **[DT4D]** [4DComplete: Non-Rigid Motion Estimation Beyond the Observable Surface](https://arxiv.org/pdf/2105.01905), Li et al. ICCV 2021 | [Website](https://github.com/rabbityl/DeformingThings4D)

TODO: Collect dataset with anonymous link, include various version (remeshed, anisotropic etc.)
## courses

- [Computing and Processing Correspondences with Functional Maps](https://www.lix.polytechnique.fr/~maks/fmaps_course/index.html), Maks Ovsjanikov et al. SIGGRAPH Asia 2016

TODO: Add more resource (public course, talk etc.)

## Tools

TODO: Add open-source toolbox for geometry processing and shape analysis.