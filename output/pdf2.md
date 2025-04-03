# A Comprehensive Benchmark for RNA 3D Structure-Function Modeling

Luis Wyss * 1 Vincent Mallet * 2 Wissam Karroucha 2 Karsten Borgwardt †1 Carlos Oliver †1 3

## Abstract

The RNA structure-function relationship has recently garnered significant attention within the deep learning community, promising to grow in importance as nucleic acid structure models advance. However, the absence of standardized and accessible benchmarks for deep learning on RNA 3D structures has impeded the development of models for RNA functional characteristics.

In this work, we introduce a set of seven benchmarking datasets for RNA structure-function prediction, designed to address this gap. Our library builds on the established Python library rnaglib, and offers easy data distribution and encoding, splitters and evaluation methods, providing a convenient all-in-one framework for comparing models. Datasets are implemented in a fully modular and reproducible manner, facilitating for community contributions and customization. Finally, we provide initial baseline results for all tasks using a graph neural network.

Source code: github.com/cgoliver/rnaglib
Documentation: rnaglib.org

## 1. Introduction

Recent years have witnessed the advent of deep learning methods for structural biology culminating in the award of the Nobel Prize in Chemistry. AlphaFold (Jumper et al., 2021) revolutionized protein structure prediction, equipping the field with millions of new structures. Breakthroughs go beyond structure prediction, notably in protein design (Watson et al., 2023; Dauparas et al., 2022), drug discovery (Schneuing et al., 2024; Corso et al., 2022) or fundamental biology (van Kempen et al., 2022). While it is tempting to attribute the success of these methods to the increase in available structural data caused by AlphaFold, most of the methods are actually not reliant on them. Instead, it seems that these breakthroughs result from progress in training neural encoders that directly model protein structures (Jing et al., 2020; Zhang et al., 2022b; Gainza et al., 2020; Wang et al., 2022). This progress is in turn rooted in solid competitions (CASP, CAPRI), and benchmarks (Townshend et al., 2021a; Kucera et al., 2023; Zhu et al., 2022; Jamasb et al., 2024; Notin et al., 2023). By setting clear goals, such benchmarks are the foundation for the development of structure encoders. Yet to date, structure-function benchmarks have focused on proteins.

Ribonucleic acids (RNAs) are a large family of molecules which support biological functions along every branch of the tree of life. Besides messenger RNAs, non-coding RNAs carry out biological functions by adopting complex 3D folds (Cech & Steitz, 2014) like proteins do and take up diverse roles in cellular functions, including gene regulation, RNA processing, and protein synthesis (Statello et al., 2021). However, our understanding of non-coding RNAs and their functions remains limited. This can be largely attributed to the negatively charged nature of RNA backbones, which makes it flexible and limits the availability of high-resolution RNA structures, and imposes significant modeling challenges. Another predominant challenge to a functional understanding of RNA 3D structure lies in the lack of infrastructure for the development and evaluation of function prediction models. In this work, we propose a benchmarking suite to act as this facilitating framework.

Our key contributions include:

- Seven tasks related to RNA 3D structure that represent various biological challenges. Each task consists of a dataset, a splitting strategy, and an evaluation method, laying the ground for comparable, reproducible model development.

- End-to-end reproducible and modular access to task data. Modular annotators, filters and splitting strategies, both novel and from existing literature, facilitate the addition of new tasks by other researchers across fields.

*Equal contribution †Equal supervision 1Max Planck Institute of Biochemistry, Munich, Germany 2Mines Paris, PSL Research University, CBIO, Paris, France 3Vanderbilt University, Nashville, Tennessee, USA. Correspondence to: Luis Wyss <wyss@biochem.mpg.de>.

A preprint.

Benchmark for RNA 3D Structure Modeling

## 2. Related Work

### 2.1. Protein Focused Benchmarking

Classic tasks on proteins appeared independently in unrelated papers, such as GO term and EC number prediction (Gligorijevic et al., 2021), fold classification (Hou et al., 2018), binding site detection and classification (Gainza et al., 2020) or binding affinity regression (Wang et al., 2005a). To our knowledge, ATOM3D (Townshend et al., 2021a) was the first systematic benchmark for molecular systems, albeit heavily focused on proteins. Similar, more comprehensive tools were then proposed, such as ProteinShake (Kucera et al., 2023) ProteinWorkshop (Jamasb et al., 2024) and TorchDrug (Zhu et al., 2022), that unify the above tasks and lower the barrier to develop protein structure encoders. ProteinGym (Notin et al., 2023) addresses the evaluation of protein mutation effects, while Buttenschoen et al. (2024); Kovtun et al. (2024); Durairaj et al. (2024) address protein interactions. These works contain notable efforts to scale datasets using predicted structures and to propose biologically appropriate splitting strategies.

### 2.2. RNA Structural Datasets and Benchmarking

In the realm of RNA 3D structure based modeling infrastructure, three papers propose cleaned datasets with the objective of facilitating machine learning. RNANet (Becquey et al., 2021) proposes a dataset joining RNA structures with their corresponding sequence alignment. RNAsolo (Adamczyk et al., 2022) provides access to cleaned RNA files in various formats through a web interface. Finally, RNA3DB (Szikszai et al., 2024) offers a curated dataset specifically designed for RNA structure prediction models. None of these methods propose benchmark tasks to compare RNA modeling and learning methods. Moreover, some datasets along with splits were independently proposed in several work, pertaining to small-molecules binding sites (Wang et al., 2018), or virtual screening (Panei et al., 2022; Carvajal-Patiño et al., 2025). The inverse folding field has seen a particular breath of advances in (Joshi et al., 2024; Tan et al., 2023; 2025; Huang et al., 2024; Wong et al., 2024). Recent benchmarking suites like BEACON (Ren et al., 2024) provide extensive tools for models on RNA, but do not rely on structural data or focus on non-coding RNAs.

### 2.3. Structure-Based RNA Models

While most deep learning models on RNA focus on secondary structure prediction and sequence-level tasks, some structure-based models have been developed. RBind (Wang et al., 2018) proposed learning on RNA structures using residue graph representations, followed by others integrating sequence features in the learning (Su et al., 2021; Wang et al., 2023). RNAmigos and RNAmigos2 (Oliver et al., 2020; Carvajal-Patiño et al., 2025) proposed incorporating non-canonical interactions in the graph, in conjunction with metric learning pretraining. In a follow-up work (Wang et al., 2023), a similar method for binding site prediction was used. Finally, gRNAde (Joshi et al., 2024) adapted the popular GVP (Jing et al., 2021) protein encoder, which uses the atomic position in the message passing algorithm, to RNA.

Despite manifold efforts at deep learning-based modeling RNA 3D structure, we lack the means to systematically compare models and onboard new practitioners.

## 3. Tools to assemble tasks

This section describes the full process of building a task: data collection, quality filtering, cleanup, and splitting.

### Data collection and annotation

Our data originates from the RCSB-PDB DataBank (Kouranov et al., 2006), where we fetch all 3D structures containing RNA. In some tasks, filter down the set of RNA-containing PDB structures with the subset proposed in (Leontis & Zirbel, 2012) and later referred to as bgsu. We annotate each system with RNA-level features such as its resolution, and residue-level features such as their coordinates, the presence of interacting compounds, and the amount of protein atoms in the residues' vicinities.

### Structure partitioning and quality filters

RNA molecules present in PDB files exhibit a bimodal distribution over the number of residues (see figures A.1a and A.1b). Many systems have less than 300 residues, while a few others (mostly ribosomal structures) have several thousands. To avoid discarding entire systems, we split whole systems into sub-structures by sets of connected components in the graph formed by all base pairings and backbone connections. Concretely, this allows independent RNA fragments appearing in the same PDB file to be treated as multiple systems; the resulting decrease in system size leads to a reduction in required computing power for downstream steps, such as similarity-based splitting.

We implement resolution and size filters, with default cutoff values including systems below 4Å resolution and with a length of 15 to 500 residues. In tasks, the upper size cutoff is used to limit computational expenses. We also provide a protein content filter which removes RNA structures that are heavily structured through an interaction with a protein. This crucial filter has been overlooked in most of existing RNA structural datasets. In addition, we implement the drug-like filter introduced in Hariboss (Panei et al., 2022) for small molecules binding to RNA.

Benchmark for RNA 3D Structure Modeling

Table 1: Description of the proposed datasets and biologically-relevant tasks.

| TASK NAME    | SHORT DESCRIPTION | NOVEL | EXPANDED |
|--------------|-------------------|-------|----------|
| 1. RNA-GO    | Classify RNAs into one or more frequent function. | ✓ | |
| 2. RNA-IF    | Find a sequence that folds into this structure. | | ✓ |
| 3. RNA-CM    | Predict which RNA residues are chemically modified from its geometry. | ✓ | |
| 4. RNA-PROT  | Predict which RNA residues are interacting with a protein. | ✓ | |
| 5. RNA-SITE  | Predict which RNA residues are interacting with a small-molecule protein. | | ✓ |
| 6. RNA-LIGAND| Classify an RNA binding site, based on the ligand it binds to. | ✓ | |
| 7. RNA-VS    | Predict the affinity of a small molecule for an RNA binding site, and use it for Virtual Screening (VS). | | ✓ |

Minimizing data leaks Because biological data points are often related in terms of evolutionary history and topological characteristics, insufficient rigor in splitting can lead to severe data leaks. Unsurprisingly, numerous examples of data leakage have pestered structural biology learning methods, one famous example being the first PDBbind dataset that was shown to incorporate several severe biases (Wang et al., 2005b; Volkov et al., 2022). Efforts like the PINDER database (Kovtun et al., 2024) have set a standard in the field for model generalisation assessment.

In our work, we implement the computation of sequence-based or structure-based similarity matrices, with CD-HIT (Fu et al., 2012) and US-align (Zhang et al., 2022a) respectively. Given a similarity matrix S, computed over a set of RNA molecules R, and a threshold θ, we introduce the matrix Sθ defined as Sθi,j = δ{Si,j ≥ θ}, where δ is the indicator function. We propose a clustering algorithm that considers (R, Sθ) as a graph, and returns connected components as clusters. This ensures that any pair of points in different clusters has a maximal similarity of θ.

Redundancy filtering and dataset splitting Starting with clusters, we propose a redundancy removal algorithm that selects the element with the highest resolution for each cluster. In most tasks, we first apply this algorithm at a sequence similarity threshold of 0.9. This stringent threshold discards copies of systems with minimal variations that are common when getting the structure of a given system in slightly different conditions. Then, we apply another structure-based redundancy threshold of 0.8.

In addition, we propose a novel splitting algorithm that starts from clustered data, but uses a less conservative structural similarity threshold (0.5 unless mentioned otherwise). Then, we aggregate those clusters together to form splits of a certain size, optionally following a label balancing secondary objective, using the linear programming PuLP (Mitchell et al., 2011). This prevents data leakage between splits, while allowing split stratification based on labels.

Metrics computation Given the type of prediction target (e.g. binary classification, multi-label, etc.), each task implements a standardized way to compute evaluation metrics. These are specified for each task in Section 4.

Anatomy of a task A "task" is now simply the collection of the aforementioned components with the choices relevant to a particular biological problem. That is, each task consists of (i) a dataset drawn from our 3D structure database and processed using our annotations, partitions and filters, with redundancy removing (ii) fixed splits based on sequence, structure or existing literature, and (iii) a well-defined evaluation protocol. This modular design will help other practitioners propose additional benchmark tasks, and lower the barrier to entry for training models on RNA structure.

## 4. Tasks on RNA 3D Structure

Our tasks suite explores various dimensions of RNA structural research briefly summarized in Table 1. We offer seven tasks among which three are based on previous research and offer publicly available split datasets (RNA-IF (Joshi et al., 2024), RNA-Site (Su et al., 2021) and RNA-VS (Carvajal-Patino et al., 2025)). Moreover, for two of these three tasks, we propose an enhanced, expanded version, resulting in a total of nine separate datasets.

Armed with an RNA structure, the main focus of each of the subsequent tasks is to either predict aspects of the RNA's biological function, or to reverse the direction and predict the sequence from which a given structure arises (RNA design). We provide an overview of these tasks from a machine learning point of view relating to the type of task and dataset size in Table 2.

Benchmark for RNA 3D Structure Modeling

Table 2: Description of the ML features of each of our tasks. The number in parenthesis refers to the number of class, in the multi-class setting. We include a comparison of the datasets we introduce with existing ones.

| TASK NAME | LEVEL | TYPE | DATASET SPLIT SIZES |
|------------|-------|------|---------------------|
| 1. RNA-GO | RNA | MULTI-CLASS (5) | 349-75-75 |
| 2. RNA-IF | RESIDUE | MULTI-CLASS (4) | 1700-448-581 |
| gRNAde (Joshi et al., 2024) | | | 11183-528-235 |
| 3. RNA-CM | RESIDUE | BINARY | 138-29-30 |
| 4. RNA-PROT | RESIDUE | BINARY | 881-189-189 |
| 6. RNA-SITE | RESIDUE | BINARY | 157-34-33 |
| RNASite (Su et al., 2021) | | | 53-6-17 |
| 6. RNA-LIGAND | POCKET | MULTI-CLASS (3) | 203-43-44 |
| 7. RNA-VS (CARVAJAL-PATIÑO ET AL., 2025) | POCKET | REGRESSION | 304-34-65 |

## 4.1. RNA-GO: Function Tagging

(Definition): This is a multi-label classification task where input RNA is mapped to a sequence of 5 possible labels representing molecular functions.

(Context): Effective function prediction models have the capacity to uncover new structure-function connections. The Gene Ontology (GO) (Ashburner et al., 2000) was developed to associate a function to a gene, and thus, indirectly to the RNA or protein it encodes. It resulted in discrete functional categories called GO terms. Predicting the GO term from a protein structure was proposed as a task in (Gligorijevic et al., 2021) and has since been regularly used as a benchmark task. A mapping between RNA sequences and GO terms is available in Rfam (Griffiths-Jones et al., 2003; Ontiveros-Palacios et al., 2025), a database of non-coding RNA families which are manually annotated with GO terms.

(Processing): We extracted all GO terms corresponding to an RNA structure. Then, we filtered these terms based on their frequency, removing the two most frequent labels (ribosomal systems and tRNAs), as well as infrequent ones (less than 50 occurrences). We then grouped together GO-terms that were fully correlated, which resulted in five classes. Finally, we discarded RNA fragments with less than 15 residues, resulting in 501 systems. We split these systems based on a sequence similarity cutoff, following Gligorijevic et al. (2021). Accuracy, AuROC and the Jaccard index are the metrics computed for evaluation.

## 4.2. RNA-IF: Molecular Design

(Definition): This is a residue-level, classification task where given an RNA structure, we mask all residue identities (by keeping only the coordinates of the backbone) and aim to learn the mapping from structure to sequence.

(Context): One avenue for designing new macromolecules is to design a coarse grained tertiary structure first, for instance scaffolding a structural motif, and then design a sequence that will fold into this given structure. This second step is denoted as Inverse Folding (IF), since it maps a known structure to an unknown sequence. Inverse folding is a well-established task in protein research with many recent breakthroughs (Watson et al., 2023). It has recently been adapted for RNA with classical models such as Rosetta (Leman et al., 2020a). Learning based approaches were pioneered by gRNAde (Leman et al., 2020a; Joshi et al., 2024) and other papers specifically address backbone generation (Anand et al., 2024) and protein-binding RNA design (Nori & Jin, 2024).

(Processing): We gather all connected components in our 3D structure database with a size between 15 and 300. Then, we removed identical sequences using a redundancy removal step with CD-HIT with a cutoff of 0.99. We cluster the data on structural similarity with a threshold of 0.9, and split these while ensuring a maximal similarity of 0.5. We also provide datasets and splits from gRNAde (Joshi et al., 2024), that were obtained in a similar manner. Our dataset differs by allowing multi-chain systems, a stricter size cutoff, and a duplicates filter, leading to a reduced dataset size. This task is evaluated with sequence recovery.

Benchmark for RNA 3D Structure Modeling

4.3. RNA-CM: Dynamic function modulation.

(Processing): To build this task, we start from the bgsu non-redundant dataset, partition it into connected components and apply our size and resolution filters. We remove systems originating from ribosomes that are fully encapsulated in a complex. Then, we retain only RNA interacting with a protein and apply our default redundancy removal and splitting strategies, resulting in 1251 data points. Performance is evaluated with accuracy and AuROC.

(Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict which, if any, of its residues are chemically modified.

(Context): In addition to the four canonical nucleotides, more than 170 modified nucleotides were found to be integrated in RNA polymers (Boccaletto et al., 2022). Multiple functions of a diverse range of ncRNA have been shown to be directly dependent on such chemical modifications (Roundtree et al., 2017). We propose to detect such chemical modifications from subtle perturbations of the canonical geometry of the RNA backbones. Training a predictor on RNA structures is not the most straightforward way of detecting chemical modifications, but may uncover function-modulating modifications that hitherto went unnoticed.

(Processing): To build this task, we start from the whole dataset, partition it into connected components and apply our size filter. Then, we filter for systems that include modified residues, relying on PDB annotations that flag such modified residues. We apply our default redundancy removal and splitting strategies, which results in 185 data points. Performance is evaluated with accuracy and AuROC.

Tasks for RNA drug design Further tasks aim to address RNA small-molecule drug-design. RNA is increasingly recognized as a promising family of targets for the development of novel small molecule therapeutics (Falese et al., 2021; Haga & Phinney, 2023; Disney, 2019; Abulwerdi et al., 2019). Being able to target RNA would drastically increase the size of the druggable space, and propose an alternative for overused protein targets in scenarios where they are insufficient. In addition, they represent a therapeutic avenue in pathologies where protein targets are absent, such as in triple-negative breast cancer (Xu et al., 2020).

4.5. RNA-Site: Drug target detection.

(Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict whether a ligand is closer than 6Å to any of its residues.

(Context): The classical flow of a structure-based drug discovery pipeline starts with the identification of relevant binding sites, which are parts of the structure likely to interact with ligands, or of particular interest for a phenotypical effect. The structure of the binding site can then be used to condition the quest for small molecule binders, for instance using molecular docking (Ruiz-Carmona et al., 2014). The framing of this problem as a machine learning task for RNA was introduced in Rbind (Wang et al., 2018).

(Processing): We provide datasets and predefined splits from RNASite (Su et al., 2021), which are also utilized by other tools in the field such as RLBind (Wang et al., 2023). This dataset contains 76 systems obtained after applying stringent clustering on an older version of the PDB.

4.4. RNA-Prot: Biological complex modeling.

(Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict whether a protein residue is closer than 8Å to any of its residues.

(Context): RNAs and proteins often bind to form a functional complex. Such complexes are involved in crucial cell processes, such as post-transcriptional control of RNAs (Glisovic et al., 2008). RNA structure is often, but not always, heavily disrupted upon interaction with a protein. We expect this task to be an easier version of the RNA-CM task.

In addition to providing the existing dataset existing in the literature, we propose a larger, up-to-date dataset. We start by following similar steps as for RNA-Prot (without removal of ribosomal systems). We then include a drug-like filter on the small-molecule side. Finally, we include a protein-content filter, removing systems with more than 10 protein

Benchmark for RNA 3D Structure Modeling

atom neighbors, ensuring that the binding is modulated by RNA only. We use the default redundancy removal and splitting, resulting in 458 systems. The predictions are evaluated using accuracy and AuROC.

## 4.6. RNA-Ligand: Pocket categorization

```
Class Probabilities
1
  Representative Ligand
  
  
k
```

(Definition): This is a binding site-level, multi-class classification task where the structure of an RNA binding site is classified according to the partner it accommodates.

(Context): Equipped with a binding site, one wants to use its structure to characterize its potential binders. On proteins, the Masif-Ligand tasks (Gainza et al., 2020) gathers all binding sites bound to the seven most frequent co-factors, and aims to classify them based on their partner. Inspired by this work we propose the RNA-Ligand task. To keep sufficiently many examples per task, we only retained the three most frequent classes. This task is less ambitious than training a molecular docking surrogate and can help understanding the potential modulators of a given binding site.

(Processing): Starting with similar steps as the RNA-Site, we obtain a set of structures that display RNAs in interaction with one or more drug-like small molecules. We then proceed to extracting the context of the binding pocket by seeding two rounds of breadth-first search with all residues closer than 8Å to an atom of the binder. This results in all binding pockets in our database and we group these pockets with a sequence clustering.

Then, our aim is to find the most frequent ligands binding in non-redundant pockets. To that end, we gather a dataset of binding site clusters that bind to one ligand only. We then compute the most frequent ligands among this dataset and retain only the top three. Subsequently, we discard all binding events to other small molecules, and add clusters that bind only one of the top three ligands to our dataset. This dataset is split on structural similarity and evaluated on MCC and AuROC.

## 4.7. RNA-VS: Drug screening

| RNA structure | Molecule 1 | Probability | Molecule 2 |
|---------------|------------|-------------|------------|
| [RNA structure] | [Molecule structure 1] | 0.3 | [Similar molecule structure 1] |
| | [Molecule structure 2] | 0.8 | [Similar molecule structure 2] |
| | [Molecule structure 3] | 0.1 | [Similar molecule structure 3] |

(Definition): This is a binding site-level, regression task. Given the structure of an RNA binding site and a small molecule represented as a molecular graph, the goal is to predict the affinity of their binding.

(Context): Beyond classification into fixed categories, virtual screening aims to score compounds based on their affinity to a binding pocket. This task is ubiquitous in drug design as it helps selecting the most promising compounds to be further assayed in the wet-lab. Our last task implements a virtual screening task introduced in (Carvajal-Patiño et al., 2025). Their model is trained to approximate normalized molecular docking scores. Trained models can then be used to rank compounds by their binding likelihood to target sites, hence achieving virtual screening.

(Processing): The dataset is reproduced from the RNAmigos2 (Carvajal-Patiño et al., 2025) paper. The authors curated a list of binding sites in a similar fashion to the one described in the RNA-Site task and clustered the sites using RMScore (Zheng et al., 2019). All binders found for each cluster are retained as positive examples, and a set of drug-like chemical decoys are added as negative partners. Molecular docking scores are computed with rDock (Ruiz-Carmona et al., 2014) on all binding site-small molecule pairs. The metric used is the AuROC.

## 5. Implementation

Next, we briefly showcase the use of our framework for a simple programmatic access to proposed tasks. In Figure 1, we show how practitioners can access our datasets from Python code, automatically downloading them from Zenodo, choosing a representation (in this example a Pytorch Geometric graph) and directly accessing the data in a simple and reproducible fashion. Additionally, by passing a single flag to our loaders, users can choose to execute all processing logic described here to build to build each task from scratch (end-to-end reproducibility).

Due to the rapid advances in the field, we can expect that additional interesting challenges will arise in the near future, complementary to the seven tasks introduced here. Thanks to the modularity of our tool, additional tasks on RNA can be easily integrated in our framework for future releases. This is illustrated in the documentation at rnaglib.org.

Benchmark for RNA 3D Structure Modeling

```python
from rnaglib.tasks import get_task
from rnaglib.representations import GraphRepresentation

task = get_task(task_id='rna_site',root='example')
task.add_representation(GraphRepresentation(framework="pyg"))
task.get_split_loaders()

for batch in task.train_dataloader:
    rna_graph = batch["graph"]
    target = rna_graph.y
```

Figure 1: Obtaining a machine learning-ready split dataset only requires a few lines of code.

## 6. Experiments

We hereby explicitly state that we limit the scope of this work to the creation of a useful deep learning library for RNA 3D structure based modeling. We refrain from suggesting novel model architectures; models discussed in this work uniquely serve the goal of illustrating the practical applicability of our library. Within these constraints, we highlight the utility of our benchmark by training a simple relational graph convolutional networks (RGCN) on our proposed tasks. The performance of this baseline is reported in Table 3. Additional details on the experimental setup and results across further metrics can be found in supplementary Section B. These results provide an initial baseline for future model developments on the introduced tasks.

| Task | Metric | Score |
|------|--------|-------|
| RNA_Ligand | AuROC | 0.6751 |
| RNA_CM | Balanced Accuracy | 0.6615 |
| RNA_Site | Balanced Accuracy | 0.6309 |
| RNA_Prot | Balanced Accuracy | 0.6254 |
| RNA_IF | Sequence Recovery | 0.3523 |
| RNA_VS | AuROC | 0.8550 |
| RNA_GO | Jaccard | 0.3167 |

Table 3: Selected representative performance metric for each task.

Furthermore, we compare our RGCN approach to existing methods on the TR60/TE18 dataset split (Su et al., 2021) for the RNA-Site task and the dataset and splits from Joshi et al. (2024) for the RNA-VS task. Performance metrics are provided in Supplementary Tables A.2 and A.3. Despite using a relatively simple model, our approach achieves competitive results, aligning with recent methods while falling short of state-of-the-art performance.

Well-defined datasets and splits, our library enables robust benchmarking and facilitates the development of innovative computational models. It also ensures reproducibility and promotes standardized evaluation, fostering confidence in computational findings.

In the future, the development of additional tasks, such as the assessment of structural models (Townshend et al., 2021b) and a greater focus on RNA embeddings, presents exciting opportunities.

As the understanding of the dynamic nature of biological macromolecules evolves, so may the preferred representation for RNA. Our tool's built-in 2.5D-graph representations offer extensive adaptability and have been used to develop recent competitive RNA structure-based models (Wang et al., 2024). Nevertheless, the library can easily be extended to incorporate new or improved representations as they emerge.

In this work, we unify various biological challenges where RNA structure is a key player, and provide direct access to high quality data and benchmarking functionality for deep learning model development and computational biology communities alike. Furthermore, the choice of integrating the benchmark in the rnaglib toolkit opens access to non-specialist researchers, and provides robust guarantees for quality and reproducibility. We hope that this benchmark will facilitate the development of new deep learning tools in the highly promising area of RNA structural biology.

## 7. Discussion

We have introduced a versatile and modular library designed to advance the application of deep learning to RNA structural analysis. By providing a suite of diverse and

# Benchmark for RNA 3D Structure Modeling

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Abulwerdi, F. A., Xu, W., Ageeli, A. A., Yonkunas, M. J., Arun, G., Nam, H., Schneekloth Jr, J. S., Dayie, T. K., Spector, D., Baird, N., et al. Selective small-molecule targeting of a triple helix encoded by the long noncoding rna, malat1. ACS chemical biology, 14(2):223–235, 2019.

Adamczyk, B., Antczak, M., and Szachniuk, M. Rnasolo: a repository of cleaned pdb-derived rna 3d structures. Bioinformatics, 38(14):3668–3670, 2022.

Alam, T., Uludag, M., Essack, M., Salhi, A., Ashoor, H., Hanks, J. B., Kapfer, C., Mineta, K., Gojobori, T., and Bajic, V. B. Farna: knowledgebase of inferred functions of non-coding rna transcripts. Nucleic acids research, 45 (5):2838–2848, 2017.

Anand, R., Joshi, C. K., Morehead, A., Jamasb, A. R., Harris, C., Mathis, S. V., Didi, K., Hooi, B., and Lio, P. Rnaframeflow: Flow matching for de novo 3d rna backbone design. arXiv preprint arXiv:2406.13839, 2024.

Ashburner, M., Ball, C. A., Blake, J. A., Botstein, D., Butler, H., Cherry, J. M., Davis, A. P., Dolinski, K., Dwight, S. S., Eppig, J. T., et al. Gene ontology: tool for the unification of biology. Nature genetics, 25(1):25–29, 2000.

Becquey, L., Angel, E., and Tahi, F. Rnanet: an automatically built dual-source dataset integrating homologous sequences and rna structures. Bioinformatics, 37(9):1218–1224, 2021.

Boccaletto, P., Stefaniak, F., Ray, A., Cappannini, A., Mukherjee, S., Purta, E., Kurkowska, M., Shirvanizadeh, N., Destefanis, E., Groza, P., et al. Modomics: a database of rna modification pathways. 2021 update. Nucleic acids research, 50(D1):D231–D235, 2022.

Buttenschoen, M., Morris, G. M., and Deane, C. M. Posebusters: Ai-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15(9):3130–3139, 2024.

Carvajal-Patino, J. G., Mallet, V., Becerra, D., Niño Vasquez, L. F., Oliver, C., and Waldispühl, J. Rnamigos2: accelerated structure-based rna virtual screening with deep graph learning. Nature Communications, 16(1):1–12, 2025.

Cech, T. R. and Steitz, J. A. The noncoding rna revolution—trashing old rules to forge new ones. Cell, 157(1): 77–94, 2014.

Corso, G., Stärk, H., Jing, B., Barzilay, R., and Jaakkola, T. Diffdock: Diffusion steps, twists, and turns for molecular docking. arXiv preprint arXiv:2210.01776, 2022.

Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R. J., Milles, L. F., Wicky, B. I., Courbet, A., de Haas, R. J., Bethel, N., et al. Robust deep learning–based protein sequence design using proteinmpnn. Science, 378 (6615):49–56, 2022.

Disney, M. D. Targeting rna with small molecules to capture opportunities at the intersection of chemistry, biology, and medicine. Journal of the American Chemical Society, 141 (17):6776–6790, 2019.

Durairaj, J., Adeshina, Y., Cao, Z., Zhang, X., Oleinikovas, V., Duignan, T., McClure, Z., Robin, X., Kovtun, D., Rossi, E., et al. Plinder: The protein-ligand interactions dataset and evaluation resource. bioRxiv, pp. 2024–07, 2024.

Falese, J. P., Donlic, A., and Hargrove, A. E. Targeting rna with small molecules: from fundamental principles towards the clinic. Chemical Society Reviews, 50(4): 2224–2243, 2021.

Fu, L., Niu, B., Zhu, Z., Wu, S., and Li, W. Cd-hit: accelerated for clustering the next-generation sequencing data. Bioinformatics, 28(23):3150–3152, 2012.

Gainza, P., Sverrisson, F., Monti, F., Rodolà, E., Boscaini, D., Bronstein, M., and Correia, B. Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. Nature Methods, 17(2):184–192, 2020.

Gligorijevic, V., Renfrew, P. D., Kosciolek, T., Leman, J. K., Berenberg, D., Vatanen, T., Chandler, C., Taylor, B. C., Fisk, I. M., Vlamakis, H., et al. Structure-based protein function prediction using graph convolutional networks. Nature communications, 12(1):3168, 2021.

Glisovic, T., Bachorik, J. L., Yong, J., and Dreyfuss, G. Rna-binding proteins and post-transcriptional gene regulation. FEBS letters, 582(14):1977–1986, 2008.

Griffiths-Jones, S., Bateman, A., Marshall, M., Khanna, A., and Eddy, S. R. Rfam: an rna family database. Nucleic acids research, 31(1):439–441, 2003.

Haga, C. L. and Phinney, D. G. Strategies for targeting rna with small molecule drugs. Expert Opinion on Drug Discovery, 18(2):135–147, 2023.

Hou, J., Adhikari, B., and Cheng, J. Deepsf: deep convolutional neural network for mapping protein sequences to folds. Bioinformatics, 34(8):1295–1303, 2018.

Benchmark for RNA 3D Structure Modeling

Huang, H., Lin, Z., He, D., Hong, L., and Li, Y. Ribodiffusion: tertiary structure-based rna inverse folding with generative diffusion models. Bioinformatics, 40 (Supplement 1):i347–i356, 2024.

Jamasb, A. R., Morehead, A., Joshi, C. K., Zhang, Z., Didi, K., Mathis, S., Harris, C., Tang, J., Cheng, J., Lio, P., et al. Evaluating representation learning on the protein structure universe. ArXiv, pp. arXiv–2406, 2024.

Jing, B., Eismann, S., Suriana, P., Townshend, R. J., and Dror, R. Learning from protein structure with geometric vector perceptrons. arXiv preprint arXiv:2009.01411, 2020.

Jing, B., Eismann, S., Soni, P. N., and Dror, R. O. Equivariant graph neural networks for 3d macromolecular structure, 2021. URL https://arxiv.org/abs/2106.03843.

Joshi, C. K., Jamasb, A. R., Viñas, R., Harris, C., Mathis, S. V., Morehead, A., and Lio, P. grnade: Geometric deep learning for 3d rna inverse design. bioRxiv, pp. 2024–03, 2024.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., et al. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589, 2021.

Kouranov, A., Xie, L., de la Cruz, J., Chen, L., Westbrook, J., Bourne, P. E., and Berman, H. M. The rcsb pdb information portal for structural genomics. Nucleic acids research, 34(suppl 1):D302–D305, 2006.

Kovtun, D., Akdel, M., Goncearenco, A., Zhou, G., Holt, G., Baugher, D., Lin, D., Adeshina, Y., Castiglione, T., Wang, X., et al. Pinder: The protein interaction dataset and evaluation resource. bioRxiv, pp. 2024–07, 2024.

Kucera, T., Oliver, C., Chen, D., and Borgwardt, K. Proteinshake: Building datasets and benchmarks for deep learning on protein structures. In Advances in Neural Information Processing Systems, volume 36, pp. 58277–58289, 2023.

Leman, J. K., Weitzner, B. D., Lewis, S. M., Adolf-Bryfogle, J., Alam, N., Alford, R. F., Aprahamian, M., Baker, D., Barlow, K. A., Barth, P., et al. Macromolecular modeling and design in rosetta: recent methods and frameworks. Nature methods, 17(7):665–680, 2020a.

Leman, J. K., Weitzner, B. D., Lewis, S. M., Adolf-Bryfogle, J., Alam, N., Alford, R. F., Aprahamian, M., Baker, D., Barlow, K. A., Barth, P., et al. Macromolecular modeling and design in rosetta: recent methods and frameworks. Nature methods, 17(7):665–680, 2020b.

Leontis, N. B. and Zirbel, C. L. Nonredundant 3d structure datasets for rna knowledge extraction and benchmarking. RNA 3D structure analysis and prediction, pp. 281–298, 2012.

Lorenz, R., Bernhart, S. H., Höner zu Siederdissen, C., Tafer, H., Flamm, C., Stadler, P. F., and Hofacker, I. L. Viennarna package 2.0. Algorithms for molecular biology, 6:1–14, 2011.

Mitchell, S., OSullivan, M., and Dunning, I. Pulp: a linear programming toolkit for python. The University of Auckland, Auckland, New Zealand, 65:25, 2011.

Nori, D. and Jin, W. Rnaflow: Rna structure and sequence design via inverse folding-based flow matching, 2024. URL https://arxiv.org/abs/2405.18768.

Notin, P., Kollasch, A., Ritter, D., van Niekerk, L., Paul, S., Spinner, H., Rollins, N., Shaw, A., Orenbuch, R., Weitzman, R., Frazer, J., Dias, M., Franceschi, D., Gal, Y., and Marks, D. Proteingym: Large-scale benchmarks for protein fitness prediction and design. In Advances in Neural Information Processing Systems, volume 36, pp. 64331–64379, 2023.

Oliver, C., Mallet, V., Gendron, R. S., Reinharz, V., Hamilton, W. L., Moitessier, N., and Waldispühl, J. Augmented base pairing networks encode rna-small molecule binding preferences. Nucleic acids research, 48(14):7690–7699, 2020.

Ontiveros-Palacios, N., Cooke, E., Nawrocki, E. P., Triebel, S., Marz, M., Rivas, E., Griffiths-Jones, S., Petrov, A. I., Bateman, A., and Sweeney, B. Rfam 15: Rna families database in 2025. Nucleic Acids Research, 53(D1):D258–D267, 2025.

Panei, F. P., Torchet, R., Menager, H., Gkeka, P., and Bonomi, M. Hariboss: a curated database of rna-small molecules structures to aid rational drug design. Bioinformatics, 38(17):4185–4193, 2022.

Ren, Y., Chen, Z., Qiao, L., Jing, H., Cai, Y., Xu, S., Ye, P., Ma, X., Sun, S., Yan, H., et al. Beacon: Benchmark for comprehensive rna tasks and language models. Advances in Neural Information Processing Systems, 37:92891–92921, 2024.

Roundtree, I. A., Evans, M. E., Pan, T., and He, C. Dynamic rna modifications in gene expression regulation. Cell, 169(7):1187–1200, 2017.

Ruiz-Carmona, S., Alvarez-Garcia, D., Foloppe, N., Garmendia-Doval, A. B., Juhos, S., Schmidtke, P., Barril, X., Hubbard, R. E., and Morley, S. D. rdock: A

Benchmark for RNA 3D Structure Modeling

fast, versatile and open source program for docking ligands to proteins and nucleic acids. PLoS Computational Biology, 10:1–8, 2014. ISSN 15537358. doi: 10.1371/journal.pcbi.1003571.

Schneuing, A., Harris, C., Du, Y., Didi, K., Jamasb, A., Igashov, I., Du, W., Gomes, C., Blundell, T. L., Lio, P., et al. Structure-based drug design with equivariant diffusion models. Nature Computational Science, 4(12): 899–909, 2024.

Statello, L., Guo, C.-J., Chen, L.-L., and Huarte, M. Gene regulation by long non-coding rnas and its biological functions. Nature reviews Molecular cell biology, 22(2): 96–118, 2021.

Su, H., Peng, Z., and Yang, J. Recognition of small molecule–rna binding sites using rna sequence and structure. Bioinformatics, 37(1):36–42, 2021.

Szikszai, M., Magnus, M., Sanghi, S., Kadyan, S., Bouatta, N., and Rivas, E. Rna3db: A structurally-dissimilar dataset split for training and benchmarking deep learning models for rna structure prediction. Journal of Molecular Biology, pp. 168552, 2024. ISSN 0022-2836. doi: https://doi.org/10.1016/j.jmb.2024.168552.

Tan, C., Zhang, Y., Gao, Z., Hu, B., Li, S., Liu, Z., and Li, S. Z. Rdesign: hierarchical data-efficient representation learning for tertiary structure-based rna design. arXiv preprint arXiv:2301.10774, 2023.

Tan, C., Zhang, Y., Gao, Z., Cao, H., Li, S., Ma, S., Blanchette, M., and Li, S. Z. R3design: deep tertiary structure-based rna sequence design and beyond. Briefings in Bioinformatics, 26(1):bbae682, 2025.

Townshend, R., Vögele, M., Suriana, P., Derry, A., Powers, A., Laloudakis, Y., Balachandar, S., Jing, B., Anderson, B., Eismann, S., Kondor, R., Altman, R., and Dror, R. Atom3d: Tasks on molecules in three dimensions. In Advances in Neural Information Processing Systems, Datasets and Benchmarks, volume 1, 2021a.

Townshend, R. J., Eismann, S., Watkins, A. M., Rangan, R., Karelina, M., Das, R., and Dror, R. O. Geometric deep learning of rna structure. Science, 373(6558):1047–1051, 2021b.

van Kempen, M., Kim, S. S., Tumescheit, C., Mirdita, M., Gilchrist, C. L., Söding, J., and Steinegger, M. Foldseek: fast and accurate protein structure search. Biorxiv, pp. 2022–02, 2022.

Volkov, M., Turk, J.-A., Drizard, N., Martin, N., Hoffmann, B., Gaston-Mathé, Y., and Rognan, D. On the frustration to predict binding affinities from protein–ligand structures with deep neural networks. Journal of medicinal chemistry, 65(11):7946–7958, 2022.

Wang, J., Quan, L., Jin, Z., Wu, H., Ma, X., Wang, X., Xie, J., Pan, D., Chen, T., Wu, T., et al. Multimodrlbp: A deep learning approach for multi-modal rna-small molecule ligand binding sites prediction. IEEE Journal of Biomedical and Health Informatics, 2024.

Wang, K., Jian, Y., Wang, H., Zeng, C., and Zhao, Y. Rbind: computational network method to predict rna binding sites. Bioinformatics, 34(18):3131–3136, 2018.

Wang, K., Zhou, R., Wu, Y., and Li, M. Rlbind: a deep learning method to predict rna–ligand binding sites. Briefings in Bioinformatics, 24(1):bbac486, 2023.

Wang, L., Liu, H., Liu, Y., Kurtin, J., and Ji, S. Learning hierarchical protein representations via complete 3d graph networks, 2022. URL https://arxiv.org/abs/2207.12600.

Wang, R., Fang, X., Lu, Y., Yang, C.-Y., and Wang, S. The pdbbind database: methodologies and updates. Journal of medicinal chemistry, 48(12):4111–4119, 2005a.

Wang, R., ueliang Fang, Lu, Y., Yang, C.-Y., and Wang, S. The pdbbind database: Methodologies and updates. Journal of Medicinal Chemistry, 22, 11 2005b. ISSN 4111–4119. doi: 10.1021/jm048957q.

Watson, J. L., Juergens, D., Bennett, N. R., Trippe, B. L., Yim, J., Eisenach, H. E., Ahern, W., Borst, A. J., Ragotte, R. J., Milles, L. F., et al. De novo design of protein structure and function with rfdiffusion. Nature, 620(7976): 1089–1100, 2023.

Wong, F., He, D., Krishnan, A., Hong, L., Wang, A. Z., Wang, J., Hu, Z., Omori, S., Li, A., Rao, J., et al. Deep generative design of rna aptamers using structural predictions. Nature Computational Science, pp. 1–11, 2024.

Xu, J., Wu, K.-j., Jia, Q.-j., and Ding, X.-f. Roles of mirna and lncrna in triple-negative breast cancer. Journal of Zhejiang University-science b, 21(9):673–689, 2020.

Zeng, P. and Cui, Q. Rsite2: an efficient computational method to predict the functional sites of noncoding rnas. Scientific Reports, 6(1):19016, 2016.

Zeng, P., Li, J., Ma, W., and Cui, Q. Rsite: a computational method to identify the functional sites of noncoding rnas. Scientific Reports, 5(1):9179, 2015.

Zhang, C., Shine, M., Pyle, A. M., and Zhang, Y. Us-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. Nature methods, 19(9): 1109–1115, 2022a.

Zhang, Z., Xu, M., Jamasb, A., Chenthamarakshan, V., Lozano, A., Das, P., and Tang, J. Protein representation learning by geometric structure pretraining. arXiv preprint arXiv:2203.06125, 2022b.

# Benchmark for RNA 3D Structure Modeling

Zheng, J., Xie, J., Hong, X., and Liu, S. Rmalign: an
rna structural alignment tool based on a novel scoring
function rmscore. BMC genomics, 20:1–10, 2019.

Zhu, Z., Shi, C., Zhang, Z., Liu, S., Xu, M., Yuan, X.,
Zhang, Y., Chen, J., Cai, H., Lu, J., et al. Torchdrug: A
powerful and flexible machine learning platform for drug
discovery. arXiv preprint arXiv:2202.08320, 2022.

# Appendix

## A. Details about datasets

In this appendix, we first detail more extensively the datasets and data preprocessing applied (Section A). Then, we provide additional results of the models tested on our tasks together with details about training and hyperparameters used to perform the experiments (Section B). Eventually, we provide the code defining RNA-CM task, which provides a model to introduce new tasks (Section ??).

### A.1. Datasets and general preprocessing

In this section, we provide additional insights about the data and the way they are being preprocessed.

As mentioned in section 3, RNA molecules present in the PDB files exhibit a bimodal distribution over the number of residues. Figure A.1a displays the distribution of the RNA sizes (defined as the number of residues) of the RNAs from the PDB.

| RNA sizes distribution | RNA chunks sizes distribution | RNA chunks sizes distribution |
|------------------------|-------------------------------|-------------------------------|
| RNA sizes   | RNA chunks         | RNA chunks filtered|
| (a) Whole RNAs         | (b) RNA chunks                | (c) RNA chunks after size filtering |

Figure A.1: Distribution of RNAs or RNA chunks sizes

In order to work at a more biologically relevant scale, we partition the raw RNAs from PDB files into connected components in order to use these connected components as samples in our different machine learning tasks (except RNA-Ligand in which the data are partitioned in a different way, into binding pockets rather than into connected components). Figure A.1c represents the distribution of the number of residues by connected component chunks.

We remove the RNA components of insufficient size for structure-based machine learning, as well as very large components which negatively impact the computational performance of the data loading as well as model training. To this end, we filter RNA chunks and only keep the ones having between 15 and 300 nucleotides. Figure A.1c displays the distribution of RNA chunks sizes after filtering.

The redundancy removal process we apply is the following. First, we perform sequence-based clustering based on a similarity threshold above which RNA fragments are clustered together (relying on sequence similarity metric CD-HIT (Fu et al., 2012)). Then, within each cluster, we select the sample with the highest resolution. Then we perform structure-based clustering and structure-based redundancy removal (relying on structural similarity metric US-align (Zhang et al., 2022a)) following the same procedure. Afterwards, when instantiating the splitters, we perform structure-based clustering with a different threshold to define the clusters which will be required to be grouped either in train, val or test set (which we name "splitting clustering"). Since we only select one representative sample per cluster, the number of clusters prefigures the number of samples of the final dataset. Therefore, a tradeoff is to be considered between having a large amount of data and discarding redundant RNAs.

For this reason, we study the impact of both the redundancy removal and the splitting clustering threshold on the number of clusters generated in the case of RNA-CM task. Results are reported in figure A.2. Figure A.2a displays, for 4 different sequence-based redundancy removal thresholds (each with a different color), the scatter plot of the number of clusters obtained based on the structure-based splitting clustering threshold. Here, the redundancy removal thresholds 0.90, 0.80

Benchmark for RNA 3D Structure Modeling

and 0.70 give the same plot since CD-HIT similarity values are strongly concentrated around 0.5 and 1. We finally choose 0.90. Figure A.2b displays, for 4 different structure-based redundancy removal thresholds (each with a different color), the scatter plot of the number of clusters obtained based on the structure-based splitting clustering threshold. In all cases, the structure-based redundancy removal is performed after a sequence-based redundancy removal using a 0.90 threshold. In all the experiments, the splitting clustering is performed based on US-align similarity (structural similarity).

Number of clusters in RNA-CM
Number of clusters in RNA-CM

Figure A.2: Number of clusters after similarity clustering based on the splitting clustering threshold

## A.2. Task-specific data preprocessing

In this section, we provide additional details about the tasks involving a specific data preparation process.

### A.2.1. RNA-LIGAND

In this task, we selected the three most frequent ligands which were neither modified RNA or DNA residues nor modified protein residues, following the ligand definition proposed by Hariboss (Panei et al., 2022). We only retained the binding pockets binding to one of these three ligands in order to ensure the amount of binding pockets binding to each ligand would be significant enough to enable learning in a multi-class classification framework. The three ligands retained are paromomycin (called PAR in the PDB nomenclature), gentamycin C1A (LLL) and aminoglycoside TC007 (8UZ). Their structures are displayed in Figure A.3.

Structures of ligands

Figure A.3: Structures of the ligands selected for RNA-Ligand task

### A.2.2. RNA-GO

When building RNA-GO, we explore all the GO-terms of RNAs from the PDB, remove the GO-terms which occur more than 1000 times (these are in fact the GO-terms corresponding to "structural constituent of ribosome", "ribosome" and "tRNA").

Benchmark for RNA 3D Structure Modeling

We also remove the GO-terms which are underrepresented (those which occur less than 50 times in our experiments). We then remove the GO-terms which are very correlated to other ones by performing GO-terms clustering based on correlation matrix (with correlation threshold 0.9) and keeping only one representative GO-term per cluster. After this preprocessing, 5 GO-terms are remaining: 0000353 (formation of quadruple SL/U4/U5/U6 snRNP), 0005682 (U5 snRNP), 0005686 (U2 snRNP), 0005688 (U6 snRNP) and 0010468 (regulation of gene expression).

## B. Additional results

Table A.1: Test performance metrics for various RNA-related tasks.

| Task       | Test F1-Score | Test AUC | Test Global Balanced Accuracy | Test MCC | Test Jaccard |
|------------|---------------|----------|-------------------------------|----------|--------------|
| RNA_Ligand | 0.2771        | 0.6751   | 0.4678                        |          |              |
| RNA_CM     | 0.1957        | 0.7393   | 0.6615                        | 0.1695   |              |
| RNA_Site   | 0.3346        | 0.5929   | 0.6309                        | 0.3098   |              |
| RNA_Prot   | 0.4545        | 0.6654   | 0.6254                        | 0.2469   |              |
| RNA_IF     | 0.3326        | 0.6201   | 0.3523*                       | 0.1319   |              |
| RNA_VS     |               | 0.855    |                               |          |              |
| RNA_GO     | 0.4074        | 0.8406   | 0.7067                        |          | 0.3167       |

Hyperparameters used:
- RNA_Ligand: n_layers=4, hidden_dim=128, lr=0.00001, dropout=0.5
- RNA_CM: n_layers=3, hidden_dim=128, lr=0.001, dropout=0.5
- RNA_Site: n_layers=4, hidden_dim=256, lr=0.001, dropout=0.5
- RNA_Prot: n_layers=4, hidden_dim=64, lr=0.01, dropout=0.2
- RNA_IF: n_layers=3, hidden_dim=128, lr=0.0001, dropout=0.5
- RNA_VS: n_layers=3, hidden_dim=64/32, lr=0.001, dropout=0.2
- RNA_GO: n_layers=3, hidden_dim=64, lr=0.001, dropout=0.5

\* For RNA_IF, the reported value in "Global Balanced Accuracy" is sequence recovery.

Table A.2: We compare a standard RGCN using the rnaglib's task module with various published results using the TR60/TE18 split. Note: Binding site definitions may vary slightly between models.

| Methods                   | MCC   | AUC   |
|---------------------------|-------|-------|
| Rsite2 (Zeng & Cui, 2016) | 0.010 | 0.474 |
| Rsite (Zeng et al., 2015) | 0.055 | 0.496 |
| RBind (Wang et al., 2018) | 0.141 | 0.540 |
| RNAsite_seq (Su et al., 2021) | 0.160 | 0.641 |
| RNAsite_str (Su et al., 2021) | 0.185 | 0.695 |
| RNAsite (Su et al., 2021) | 0.186 | 0.703 |
| rnaglib RNA-Site          | 0.113 | 0.606 |

Table A.3: Sequence recovery scores for RNA inverse folding models. We use a standard two layer RGCN part of rnaglib's task module on the dataset and split published by Joshi et al. (2024)

| Method                         | Sequence Recovery |
|--------------------------------|-------------------|
| gRNAde (Joshi et al., 2024)    | 0.568             |
| Rosetta (Leman et al., 2020b)  | 0.450             |
| RDesign (Tan et al., 2023)     | 0.430             |
| FARNA (Alam et al., 2017)      | 0.321             |
| ViennaRNA (Lorenz et al., 2011)| 0.269             |
| rnaglib RNA-IF                 | 0.410             |
![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image365-page4.png)

**Caption:** 1. RNA-GO RNA MULTI-CLASS (5) 349-75-75 2. RNA-IF RESIDUE MULTI-CLASS (4) 1700-448-581 gRNAde (Joshi et al., 2024) 11183-528-235 3. RNA-CM RESIDUE BINARY 138-29-30 4. RNA-PROT RESIDUE BINARY 881-189-189 6. RNA-SITE RESIDUE BINARY 157-34-33 RNASite (Su et al., 2021) 53-6-17 6. RNA-LIGAND POCKET MULTI-CLASS (3) 203-43-44 7. RNA-VS (CARVAJAL-PATI ˜NO ET AL., 2025) POCKET REGRESSION 304-34-65  (Context): Effective function prediction models have the ca- pacity to uncover new structure-function connections. The Gene Ontology (GO) (Ashburner et al., 2000) was devel- oped to associate a function to a gene, and thus, indirectly to the RNA or protein it encodes. It resulted in discrete functional categories called GO terms. Predicting the GO term from a protein structure was proposed as a task in (Gligorijevi´c et al., 2021) and has since been regularly used as a benchmark task. A mapping between RNA sequences and GO terms is available in Rfam (Griffiths-Jones et al., 2003; Ontiveros-Palacios et al., 2025), a database of non- coding RNA families which are manually annotated with GO terms. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image382-page4.png)

**Caption:** 1. RNA-GO RNA MULTI-CLASS (5) 349-75-75 2. RNA-IF RESIDUE MULTI-CLASS (4) 1700-448-581 gRNAde (Joshi et al., 2024) 11183-528-235 3. RNA-CM RESIDUE BINARY 138-29-30 4. RNA-PROT RESIDUE BINARY 881-189-189 6. RNA-SITE RESIDUE BINARY 157-34-33 RNASite (Su et al., 2021) 53-6-17 6. RNA-LIGAND POCKET MULTI-CLASS (3) 203-43-44 7. RNA-VS (CARVAJAL-PATI ˜NO ET AL., 2025) POCKET REGRESSION 304-34-65  (Definition): This is a multi-label classification task where input RNA is mapped to a sequence of 5 possible labels representing molecular functions. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image383-page4.png)

**Caption:** 1. RNA-GO RNA MULTI-CLASS (5) 349-75-75 2. RNA-IF RESIDUE MULTI-CLASS (4) 1700-448-581 gRNAde (Joshi et al., 2024) 11183-528-235 3. RNA-CM RESIDUE BINARY 138-29-30 4. RNA-PROT RESIDUE BINARY 881-189-189 6. RNA-SITE RESIDUE BINARY 157-34-33 RNASite (Su et al., 2021) 53-6-17 6. RNA-LIGAND POCKET MULTI-CLASS (3) 203-43-44 7. RNA-VS (CARVAJAL-PATI ˜NO ET AL., 2025) POCKET REGRESSION 304-34-65  (Definition): This is a multi-label classification task where input RNA is mapped to a sequence of 5 possible labels representing molecular functions. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image471-page5.png)

**Caption:** Benchmark for RNA 3D Structure Modeling  (Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict which, if any, of its residues are chemically modified. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image473-page5.png)

**Caption:** Benchmark for RNA 3D Structure Modeling  (Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict which, if any, of its residues are chemically modified. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image505-page5.png)

**Caption:** (Processing): To build this task, we start from the whole dataset, partition it into connected components and apply our size filter. Then, we filter for systems that include modified residues, relying on PDB annotations that flag such modified residues. We apply our default redundancy removal and splitting strategies, which results in 185 data points. Performance is evaluated with accuracy and AuROC.  (Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict whether a protein residue is closer than 8 ˚A to any of its residues. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image507-page5.png)

**Caption:** (Processing): To build this task, we start from the whole dataset, partition it into connected components and apply our size filter. Then, we filter for systems that include modified residues, relying on PDB annotations that flag such modified residues. We apply our default redundancy removal and splitting strategies, which results in 185 data points. Performance is evaluated with accuracy and AuROC.  (Definition): This is a residue-level, binary classification task where given an RNA structure, we aim to predict whether a protein residue is closer than 8 ˚A to any of its residues. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image538-page5.png)

**Caption:**  4.4. RNA-Prot: Biological complex modeling. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image540-page5.png)

**Caption:**  4.4. RNA-Prot: Biological complex modeling. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image930-page6.png)

**Caption:** atom neighbors, ensuring that the binding is modulated by RNA only. We use the default redundancy removal and splitting, resulting in 458 systems. The predictions are evaluated using accuracy and AuROC.  (Definition): This is a binding site-level, multi-class classi- fication task where the structure of an RNA binding site is classified according to the partner it accommodates. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image930-page6.png)

**Caption:** Benchmark for RNA 3D Structure Modeling  (Definition): This is a binding site-level, multi-class classi- fication task where the structure of an RNA binding site is classified according to the partner it accommodates. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image1021-page12.png)

**Caption:** A.1. Datasets and general preprocessing  (a) Whole RNAs (b) RNA chunks (c) RNA chunks after size filtering 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image1025-page12.png)

**Caption:** A.1. Datasets and general preprocessing  (a) Whole RNAs (b) RNA chunks (c) RNA chunks after size filtering 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image1029-page12.png)

**Caption:** A.1. Datasets and general preprocessing  (a) Whole RNAs (b) RNA chunks (c) RNA chunks after size filtering 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image1049-page13.png)

**Caption:** and 0.70 give the same plot since CD-HIT similarity values are strongly concentrated around 0.5 and 1. We finally choose 0.90. Figure A.2b displays, for 4 different structure-based redundancy removal thresholds (each with a different color), the scatter plot of the number of clusters obtained based on the structure-based splitting clustering threshold. In all cases, the structure-based redundancy removal is performed after a sequence-based redundancy removal using a 0.90 threshold. In all the experiments, the splitting clustering is performed based on US-align similarity (structural similarity).  (a) Using sequence-based redundancy removal (b) Using structure-based redundancy removal 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf2\image1052-page13.png)

**Caption:** and 0.70 give the same plot since CD-HIT similarity values are strongly concentrated around 0.5 and 1. We finally choose 0.90. Figure A.2b displays, for 4 different structure-based redundancy removal thresholds (each with a different color), the scatter plot of the number of clusters obtained based on the structure-based splitting clustering threshold. In all cases, the structure-based redundancy removal is performed after a sequence-based redundancy removal using a 0.90 threshold. In all the experiments, the splitting clustering is performed based on US-align similarity (structural similarity).  (a) Using sequence-based redundancy removal (b) Using structure-based redundancy removal 

