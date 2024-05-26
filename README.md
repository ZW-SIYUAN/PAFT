# Anonymouscode for PAFT

This GitHub page presents the Python implementation of the proposed PAFT method.  We included the six publicly available datasets that were utilized for evaluation in the paper.

## Dependency and Baseline 

CTGAN ```https://github.com/sdv-dev/CTGAN```

TabSyn ```https://github.com/amazon-science/tabsyn```

GReaT ```https://github.com/kathrinse/be_great```

## FD Discovery
Any column dependency algorithm can be fit here. In the paper, the "A Hybrid Approach to Functional Dependency Discovery" is employed. However, any column dependency discovery, even customized, works.

HyFD ```https://github.com/codocedo/hyfd```

Run ```python3 hyfd/hyfd.py datasets/*.csv```


## PAFT 
Run paft with two phases:

Step1. Run ```python paft_fd_distilation_and_optimization.py```

Step2. Run ```python paft_fine_tuning.py```