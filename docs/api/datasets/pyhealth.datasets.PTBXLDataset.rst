pyhealth.datasets.PTBXLDataset
===============================

Overview
--------

PTB-XL is a large publicly available 12-lead ECG dataset collected at the
Physikalisch-Technische Bundesanstalt (PTB) between 1989 and 1996. It contains
21,837 clinical ECG recordings (10 seconds, 500 Hz) from 18,869 patients. Each
recording is annotated with SCP-ECG statements which are mapped to five
diagnostic superclasses: NORM, MI (Myocardial Infarction), HYP (Hypertrophy),
STTC (ST/T-Change), and CD (Conduction Disturbance).

The :class:`~pyhealth.datasets.PTBXLDataset` exposes the four pathological
superclasses as independent binary classification tasks, directly replicating
the PTB-XL evaluation protocol used in *Raghu et al., CHIL 2022 (TaskAug)*.

**Data access:** Download from `PhysioNet PTB-XL 1.0.3
<https://physionet.org/content/ptb-xl/1.0.3/>`_ — no registration required.

**Reference:**

   Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. (2020). PTB-XL, a large
   publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154.
   https://doi.org/10.1038/s41597-020-0495-6

API Reference
-------------

.. autoclass:: pyhealth.datasets.PTBXLDataset
    :members:
    :undoc-members:
    :show-inheritance:

See Also
--------

- :mod:`pyhealth.tasks.ecg_classification_ptbxl` — ECG binary classification
  task functions (MI, HYP, STTC, CD)
- :mod:`pyhealth.models.taskaug_resnet` — ResNet1D backbone, TaskAugPolicy,
  and BiLevelTrainer

toctree entry for ``docs/api/datasets.rst``::

    pyhealth.datasets.PTBXLDataset
