import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
# from torch_geometric.utils import one_hot, scatter
import numpy as np
import pickle

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

atomrefs = {
    6: [0.0, 0.0, 0.0, 0.0, 0.0],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0.0, 0.0, 0.0, 0.0, 0.0],
}


class CompeteData(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        split: str = "train",
    ):
        self.type2index = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35}
        self.index2charge = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35])
        splitname2index = {"train": 0, "val": 1, "test": 2}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[splitname2index[split]]
        )

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ["train.npz", "valid.npz", "test.npz"]

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> str:
        return ["train.pt", "valid.pt", "test.pt"]

    def process(self):
        target_keys = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "U0",
            "U",
            "H",
            "G",
            "Cv",
            "U0_thermo",
            "U_thermo",
            "H_thermo",
            "G_thermo",
            "A",
            "B",
            "C",
        ]

        def _make_data_instance(
            # _index: torch.Tensor,
            _num_atoms: torch.Tensor,
            _charges: torch.Tensor,
            _positions: torch.Tensor,
            # _targets: torch.Tensor,
        ):
            """
            Args:
                _index: scalar
                _num_atoms: scalar
                _charges : [K]
                _positions: [K, 3]
                _targets: [19]
            """
            _charges, _positions = _charges[:_num_atoms], _positions[:_num_atoms, :]
            _type_onehot = (
                _charges.reshape(-1, 1) == self.index2charge.reshape(1, -1)
            ).to(torch.float32)
            _charges = _charges.reshape(-1, 1).to(torch.float32)
            return Data(
                x=_type_onehot,
                charges=_charges,
                pos=_positions,
                # y=_targets,
                # idx=_index.reshape(-1),
            )

        def _load_datalist_from_npz(path: str) -> List[Data]:
            _data_dict = np.load(path)
            # targets = torch.concat(
            #     [
            #         torch.tensor(_data_dict[k], dtype=torch.float32).reshape(-1, 1)
            #         for k in target_keys
            #     ],
            #     dim=-1,
            # )
            # index = torch.tensor(_data_dict["index"], dtype=torch.int64)
            num_atoms = torch.tensor(_data_dict["natoms"], dtype=torch.int32)
            charges = torch.tensor(_data_dict["charges"], dtype=torch.int32)  # [N, K]
            positions = torch.tensor(
                _data_dict["coordinates"], dtype=torch.float32
            )  # [N, K, 3]
            assert torch.all(torch.sum(charges != 0, dim=1) == num_atoms)
            assert torch.all(
                torch.any(
                    torch.sum(positions != 0, dim=1)
                    == torch.reshape(num_atoms, [-1, 1]),
                    dim=1,
                )
            )
            datalist = []
            for i in range(num_atoms.shape[0]):
                datalist.append(
                    _make_data_instance(
                    num_atoms[i], charges[i], positions[i]
                    # , targets[i]
                    )
                )
            return datalist

        for _f_raw, _f_processed in zip(self.raw_file_names, self.processed_file_names):
            print(f"Processing {_f_raw} -> {_f_processed}")
            raw_file = osp.join(self.raw_dir, _f_raw)
            processed_file = osp.join(self.processed_dir, _f_processed)
            data_list = _load_datalist_from_npz(raw_file)

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), processed_file)
